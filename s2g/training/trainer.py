"""
S2G custom Seq2SeqTrainer.

Shared by ``pretrain.py`` (REBEL, Experiment 3) and ``finetune.py``
(CoNLL04 / NYT-multi / SciERC, Experiment 1).

``compute_loss`` runs one forward pass per task and sums the per-task
cross-entropy losses before returning a single scalar for the optimizer
step.  ``evaluate()`` replaces the standard HF loop with sequential
pipeline evaluation (Boundary → NER → RE) or independent joint evaluation
(Joint ‖ Joint+).  Two LR schedulers are supported: ``"cosine"``
(Experiment 1) and ``"inverse_sqrt"`` (Experiment 3).

Constructor extra kwargs
------------------------
All standard ``Seq2SeqTrainer`` kwargs are forwarded to the parent.  The
following keyword-only arguments are consumed by ``S2GTrainer``:

``model_variant``       ``"pipeline"`` | ``"joint"``
``tokens``              ``PIPELINE_TOKENS`` or ``JOINT_TOKENS``
``entity_schema``       Sorted list of entity-type strings for SSI.
``rel_schema``          Sorted list of relation-type strings for SSI.
``eval_cfg``            Dict with keys ``max_source_length``,
                        ``max_target_length``, ``eval_batch_size``,
                        ``eval_beams``.
``train_eval_dataset``  Optional :class:`~s2g.data.S2GDataset`
                        for the training-subsample evaluation pass.
``scheduler_type``      ``"inverse_sqrt"`` (default) | ``"cosine"``
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import Seq2SeqTrainer

from s2g.evaluation.metrics import compute_metrics_for_task
from s2g.linearisation import (
    AnyTokens,
    EntityBlock,
    JOINT_TOKENS,
    PIPELINE_TOKENS,
    build_boundary_encoder_input,
    build_joint_encoder_input,
    build_joint_plus_encoder_input,
    build_ner_encoder_input,
    build_re_encoder_input,
    extract_triplets,
    find_all_token_spans,
    parse_sel,
)

logger = logging.getLogger(__name__)

# Task key prefixes present in multi-task batches.
_PIPELINE_TASK_KEYS: Tuple[str, ...] = ("boundary", "ner", "re")
_JOINT_TASK_KEYS:    Tuple[str, ...] = ("joint", "joint_plus")
_ALL_TASK_KEYS:      Tuple[str, ...] = _PIPELINE_TASK_KEYS + _JOINT_TASK_KEYS


# ---- MODULE HELPERS ----


def _unwrap_model(model: Any) -> Any:
    """Return the underlying module when wrapped in DDP."""
    return model.module if hasattr(model, "module") else model


def _clean_decoded(text: str, tokenizer: Any) -> str:
    """Strip pad / eos / bos artefacts from a decoded SEL string."""
    for tok in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token):
        if tok:
            text = text.replace(tok, "")
    return " ".join(text.split())


def _to_spans(
    source_tokens: List[str],
    entities: List[EntityBlock],
) -> List[Tuple[int, int]]:
    """Convert entity block texts to ALL token-index span occurrences.

    Calls :func:`find_all_token_spans` for each entity so that every
    occurrence of a predicted entity text in the source sentence is
    marked in the downstream encoder input — not just the first.
    Duplicate spans are deduplicated while preserving left-to-right
    order; remaining overlaps are resolved by ``_resolve_overlaps``
    inside the augmentation function.
    """
    spans: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for ent in entities:
        for span in find_all_token_spans(source_tokens, ent["text"]):
            if span not in seen:
                seen.add(span)
                spans.append(span)
    return spans


def _to_entity_data(
    source_tokens: List[str],
    entities: List[EntityBlock],
) -> List[Tuple[int, int, str]]:
    """Convert typed entity blocks to ALL ``(start, end, type)`` occurrences.

    Every occurrence of each typed entity text is included, not just the
    first, so the RE encoder input is augmented at all positions where
    the entity appears.  Entities without a predicted type are dropped.
    Duplicate (span, type) pairs are deduplicated.
    """
    data: List[Tuple[int, int, str]] = []
    seen: Set[Tuple[int, int]] = set()
    for ent in entities:
        ent_type = ent.get("type")
        if not ent_type:
            continue
        for span in find_all_token_spans(source_tokens, ent["text"]):
            if span not in seen:
                seen.add(span)
                data.append((span[0], span[1], ent_type))
    return data


def _assemble_re_quintuples(
    re_entities: List[EntityBlock],
    ner_type_map: Dict[str, str],
) -> List[Tuple[str, str, str, str, str]]:
    """Build RE quintuples using entity types from the NER model.

    For the RE task, the model does not output entity types.  Types are
    sourced from the NER model's predictions for the same instance and
    looked up by entity text.

    Args:
        re_entities:  Entity blocks from the RE SEL parse.
        ner_type_map: ``{entity_text: entity_type}`` from the NER parse.

    Returns:
        List of ``(head, head_type, rel_type, tail, tail_type)`` tuples.
    """
    quintuples = []
    for ent in re_entities:
        head_type = ner_type_map.get(ent["text"], "")
        for rel in ent["relations"]:
            tail_type = ner_type_map.get(rel["tail"], "")
            quintuples.append(
                (ent["text"], head_type, rel["type"], rel["tail"], tail_type)
            )
    return quintuples


def _assemble_joint_plus_quintuples(
    entities: List[EntityBlock],
) -> List[Tuple[str, str, str, str, str]]:
    """Build Joint+ quintuples from the model's own entity-type predictions.

    Joint+ outputs both entity types and relations, so types do not need
    to come from an external source.

    Args:
        entities: Entity blocks from the Joint+ SEL parse.

    Returns:
        List of ``(head, head_type, rel_type, tail, tail_type)`` tuples.
    """
    type_map = {e["text"]: e.get("type", "") for e in entities}
    quintuples = []
    for ent in entities:
        for rel in ent["relations"]:
            quintuples.append(
                (
                    ent["text"], ent.get("type", ""),
                    rel["type"],
                    rel["tail"], type_map.get(rel["tail"], ""),
                )
            )
    return quintuples


# ---- S2G TRAINER ----


class S2GTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer subclass for S2G multi-task fine-tuning and pre-training.

    See module docstring for full documentation.
    """

    def __init__(self, **kwargs: Any) -> None:
        # ---- Extract S2G-specific kwargs before calling super(). ----
        self._variant:            str             = kwargs.pop("model_variant")
        self._tokens:             AnyTokens       = kwargs.pop("tokens")
        self._entity_schema:      List[str]       = kwargs.pop("entity_schema", [])
        self._rel_schema:         List[str]       = kwargs.pop("rel_schema", [])
        self._eval_cfg:           Dict[str, Any]  = kwargs.pop("eval_cfg")
        self._train_eval_dataset                  = kwargs.pop("train_eval_dataset", None)
        self._scheduler_type:     str             = kwargs.pop("scheduler_type", "inverse_sqrt")

        super().__init__(**kwargs)

        # Unpack eval config for convenience.
        self._max_src: int  = self._eval_cfg["max_source_length"]
        self._max_tgt: int  = self._eval_cfg["max_target_length"]
        self._eval_bs: int  = self._eval_cfg["eval_batch_size"]
        self._eval_beams: int = self._eval_cfg["eval_beams"]

    # --- Scheduler ---

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Create the learning-rate scheduler.

        ``"inverse_sqrt"`` (default)
            Warmup phase followed by 1/√t decay — used for REBEL
            pre-training (Experiment 3)::

                lr_t = lr × (t / warmup)          if t < warmup
                lr_t = lr × sqrt(warmup / t)       otherwise

        ``"cosine"``
            Linear warmup followed by cosine decay to zero — used for
            benchmark fine-tuning (Experiment 1).
        """
        if self.lr_scheduler is not None:
            return

        if self._scheduler_type == "inverse_sqrt":
            opt    = optimizer or self.optimizer
            warmup = self.args.get_warmup_steps(num_training_steps)

            def _lr_lambda(step: int) -> float:
                step = max(step, 1)
                if step < warmup:
                    return step / max(warmup, 1)
                return math.sqrt(warmup / step)

            self.lr_scheduler = LambdaLR(opt, _lr_lambda)
        else:
            # "cosine" and any other type: delegate to the parent, which
            # reads lr_scheduler_type and warmup_steps from TrainingArguments.
            super().create_scheduler(num_training_steps, optimizer)

    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Sum cross-entropy losses across all tasks present in *inputs*.

        The collator returns one ``(input_ids, attention_mask, labels)``
        triple per task, keyed ``{task_key}_{field}``.  This method
        iterates over the task keys that are present, runs a separate
        forward pass for each, and sums the per-task losses before
        returning a single scalar for the optimizer step.

        The final task's outputs are returned when ``return_outputs=True``
        (consistent with the parent's signature; only the loss is used by
        the Trainer, so the choice of which task's outputs to return is
        inconsequential).
        """
        active_tasks = [
            k for k in _ALL_TASK_KEYS
            if f"{k}_input_ids" in inputs
        ]
        if not active_tasks:
            raise ValueError(
                "compute_loss: no task keys found in inputs.  "
                "Expected at least one of: "
                f"{_ALL_TASK_KEYS}."
            )

        total_loss: Optional[torch.Tensor] = None
        last_outputs = None

        for task_key in active_tasks:
            task_inputs = {
                "input_ids":      inputs[f"{task_key}_input_ids"],
                "attention_mask": inputs[f"{task_key}_attention_mask"],
                "labels":         inputs[f"{task_key}_labels"],
            }
            outputs = model(**task_inputs)
            loss    = outputs.loss
            total_loss   = loss if total_loss is None else total_loss + loss
            last_outputs = outputs

        if return_outputs:
            return total_loss, last_outputs
        return total_loss

    # --- Evaluation entry point ---

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: Any = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs: Any,
    ) -> Dict[str, float]:
        """Run sequential pipeline evaluation and, optionally, train-eval.

        For the **Pipeline model**, tasks are chained:
        Boundary → NER (with Boundary predictions) → RE (with NER predictions).

        For the **Joint model**, Joint and Joint+ are run independently on
        raw text; neither depends on the other.

        The train-eval pass runs on ``self._train_eval_dataset`` (when
        provided) and uses the prefix ``"train_eval_"``.  It is fired on
        every call to ``evaluate()`` made by the standard Trainer cycle
        (``metric_key_prefix == "eval"``).

        Evaluation runs only on the main process (``local_rank == 0``) to
        avoid redundant generation across DDP workers.  Non-main processes
        return an empty dict immediately; they do not idle (no barrier is
        inserted here).
        """
        # Skip non-main processes — generation is self-contained on rank 0.
        if self.args.local_rank > 0:
            return {}

        val_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if val_dataset is None:
            logger.warning("evaluate() called with no eval_dataset; skipping.")
            return {}

        self.model.eval()

        all_metrics: Dict[str, float] = {}

        # ---- Validation set ----
        val_metrics = self._evaluate_dataset(val_dataset, prefix=metric_key_prefix)
        all_metrics.update(val_metrics)

        # ---- Train subsample (only during the standard val cycle) ----
        if (
            self._train_eval_dataset is not None
            and metric_key_prefix == "eval"
        ):
            train_metrics = self._evaluate_dataset(
                self._train_eval_dataset, prefix="train_eval"
            )
            all_metrics.update(train_metrics)

        self.model.train()
        self.log(all_metrics)
        return all_metrics

    # --- Dataset-level dispatch ---

    def _evaluate_dataset(
        self,
        dataset: Any,
        prefix: str,
    ) -> Dict[str, float]:
        """Dispatch to the variant-specific evaluation routine."""
        instances = [dataset[i] for i in range(len(dataset))]
        if self._variant == "pipeline":
            metrics = self._evaluate_pipeline(instances)
        else:
            metrics = self._evaluate_joint(instances)
        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    # --- Pipeline evaluation (Boundary → NER → RE) ---

    def _evaluate_pipeline(
        self,
        instances: List[Dict],
    ) -> Dict[str, float]:
        """Run Boundary → NER → RE sequential evaluation.

        SSI at inference time uses the full entity/relation schema (no
        budget sampling) for deterministic, reproducible evaluation.
        """
        metrics: Dict[str, float] = {}

        # ---- Step 1: Boundary ----
        b_inputs = [
            build_boundary_encoder_input(inst["text"], tok=self._tokens)
            for inst in instances
        ]
        b_per_inst = self._run_generation(b_inputs)

        # ---- Step 2: NER (Boundary predictions → token spans) ----
        n_inputs = []
        for inst, b_ents in zip(instances, b_per_inst):
            spans = _to_spans(inst["tokens"], b_ents)
            n_inputs.append(
                build_ner_encoder_input(
                    self._entity_schema, inst["tokens"], spans,
                    random_order=False, tok=self._tokens,
                )
            )
        n_per_inst = self._run_generation(n_inputs)

        # ---- Step 3: RE (NER predictions → entity+type spans) ----
        r_inputs = []
        ner_type_maps: List[Dict[str, str]] = []
        for inst, n_ents in zip(instances, n_per_inst):
            entity_data = _to_entity_data(inst["tokens"], n_ents)
            r_inputs.append(
                build_re_encoder_input(
                    self._rel_schema, inst["tokens"], entity_data,
                    random_order=False, tok=self._tokens,
                )
            )
            ner_type_maps.append({e["text"]: e.get("type", "") for e in n_ents})
        r_per_inst = self._run_generation(r_inputs)

        # ---- Boundary metrics (prefixed "boundary_") ----
        metrics.update({
            f"boundary_{k}": v for k, v in compute_metrics_for_task(
                "boundary",
                all_pred_entities=[[e["text"] for e in b] for b in b_per_inst],
                all_gold_entities=[[e["text"] for e in inst["entities"]] for inst in instances],
            ).items()
        })

        # ---- NER metrics (prefixed "ner_") ----
        metrics.update({
            f"ner_{k}": v for k, v in compute_metrics_for_task(
                "ner",
                all_pred_entities=[[e["text"] for e in n] for n in n_per_inst],
                all_gold_entities=[[e["text"] for e in inst["entities"]] for inst in instances],
                all_pred_entity_mentions=[
                    [(e["text"], e.get("type") or "") for e in n if e.get("type")]
                    for n in n_per_inst
                ],
                all_gold_entity_mentions=[
                    [(e["text"], e.get("type", "")) for e in inst["entities"]]
                    for inst in instances
                ],
            ).items()
        })

        # ---- RE metrics (prefixed "re_") ----
        pred_triplets   = [extract_triplets(r) for r in r_per_inst]
        pred_quintuples = [
            _assemble_re_quintuples(r, nm)
            for r, nm in zip(r_per_inst, ner_type_maps)
        ]
        gold_triplets   = [
            [(rel["head"]["text"], rel["type"], rel["tail"]["text"])
             for rel in inst["relations"]]
            for inst in instances
        ]
        gold_quintuples = [
            [(rel["head"]["text"], rel["head"].get("type", ""),
              rel["type"],
              rel["tail"]["text"], rel["tail"].get("type", ""))
             for rel in inst["relations"]]
            for inst in instances
        ]
        metrics.update({
            f"re_{k}": v for k, v in compute_metrics_for_task(
                "re",
                all_pred_triplets=pred_triplets,
                all_gold_triplets=gold_triplets,
                all_pred_quintuples=pred_quintuples,
                all_gold_quintuples=gold_quintuples,
            ).items()
        })

        return metrics

    # --- Joint evaluation (Joint ‖ Joint+) ---

    def _evaluate_joint(
        self,
        instances: List[Dict],
    ) -> Dict[str, float]:
        """Run Joint and Joint+ evaluations independently.

        Neither task depends on the other's output.
        """
        metrics: Dict[str, float] = {}

        # ---- Joint ----
        j_inputs = [
            build_joint_encoder_input(
                self._rel_schema, inst["text"],
                random_order=False, tok=self._tokens,
            )
            for inst in instances
        ]
        j_per_inst   = self._run_generation(j_inputs)
        j_triplets   = [extract_triplets(j) for j in j_per_inst]
        gold_triplets = [
            [(rel["head"]["text"], rel["type"], rel["tail"]["text"])
             for rel in inst["relations"]]
            for inst in instances
        ]
        # ---- Joint metrics (prefixed "joint_") ----
        metrics.update({
            f"joint_{k}": v for k, v in compute_metrics_for_task(
                "joint",
                all_pred_triplets=j_triplets,
                all_gold_triplets=gold_triplets,
            ).items()
        })

        # ---- Joint+ ----
        jp_inputs = [
            build_joint_plus_encoder_input(
                self._entity_schema, self._rel_schema, inst["text"],
                random_order=False, tok=self._tokens,
            )
            for inst in instances
        ]
        jp_per_inst = self._run_generation(jp_inputs)

        # ---- Joint+ metrics (prefixed "joint_plus_") ----
        jp_triplets     = [extract_triplets(jp) for jp in jp_per_inst]
        jp_quintuples   = [_assemble_joint_plus_quintuples(jp) for jp in jp_per_inst]
        gold_quintuples = [
            [(rel["head"]["text"], rel["head"].get("type", ""),
              rel["type"],
              rel["tail"]["text"], rel["tail"].get("type", ""))
             for rel in inst["relations"]]
            for inst in instances
        ]
        metrics.update({
            f"joint_plus_{k}": v for k, v in compute_metrics_for_task(
                "joint+",
                all_pred_triplets=jp_triplets,
                all_gold_triplets=gold_triplets,
                all_pred_quintuples=jp_quintuples,
                all_gold_quintuples=gold_quintuples,
                all_pred_entities=[[e["text"] for e in jp] for jp in jp_per_inst],
                all_gold_entities=[[e["text"] for e in inst["entities"]] for inst in instances],
                all_pred_entity_mentions=[
                    [(e["text"], e.get("type") or "") for e in jp if e.get("type")]
                    for jp in jp_per_inst
                ],
                all_gold_entity_mentions=[
                    [(e["text"], e.get("type", "")) for e in inst["entities"]]
                    for inst in instances
                ],
            ).items()
        })

        return metrics

    # --- Generation helper ---

    def _autocast_ctx(self) -> contextlib.AbstractContextManager:
        """Return an autocast context appropriate for the training precision."""
        if self.args.bf16 and self.args.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self.args.fp16 and self.args.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _run_generation(
        self,
        encoder_inputs: List[str],
    ) -> List[List[EntityBlock]]:
        """Tokenise, generate, decode, and parse SEL for a list of inputs.

        Processes *encoder_inputs* in batches of ``eval_batch_size``.

        Args:
            encoder_inputs: Encoder input strings (one per instance).

        Returns:
            Per-instance lists of parsed entity blocks.
        """
        all_entities: List[List[EntityBlock]] = []
        raw_model = _unwrap_model(self.model)

        for i in range(0, len(encoder_inputs), self._eval_bs):
            batch_strs = encoder_inputs[i : i + self._eval_bs]

            tok_out = self.tokenizer(
                batch_strs,
                max_length=self._max_src,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids      = tok_out["input_ids"].to(self.args.device)
            attention_mask = tok_out["attention_mask"].to(self.args.device)

            with torch.inference_mode(), self._autocast_ctx():
                generated = raw_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=self._eval_beams,
                    max_length=self._max_tgt,
                    length_penalty=0.0,
                    early_stopping=False,
                    no_repeat_ngram_size=0,
                )

            decoded = self.tokenizer.batch_decode(
                generated, skip_special_tokens=False
            )
            for text in decoded:
                text = _clean_decoded(text, self.tokenizer)
                entities, _ = parse_sel(text, tok=self._tokens)
                all_entities.append(entities)

        return all_entities