"""
Training callbacks for the S2G pipeline.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import wandb
from transformers import (
    EarlyStoppingCallback, 
    PreTrainedTokenizerBase,
    AutoModelForSeq2SeqLM,
    TrainerCallback, 
    TrainerControl, 
    TrainerState
)

from s2g.data import S2GCollator
from s2g.evaluation.gold import build_gold_blocks
from s2g.linearisation import extract_triplets, parse_graph

logger = logging.getLogger(__name__)


class StepTrackingCallback(TrainerCallback):
    def __init__(self, collator: S2GCollator) -> None: 
        self.collator = collator
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self.collator.current_step = state.global_step


class GenerateTextSamplesCallback(TrainerCallback):
    """
    Logs a W&B table of gold vs. predicted graphs for a fixed handful of instances.

    Everything that affects generation is taken from the same place the validation
    loop takes it, so the table shows what validation would have seen rather than
    an approximation of it:

    * the sample instances are a ``Subset`` of the **evaluation dataset**, so they
      pass through the same indexing and the same collator;
    * the collator is rebuilt with ``to_eval_mode()`` on every call, exactly as
      ``S2GTrainer.get_eval_dataloader`` does — a collator built once and reused
      would advance its private RNG between calls and quietly vary the sampled
      negative schema from one table to the next;
    * beam count and generation length are read from ``args`` at call time rather
      than frozen at construction, so they cannot drift from the values the
      trainer actually generates with;
    * autocast follows ``args.bf16`` / ``args.fp16``. Under mixed precision the
      parameters stay ``float32``, so keying autocast off the parameter dtype (as
      this callback used to) silently generated in full precision while validation
      generated under autocast.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        variant: str,
        sample_dataset: Any,
        collator: S2GCollator,
        interval: int = 1_000,
    ) -> None:
        if variant not in {'re', 'boundary_re', 'joint', 'boundary_joint'}:
            raise ValueError(f"Unknown variant {variant!r}.")

        self.tokenizer = tokenizer
        self.sample_dataset = sample_dataset
        self.collator = collator
        self.variant = variant
        self.tok = collator.tok
        self.interval = interval
        self.last_logged = -1

    @property
    def instances(self) -> List[Dict]:
        """The sample instances, in dataset order."""
        return [self.sample_dataset[i] for i in range(len(self.sample_dataset))]

    def on_step_end(
            self, 
            args, 
            state: TrainerState, 
            control: TrainerControl, 
            model: AutoModelForSeq2SeqLM = None, 
            **kwargs
        ) -> None:
        if not state.is_world_process_zero or \
            state.global_step in {0, self.last_logged} or state.global_step % self.interval != 0: 
            return
            
        self.last_logged = state.global_step
        if model is None:
            logger.warning("GenerateTextSamplesCallback: no model at step %d.", state.global_step)
            return
        
        try:
            self.log_samples(model, args, state, is_initial=False)
        except Exception:
            logger.exception("GenerateTextSamplesCallback failed at step %d.", state.global_step)

    def on_train_begin(
            self, 
            args, 
            state: TrainerState, 
            control: TrainerControl, 
            model: AutoModelForSeq2SeqLM = None, 
            **kwargs
        ) -> None:
        if not state.is_world_process_zero: 
            return
            
        if model is None: 
            logger.warning("GenerateTextSamplesCallback: no model at train begin.")
            return
            
        try:
            self.log_samples(model, args, state, is_initial=True)
            self.last_logged = 0
        except Exception:
            logger.exception("GenerateTextSamplesCallback failed at train begin.")

    def generation_context(self, args, device: torch.device, model: AutoModelForSeq2SeqLM):
        """
        Autocast context matching what the trainer uses at validation.

        Under mixed precision the parameters stay ``float32``, so the dtype has to
        come from ``args``; the parameter dtype is only consulted for a model that
        was genuinely cast to half precision.
        """
        if device.type != 'cuda':
            return contextlib.nullcontext()

        if args is not None and getattr(args, 'bf16', False):
            return torch.autocast(device.type, torch.bfloat16)
        if args is not None and getattr(args, 'fp16', False):
            return torch.autocast(device.type, torch.float16)

        param_dtype = next(model.parameters()).dtype
        if param_dtype in {torch.bfloat16, torch.float16}:
            return torch.autocast(device.type, param_dtype)
        return contextlib.nullcontext()

    @staticmethod
    def format_entities(blocks: List[Dict], include_types: bool) -> str:
        """
        One line per entity — heads and reconciled tails alike.

        A missing type prints as a bare mention rather than ``[None]``: an untyped
        block in a typed variant means the model omitted the type, and rendering
        that as the literal string ``None`` reads as a predicted type.
        """
        if not blocks:
            return "(none)"

        lines = []
        for block in blocks:
            text = block.get('text', '')
            ent_type = block.get('type')
            lines.append(f"{text} [{ent_type}]" if include_types and ent_type else text)
        return "\n".join(lines)

    def warn_on_gold_drift(self, instance: Dict, label_blocks: List[Dict]) -> None:
        """
        Flag a mismatch between the gold parsed back out of the labels and the gold
        the evaluator builds from the annotation.

        The table shows the label round trip, since that is literally what the model
        is trained against — but that path is lossy where the evaluator's is not
        (a target truncated at ``max_target_length`` loses its tail). Surfacing the
        disagreement keeps a truncated target from looking like a model error.
        """
        try:
            dataset_blocks = build_gold_blocks(instance, self.variant, self.collator.dedup)
        except Exception:
            logger.debug("Could not build dataset gold for drift check.", exc_info=True)
            return

        summarise = lambda blocks: (
            sorted((b.get('text', ''), b.get('type')) for b in blocks),
            sorted(extract_triplets(blocks, include_types=True)),
        )
        if summarise(label_blocks) != summarise(dataset_blocks):
            logger.warning(
                "Sample gold drift for %r: the label round trip yields %d entities, the "
                "annotation %d. Usually a target truncated at max_target_length.",
                instance.get('text', '')[:60], len(label_blocks), len(dataset_blocks),
            )

    def log_samples(self, model: AutoModelForSeq2SeqLM, args=None, state: TrainerState = None, is_initial: bool = False) -> None:
        if wandb.run is None:
            return

        instances = self.instances
        if not instances:
            logger.warning("GenerateTextSamplesCallback: empty sample dataset, nothing to log.")
            return

        # Rebuilt per call, mirroring ``S2GTrainer.get_eval_dataloader``.
        eval_collator = self.collator.to_eval_mode()
        batch = eval_collator(instances)
        device = next(model.parameters()).device

        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attn_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # Beam count and length come from the trainer's own generation settings.
        num_beams = getattr(args, 'generation_num_beams', None) or 1
        max_length = getattr(args, 'generation_max_length', None) or eval_collator.cfg['max_target_length']
        ctx = self.generation_context(args, device, model)

        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode(), ctx:
                gen_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attn_mask,
                    'num_beams': num_beams,
                    'max_length': max_length,
                }

                generated_ids = (model.module if hasattr(model, 'module') else model).generate(**gen_kwargs) # model.module for DDP
        finally:
            # Restore whatever mode the model arrived in: ``on_train_begin`` can fire
            # before the trainer has put the model into training mode.
            model.train(was_training)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        g_ids = labels.clone()
        g_ids.masked_fill_(g_ids == -100, pad_id)
        pred_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        gold_texts = self.tokenizer.batch_decode(g_ids, skip_special_tokens=False)

        # Get prompts
        prompts = []
        for seq in input_ids:
            seq_nopad = seq[seq != pad_id]
            prompts.append(self.tokenizer.decode(seq_nopad, skip_special_tokens=False))

        specials_to_remove = [t for t in (self.tokenizer.pad_token, self.tokenizer.eos_token, self.tokenizer.bos_token) if t]
        rows = []
        
        cols = [
            'Source', 
            'Encoder Input', 
            'Predicted Entities', 
            'Gold Entities', 
            'Predicted Triplets', 
            'Gold Triplets', 
            'Predicted Graph', 
            'Gold Graph'
        ]
        
        include_types = self.variant in {'joint', 're'}

        for i, inst in enumerate(instances):
            p_graph, g_graph = pred_texts[i], gold_texts[i]

            for tok in specials_to_remove:
                p_graph = p_graph.replace(tok, "")
                g_graph = g_graph.replace(tok, "")

            p_graph = " ".join(p_graph.split())
            g_graph = " ".join(g_graph.split())

            # ``parse_graph`` reconciles tails via ``resolve_tail_entities``, so these
            # entity lists carry both heads and tail-only mentions — matching what the
            # evaluator scores.
            p_ent, _ = parse_graph(p_graph, tok=self.tok)
            g_ent, _ = parse_graph(g_graph, tok=self.tok)

            self.warn_on_gold_drift(inst, g_ent)

            row = [inst['text'], prompts[i]]

            p_e = self.format_entities(p_ent, include_types)
            g_e = self.format_entities(g_ent, include_types)

            # Format triplets
            p_triplets = extract_triplets(p_ent, include_types=include_types)
            g_triplets = extract_triplets(g_ent, include_types=include_types)
            p_t = "\n".join([f"{t[0]} --[{t[1]}]--> {t[2]}" for t in p_triplets]) if p_triplets else "(none)"
            g_t = "\n".join([f"{t[0]} --[{t[1]}]--> {t[2]}" for t in g_triplets]) if g_triplets else "(none)"

            row.extend([p_e, g_e, p_t, g_t, p_graph, g_graph])
            rows.append(row)

        wandb.log(
            {f"samples/{self.variant}": wandb.Table(columns=cols, data=rows)}, 
            step=0 if is_initial else state.global_step
        )
        logger.info("Logged %d %s samples to W&B at step %d.", len(rows), self.variant, 0 if is_initial else state.global_step)


class PeriodicCheckpointCallback(TrainerCallback):
    """
    Optional safety-net checkpoints, on top of the ones the trainer already writes
    at each validation check.

    ``every_n_steps=None`` (or any non-positive value) disables the extra saves and
    leaves checkpointing entirely to ``save_strategy`` / ``save_steps``. Writing
    ``run_metadata.json`` is deliberately **not** tied to that switch: it is keyed to
    ``on_save``, so it tracks every checkpoint the run produces however it was
    triggered. Folding it into the forced-save branch, as it used to be, meant that
    turning the extra saves off silently disabled W&B run resumption too.
    """

    def __init__(self, output_dir: str, every_n_steps: Optional[int] = None, wandb_run_id: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir)
        self.every_n_steps = every_n_steps if every_n_steps and every_n_steps > 0 else None
        self.wandb_run_id = wandb_run_id
        self.last_saved = -1

        if self.every_n_steps is None:
            logger.info("Periodic checkpointing disabled; saving on validation checks only.")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if self.every_n_steps is None:
            return
        if state.global_step in {0, self.last_saved} or state.global_step % self.every_n_steps != 0:
            return

        self.last_saved, control.should_save = state.global_step, True

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        """Record the W&B run id against the latest checkpoint, for resumption."""
        if not (self.wandb_run_id and state.is_world_process_zero):
            return

        m_path = self.output_dir / "run_metadata.json"
        m_path.parent.mkdir(parents=True, exist_ok=True)
        with open(m_path, 'w', encoding='utf-8') as f:
            json.dump({'wandb_run_id': self.wandb_run_id, 'last_step': state.global_step}, f, indent=2)


class S2GEarlyStoppingCallback(EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        """
        Override to prevent early stopping counter from incrementing if the best metric so far is still <= 0.0
        """
        super().check_metric_value(args, state, control, metric_value)
        if args.greater_is_better and (state.best_metric is None or state.best_metric <= 0.0):
            self.early_stopping_patience_counter = 0


def load_run_metadata(output_dir: str) -> Optional[Dict[str, Any]]:
    m_path = Path(output_dir) / "run_metadata.json"
    if not m_path.exists():
        logger.warning("No run metadata at %s; starting a fresh W&B run.", m_path)
        return None

    with open(m_path, 'r', encoding='utf-8') as f:
        return json.load(f)
