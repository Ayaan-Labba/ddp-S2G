"""
Train a REBEL-format relation extraction model using the S2G data pipeline.

Linearisation
-------------
Identical to REBEL's conll04_typed.py — typed, grouped by head entity:

    First relation in a head-group:
        <triplet> HEAD_TEXT <HEAD_TYPE> TAIL_TEXT <TAIL_TYPE> REL_TEXT

    Additional relations with the same head:
        <HEAD_TYPE> TAIL2_TEXT <TAIL2_TYPE> REL2_TEXT

    Next head-group:
        <triplet> HEAD2_TEXT <HEAD2_TYPE> ...

Entity type tags (<person>, <organization>, …) are derived automatically
from the entity schema file and added as new vocabulary tokens.

Encoder input: plain source text — no SSI schema prompts.

Backbone
--------
Any HuggingFace AutoModelForSeq2SeqLM checkpoint.  Two presets:
    facebook/bart-large    (REBEL's original backbone)
    google/flan-t5-base    (lighter alternative)

Usage
-----
python -m s2g.scripts.train_rebel \\
    --config configs/tasks/rebel_re/conll04.yaml

# Override backbone for a Flan-T5 run:
python -m s2g.scripts.train_rebel \\
    --config configs/tasks/rebel_re/conll04.yaml \\
    model.name=google/flan-t5-base \\
    data.output_dir=outputs/rebel_re/t5-conll04
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from s2g.data import S2GDataset
from s2g.evaluation.metrics import compute_metrics_for_task
from s2g.scripts.config_utils import load_schema, load_entity_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

@dataclass
class _ModelCfg:
    name: str = "facebook/bart-large"
    pretrained_checkpoint: Optional[str] = None

@dataclass
class _TokCfg:
    max_source_length: int = 1024
    max_target_length: int = 128

@dataclass
class _OptCfg:
    optim: str = "adamw_torch"
    lr: float = 5e-5
    weight_decay: float = 0.01   # REBEL uses 0.01, not 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

@dataclass
class _SchCfg:
    type: str = "linear"         # REBEL CoNLL04: linear decay with warmup
    warmup_steps: int = 100

@dataclass
class _TrainCfg:
    max_steps: int = 1000
    batch_size: int = 8
    gradient_acc_steps: int = 4  # effective batch = 32
    gradient_clip_value: float = 1.0
    gradient_checkpointing: bool = False
    precision: str = "bf16"
    seed: int = 42

@dataclass
class _ValCfg:
    check_interval: int = 50
    percent_check: float = 1.0
    batch_size: int = 32
    early_stopping_patience: int = 10
    early_stopping_metric: str = "strict_f1"

@dataclass
class _GenCfg:
    num_beams: int = 3
    length_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = False

@dataclass
class _CkptCfg:
    save_top_k: int = 1
    resume_from: Optional[str] = None

@dataclass
class _WandbCfg:
    project: str = "rebel-re"
    entity: Optional[str] = None
    run_name: Optional[str] = None

@dataclass
class _DataCfg:
    data_dir: Optional[str] = None
    schema_file: Optional[str] = None
    entity_schema_file: Optional[str] = None
    output_dir: str = "outputs/rebel_re/run"

@dataclass
class _HwCfg:
    num_workers: int = 4
    persistent_workers: bool = True
    gpu_ids: Optional[List[int]] = None

@dataclass
class REBELConfig:
    model:        _ModelCfg  = field(default_factory=_ModelCfg)
    tokenization: _TokCfg    = field(default_factory=_TokCfg)
    optimizer:    _OptCfg    = field(default_factory=_OptCfg)
    scheduler:    _SchCfg    = field(default_factory=_SchCfg)
    train:        _TrainCfg  = field(default_factory=_TrainCfg)
    validation:   _ValCfg    = field(default_factory=_ValCfg)
    generation:   _GenCfg    = field(default_factory=_GenCfg)
    checkpoint:   _CkptCfg   = field(default_factory=_CkptCfg)
    wandb:        _WandbCfg  = field(default_factory=_WandbCfg)
    data:         _DataCfg   = field(default_factory=_DataCfg)
    hardware:     _HwCfg     = field(default_factory=_HwCfg)


def _load_cfg() -> DictConfig:
    """Load config from --config flag + OmegaConf dotlist overrides."""
    args = sys.argv[1:]
    yaml_path, remaining = None, []
    i = 0
    while i < len(args):
        if args[i] in ("--config", "-c") and i + 1 < len(args):
            yaml_path = args[i + 1]; i += 2
        elif args[i].startswith("--config="):
            yaml_path = args[i].split("=", 1)[1]; i += 1
        else:
            remaining.append(args[i]); i += 1

    cfg = OmegaConf.structured(REBELConfig)
    if yaml_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(yaml_path))
    if remaining:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(remaining))
    return cfg


# ---------------------------------------------------------------------------
# Special-token helpers
# ---------------------------------------------------------------------------

def _make_type_tag(entity_type: str) -> str:
    """Convert an entity type string to its special token, e.g. 'person' → '<person>'."""
    return f"<{entity_type.lower().replace(' ', '_')}>"


def _add_rebel_tokens(
    tokenizer, model, entity_schema: List[str]
) -> Tuple[str, Dict[str, str]]:
    """
    Register <triplet> and one type tag per entity type as new vocabulary tokens.
    Warm-start each new embedding with the mean of semantically related subwords.

    Returns
    -------
    triplet_token : str
    type_tag_map  : Dict[entity_type_str -> type_tag_str]
    """
    triplet_token = "<triplet>"
    type_tag_map: Dict[str, str] = {t: _make_type_tag(t) for t in entity_schema}
    all_new = [triplet_token] + list(type_tag_map.values())

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": all_new})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Added %d new special tokens: %s", num_added, all_new)

        # Warm-start: initialise new token embeddings from related existing subwords.
        # For type tags, use the type name itself as the seed text.
        warm_map: Dict[str, str] = {triplet_token: "."}
        warm_map.update({tag: etype for etype, tag in type_tag_map.items()})

        with torch.no_grad():
            in_emb  = model.get_input_embeddings().weight
            out_mod = model.get_output_embeddings()
            out_emb = out_mod.weight if out_mod is not None else None

            for tok_str, seed_text in warm_map.items():
                new_id   = tokenizer.convert_tokens_to_ids(tok_str)
                seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
                if seed_ids and new_id != tokenizer.unk_token_id:
                    in_emb[new_id].copy_(in_emb[seed_ids].mean(dim=0))
                    if out_emb is not None and out_emb.data_ptr() != in_emb.data_ptr():
                        out_emb[new_id].copy_(out_emb[seed_ids].mean(dim=0))

    return triplet_token, type_tag_map


# ---------------------------------------------------------------------------
# REBEL-style collator
# ---------------------------------------------------------------------------

class REBELCollator:
    """
    Collates S2G JSONL instances into REBEL's typed linearisation format.

    Target format (per instance):
        <triplet> HEAD <HEAD_TYPE> TAIL <TAIL_TYPE> REL
        <HEAD_TYPE> TAIL2 <TAIL2_TYPE> REL2     ← same head, next tail
        <triplet> HEAD2 <HEAD2_TYPE> …           ← new head

    Instances with no relations produce an empty target string (matching
    REBEL's behaviour: the model is trained to predict nothing).
    """

    def __init__(
        self,
        tokenizer,
        type_tag_map: Dict[str, str],  # entity_type_str → "<type>"
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self._tokenizer         = tokenizer
        self._type_tag_map      = type_tag_map
        self._max_src           = max_source_length
        self._max_tgt           = max_target_length
        self._unk_tag           = "<unk_type>"

    def _type_tag(self, entity_type: Optional[str]) -> str:
        if not entity_type:
            return self._unk_tag
        return self._type_tag_map.get(entity_type, _make_type_tag(entity_type))

    def linearize(self, instance: Dict) -> str:
        """
        Build the REBEL typed target string for one instance.

        Groups relations by head entity (keyed on offset), sorted by
        head start position; within each group, tails are sorted by
        tail start position — identical to REBEL's conll04_typed.py.
        """
        relations = instance.get("relations", [])
        if not relations:
            return ""

        # Sort by (head_start, tail_start) so grouping matches REBEL's prev_head logic
        def _sort_key(r):
            h_off = r["head"].get("offset", [0, 0])
            t_off = r["tail"].get("offset", [0, 0])
            return (int(h_off[0]), int(t_off[0]))

        sorted_rels = sorted(relations, key=_sort_key)

        parts: List[str] = []
        prev_head_key: Optional[Tuple[int, int]] = None

        for rel in sorted_rels:
            h     = rel["head"]
            t     = rel["tail"]
            h_off = tuple(int(x) for x in h.get("offset", [0, 0]))
            h_tag = self._type_tag(h.get("type"))
            t_tag = self._type_tag(t.get("type"))
            rel_t = rel.get("type", "")

            if prev_head_key == h_off:
                # Same head — only add <HEAD_TYPE> TAIL <TAIL_TYPE> REL
                parts.append(f"{h_tag} {t['text']} {t_tag} {rel_t}")
            elif prev_head_key is None:
                # First triplet in the sequence
                parts.append(f"<triplet> {h['text']} {h_tag} {t['text']} {t_tag} {rel_t}")
                prev_head_key = h_off
            else:
                # New head
                parts.append(f"<triplet> {h['text']} {h_tag} {t['text']} {t_tag} {rel_t}")
                prev_head_key = h_off

        return " ".join(parts)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        encoder_inputs = [inst["text"] for inst in batch]
        decoder_targets = [self.linearize(inst) for inst in batch]

        model_inputs = self._tokenizer(
            encoder_inputs,
            max_length=self._max_src,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        label_enc = self._tokenizer(
            decoder_targets,
            max_length=self._max_tgt,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        label_ids = label_enc["input_ids"].clone()
        label_ids.masked_fill_(label_ids == self._tokenizer.pad_token_id, -100)

        return {
            "input_ids":      model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels":         label_ids,
        }


# ---------------------------------------------------------------------------
# REBEL typed parser
# ---------------------------------------------------------------------------

def parse_rebel_typed(
    text: str, type_tags: Set[str]
) -> List[Dict[str, str]]:
    """
    Parse a REBEL typed linearisation string back to a list of relation dicts.

    Each dict has keys: head, head_type, type (relation), tail, tail_type.

    Mirrors REBEL's extract_triplets_typed() state machine from utils.py.
    """
    triplets: List[Dict[str, str]] = []
    if not text.strip():
        return triplets

    # Strip BART/T5 control tokens before splitting
    text = text.replace("<s>", " ").replace("</s>", " ").replace("<pad>", " ")

    current = "x"            # x=idle, t=subj, s=obj, o=rel
    subject: List[str]    = []
    obj_:    List[str]    = []
    rel:     List[str]    = []
    subject_type  = ""
    object_type   = ""

    def _flush():
        nonlocal subject_type, object_type
        s = " ".join(subject).strip()
        o = " ".join(obj_).strip()
        r = " ".join(rel).strip()
        if s and o and r:
            triplets.append({
                "head":      s,
                "head_type": subject_type,
                "type":      r,
                "tail":      o,
                "tail_type": object_type,
            })

    for token in text.split():
        if token == "<triplet>":
            if current == "o":
                _flush()
            subject.clear(); obj_.clear(); rel.clear()
            subject_type = ""; object_type = ""
            current = "t"

        elif token in type_tags:
            # A type tag acts differently depending on where we are:
            # • after subject text  (current='t') → store subj_type, switch to OBJ
            # • after object  text  (current='s') → store obj_type,  switch to REL
            # • after relation text (current='o') → flush current triple,
            #                                        new obj for SAME subject
            if current == "t":
                subject_type = token
                obj_.clear(); object_type = ""; rel.clear()
                current = "s"
            elif current == "s":
                object_type = token
                rel.clear()
                current = "o"
            elif current == "o":
                _flush()
                # New tail for the same head; subject_type becomes this tag
                subject_type = token
                obj_.clear(); object_type = ""; rel.clear()
                current = "s"
            # else: ignore spurious type tag

        else:
            if   current == "t": subject.append(token)
            elif current == "s": obj_.append(token)
            elif current == "o": rel.append(token)

    if current == "o":
        _flush()

    return triplets


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(
    tokenizer,
    type_tags: Set[str],
    rel_schema: List[str],
    entity_schema: List[str],
) -> Callable:
    """
    Returns a compute_metrics function compatible with HF Seq2SeqTrainer.

    Computes:
    - corpus-level boundary F1  (head_text, rel_type, tail_text)
    - corpus-level strict  F1  (head_text, head_type, rel_type, tail_text, tail_type)
    - REBEL-style per-type macro for both
    """
    specials = set()
    for tok in (tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token):
        if tok:
            specials.add(tok)

    def _clean(text: str) -> str:
        for s in specials:
            text = text.replace(s, " ")
        return " ".join(text.split())

    def compute_metrics(eval_preds) -> Dict[str, float]:
        preds, label_ids = eval_preds.predictions, eval_preds.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]

        preds_ids  = np.where(preds     != -100, preds,     tokenizer.pad_token_id)
        labels_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

        pred_strs = tokenizer.batch_decode(preds_ids,  skip_special_tokens=False)
        gold_strs = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)

        pred_parsed = [parse_rebel_typed(_clean(s), type_tags) for s in pred_strs]
        gold_parsed = [parse_rebel_typed(_clean(s), type_tags) for s in gold_strs]

        # Build triplets and quintuples for our updated metrics.py
        pred_trips  = [[(t["head"],                       t["type"], t["tail"]                      ) for t in lst] for lst in pred_parsed]
        gold_trips  = [[(t["head"],                       t["type"], t["tail"]                      ) for t in lst] for lst in gold_parsed]
        pred_quints = [[(t["head"], t["head_type"], t["type"], t["tail"], t["tail_type"]) for t in lst] for lst in pred_parsed]
        gold_quints = [[(t["head"], t["head_type"], t["type"], t["tail"], t["tail_type"]) for t in lst] for lst in gold_parsed]

        return compute_metrics_for_task(
            "re",
            rel_schema=rel_schema,
            entity_schema=entity_schema,
            all_pred_triplets=pred_trips,
            all_gold_triplets=gold_trips,
            all_pred_quintuples=pred_quints,
            all_gold_quintuples=gold_quints,
        )

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = _load_cfg()

    # ── GPU selection ────────────────────────────────────────────────────────
    if cfg.hardware.gpu_ids is not None and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.hardware.gpu_ids))

    set_seed(cfg.train.seed)

    out_dir   = Path(cfg.data.output_dir)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ─────────────────────────────────────────────────────────────────
    try:
        import wandb
        if local_rank == 0:
            wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.wandb.run_name)
    except ImportError:
        logger.info("wandb not installed — skipping W&B logging.")

    # ── Schemas ─────────────────────────────────────────────────────────────
    rel_schema    = load_schema(cfg.data.schema_file)
    entity_schema = load_entity_schema(cfg.data.entity_schema_file)
    logger.info("Relation schema (%d): %s", len(rel_schema), rel_schema)
    logger.info("Entity schema   (%d): %s", len(entity_schema), entity_schema)

    # ── Datasets ─────────────────────────────────────────────────────────────
    data_dir      = Path(cfg.data.data_dir)
    train_dataset = S2GDataset(data_dir / "train.jsonl", seed=cfg.train.seed)
    val_dataset   = S2GDataset(data_dir / "val.jsonl", seed=cfg.train.seed)
    logger.info("Train: %d  Val: %d", len(train_dataset), len(val_dataset))

    # ── Tokenizer + Model ────────────────────────────────────────────────────
    ckpt = cfg.model.pretrained_checkpoint or cfg.model.name
    logger.info("Loading tokenizer and model from %s", ckpt)
    precision_to_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }
    dtype = precision_to_dtype.get(cfg.train.precision, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model     = AutoModelForSeq2SeqLM.from_pretrained(ckpt, dtype=dtype)

    # Ensure model parameters are explicitly cast to the configured precision
    if cfg.train.precision == "fp32":
        model = model.float()
    elif cfg.train.precision == "bf16":
        model = model.to(torch.bfloat16)
    elif cfg.train.precision == "fp16":
        model = model.half()

    # ── Special tokens ───────────────────────────────────────────────────────
    triplet_token, type_tag_map = _add_rebel_tokens(tokenizer, model, entity_schema)
    type_tags: Set[str] = set(type_tag_map.values())
    logger.info("Type-tag map: %s", type_tag_map)

    # ── Set generation config (REBEL-faithful defaults) ──────────────────────
    # Setting these on model.generation_config means Seq2SeqTrainer's
    # predict_with_generate path will pick them up automatically.
    model.generation_config.num_beams           = cfg.generation.num_beams
    model.generation_config.length_penalty       = cfg.generation.length_penalty
    model.generation_config.no_repeat_ngram_size = cfg.generation.no_repeat_ngram_size
    model.generation_config.early_stopping       = cfg.generation.early_stopping

    # Explicitly suppress any forced_bos that some BART checkpoints set,
    # since we are generating a custom-token sequence.
    if hasattr(model.generation_config, "forced_bos_token_id"):
        model.generation_config.forced_bos_token_id = None

    # ── Collator ─────────────────────────────────────────────────────────────
    collator = REBELCollator(
        tokenizer,
        type_tag_map,
        cfg.tokenization.max_source_length,
        cfg.tokenization.max_target_length,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    # We use plain Seq2SeqTrainer (single-task, no pipeline complexity).
    # predict_with_generate=True lets HF handle beam decoding during eval.
    compute_metrics_fn = make_compute_metrics(
        tokenizer, type_tags, rel_schema, entity_schema
    )

    # For the HF lr_scheduler_type:
    # "linear"  → linear decay with warmup (REBEL CoNLL04 default)
    # "cosine"  → cosine annealing
    # "constant_with_warmup" → flat after warmup
    # "inverse_sqrt" is not a native HF type — handled separately if needed
    hf_scheduler = cfg.scheduler.type   # e.g. "linear" or "cosine"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),

        # Steps
        max_steps=cfg.train.max_steps,

        # Batching
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_acc_steps,
        per_device_eval_batch_size=cfg.validation.batch_size,

        # Optimizer
        optim=cfg.optimizer.optim,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        adam_beta1=cfg.optimizer.adam_beta1,
        adam_beta2=cfg.optimizer.adam_beta2,
        adam_epsilon=cfg.optimizer.adam_epsilon,
        max_grad_norm=cfg.train.gradient_clip_value,

        # Scheduler
        lr_scheduler_type=hf_scheduler,
        warmup_steps=cfg.scheduler.warmup_steps,

        # Precision
        fp16=(cfg.train.precision == "fp16"),
        bf16=(cfg.train.precision == "bf16"),
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Generation during eval
        predict_with_generate=True,
        generation_max_length=cfg.tokenization.max_target_length,
        generation_num_beams=cfg.generation.num_beams,

        # Evaluation / checkpointing
        eval_strategy="steps",
        eval_steps=cfg.validation.check_interval,
        save_strategy="steps",
        save_steps=cfg.validation.check_interval,
        save_total_limit=cfg.checkpoint.save_top_k + 1,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.validation.early_stopping_metric,
        greater_is_better=True,

        # Misc
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,
        dataloader_num_workers=cfg.hardware.num_workers,
        dataloader_persistent_workers=cfg.hardware.persistent_workers,
        logging_strategy="steps",
        logging_steps=10,
        report_to="wandb",
        run_name=cfg.wandb.run_name,
        remove_unused_columns=False,
        label_names=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.validation.early_stopping_patience
            )
        ],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from)

    # ── Save ─────────────────────────────────────────────────────────────────
    if trainer.is_world_process_zero():
        best_dir = out_dir / "best_model"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        # Save type-tag map so the model can be loaded for inference
        with open(best_dir / "type_tag_map.json", "w", encoding="utf-8") as f:
            json.dump(type_tag_map, f, indent=2)
        with open(best_dir / "rel_schema.json", "w", encoding="utf-8") as f:
            json.dump(rel_schema, f, indent=2)
        logger.info("Best model saved to %s", best_dir)

    # ── Final val evaluation ─────────────────────────────────────────────────
    val_metrics = trainer.evaluate(
        eval_dataset=S2GDataset(data_dir / "val.jsonl", seed=cfg.train.seed)
    )
    if trainer.is_world_process_zero():
        logger.info("Val metrics: %s", val_metrics)
        with open(out_dir / "val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_path = data_dir / "test.jsonl"
    if test_path.exists() and trainer.is_world_process_zero():
        test_metrics = trainer.evaluate(
            eval_dataset=S2GDataset(test_path, seed=cfg.train.seed),
            metric_key_prefix="test",
        )
        logger.info("Test metrics: %s", test_metrics)
        with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()