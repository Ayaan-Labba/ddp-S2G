"""
Length-budget scan for S2G encoder inputs and SEL targets.

Constructs worst-case encoder inputs (full schema in SSI) and SEL targets
(all absent schema types in the null block) for every instance in train,
val, and test. Tokenises without truncation and reports a percentile table
(p50, p75, p90, p95, p99, max) of token lengths per task, per split, and
overall.

The overall p99 values are the recommended settings for
``tokenization.max_source_length`` and ``tokenization.max_target_length``.
Only a tokeniser is loaded; no model weights.

Usage::

    python -m s2g.scripts.measure_lengths \\
        --config configs/finetune.yaml \\
        data.data_dir=data/conll04 \\
        data.schema_file=data/conll04/relation.schema \\
        data.entity_schema_file=data/conll04/entity.schema
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from s2g.data import S2GDataset
from s2g.linearisation import (
    JOINT_TOKENS,
    PIPELINE_TOKENS,
    add_special_tokens_to_tokenizer,
    build_boundary_encoder_input,
    build_joint_encoder_input,
    build_joint_plus_encoder_input,
    build_ner_encoder_input,
    build_re_encoder_input,
    build_sel,
    organize_by_entity,
)
from s2g.scripts.config_utils import load_config, load_entity_schema, load_schema

logger = logging.getLogger(__name__)

_PCTS: Tuple[int, ...] = (50, 75, 90, 95, 99, 100)
_PCT_LABELS: Tuple[str, ...] = tuple("max" if p == 100 else f"p{p}" for p in _PCTS)


# ---- SCAN LOGIC ----


def _pct_dict(values: List[int]) -> Dict[int, int]:
    arr = np.array(values, dtype=np.int32)
    return {p: int(np.percentile(arr, p, method="lower")) for p in _PCTS}


def _encode_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def _scan_pipeline(
    dataset: S2GDataset,
    tokenizer,
    entity_schema: List[str],
    rel_schema: List[str],
) -> Dict[str, Dict[int, int]]:
    """Return length percentile dicts for all Pipeline task encoder/decoder strings."""
    tok = PIPELINE_TOKENS
    lengths: Dict[str, List[int]] = {k: [] for k in (
        "boundary_src", "ner_src", "re_src",
        "boundary_tgt", "ner_tgt",  "re_tgt",
    )}

    for i in tqdm(range(len(dataset)), desc="pipeline", leave=False):
        inst  = dataset[i]
        ents  = inst["entities"]
        rels  = inst["relations"]
        toks  = inst["tokens"]
        text  = inst["text"]
        pos_e = set(inst["entity_types"])
        pos_r = set(inst["rel_types"])
        neg_e = [t for t in entity_schema if t not in pos_e]
        neg_r = [t for t in rel_schema   if t not in pos_r]

        blocks      = organize_by_entity(ents, rels)
        ent_spans   = [(int(e["offset"][0]), int(e["offset"][1])) for e in ents]
        ent_data    = [(int(e["offset"][0]), int(e["offset"][1]), e["type"]) for e in ents]

        # Encoder inputs — full schema in SSI (worst case).
        lengths["boundary_src"].append(_encode_len(tokenizer,
            build_boundary_encoder_input(text, tok=tok)))
        lengths["ner_src"].append(_encode_len(tokenizer,
            build_ner_encoder_input(entity_schema, toks, ent_spans, tok=tok)))
        lengths["re_src"].append(_encode_len(tokenizer,
            build_re_encoder_input(rel_schema, toks, ent_data, tok=tok)))

        # Decoder targets — all absent schema types in null block (worst case).
        lengths["boundary_tgt"].append(_encode_len(tokenizer,
            build_sel(blocks, "boundary", tok)))
        lengths["ner_tgt"].append(_encode_len(tokenizer,
            build_sel(blocks, "ner", tok, rejected_ent_types=neg_e)))
        lengths["re_tgt"].append(_encode_len(tokenizer,
            build_sel(blocks, "re", tok, rejected_rel_types=neg_r)))

    return {k: _pct_dict(v) for k, v in lengths.items()}


def _scan_joint(
    dataset: S2GDataset,
    tokenizer,
    entity_schema: List[str],
    rel_schema: List[str],
) -> Dict[str, Dict[int, int]]:
    """Return length percentile dicts for all Joint task encoder/decoder strings."""
    tok = JOINT_TOKENS
    lengths: Dict[str, List[int]] = {k: [] for k in (
        "joint_src", "joint_plus_src",
        "joint_tgt", "joint_plus_tgt",
    )}

    for i in tqdm(range(len(dataset)), desc="joint", leave=False):
        inst  = dataset[i]
        text  = inst["text"]
        ents  = inst["entities"]
        rels  = inst["relations"]
        pos_e = set(inst["entity_types"])
        pos_r = set(inst["rel_types"])
        neg_e = [t for t in entity_schema if t not in pos_e]
        neg_r = [t for t in rel_schema   if t not in pos_r]
        blocks = organize_by_entity(ents, rels)

        lengths["joint_src"].append(_encode_len(tokenizer,
            build_joint_encoder_input(rel_schema, text, tok=tok)))
        lengths["joint_plus_src"].append(_encode_len(tokenizer,
            build_joint_plus_encoder_input(entity_schema, rel_schema, text, tok=tok)))
        lengths["joint_tgt"].append(_encode_len(tokenizer,
            build_sel(blocks, "joint", tok, rejected_rel_types=neg_r)))
        lengths["joint_plus_tgt"].append(_encode_len(tokenizer,
            build_sel(blocks, "joint+", tok,
                      rejected_ent_types=neg_e, rejected_rel_types=neg_r)))

    return {k: _pct_dict(v) for k, v in lengths.items()}


# ---- REPORTING ----


def _print_table(title: str, rows: Dict[str, Dict[int, int]]) -> None:
    col_w  = 8
    header = "".join(f"{lbl:>{col_w}}" for lbl in _PCT_LABELS)
    sep    = "=" * (20 + col_w * len(_PCTS))
    thin   = "-" * (20 + col_w * len(_PCTS))
    logger.info(sep)
    logger.info(title)
    logger.info(thin)
    logger.info(f"{'task/split':<20}{header}")
    for name, pcts in rows.items():
        row = "".join(f"{pcts[p]:>{col_w}d}" for p in _PCTS)
        logger.info(f"{name:<20}{row}")
    logger.info(sep)


def _overall_max(per_split: Dict[str, Dict[str, Dict[int, int]]]) -> Dict[str, Dict[int, int]]:
    """Element-wise max across splits — conservative upper bound."""
    result: Dict[str, Dict[int, int]] = {}
    for split_stats in per_split.values():
        for task, pcts in split_stats.items():
            if task not in result:
                result[task] = dict(pcts)
            else:
                for p, v in pcts.items():
                    result[task][p] = max(result[task][p], v)
    return result


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# ---- MAIN ----


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config()
    logger.info("Configuration loaded: %s", cfg.config_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    variant   = cfg.model.model_variant
    tokens    = PIPELINE_TOKENS if variant == "pipeline" else JOINT_TOKENS
    num_added = add_special_tokens_to_tokenizer(tokenizer, tokens)
    logger.info("Tokenizer: %s (+%d S2G tokens)", cfg.model.name, num_added)

    rel_schema    = load_schema(cfg.data.schema_file)
    entity_schema = load_entity_schema(cfg.data.entity_schema_file)
    logger.info(
        "Schema: %d relation types, %d entity types.",
        len(rel_schema), len(entity_schema),
    )

    scan_fn = _scan_pipeline if variant == "pipeline" else _scan_joint
    data_dir = Path(cfg.data.data_dir)

    per_split: Dict[str, Dict[str, Dict[int, int]]] = {}
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.jsonl"
        if not path.exists():
            logger.warning("Skipping %s: %s not found.", name, path)
            continue
        dataset = S2GDataset(path, seed=cfg.train.seed)
        logger.info("Scanning %s (%d instances)...", name, len(dataset))
        per_split[name] = scan_fn(
            dataset, tokenizer, entity_schema, rel_schema,
        )

    if not per_split:
        raise RuntimeError("No splits scanned; check data.data_dir.")

    # Report per-split then overall.
    for split_name, stats in per_split.items():
        _print_table(f"Split: {split_name}", stats)

    overall = _overall_max(per_split)
    _print_table("Overall (element-wise max across splits)", overall)

    # Identify worst-case tasks for YAML suggestions.
    src_key = "re_src"       if variant == "pipeline" else "joint_plus_src"
    tgt_key = "re_tgt"       if variant == "pipeline" else "joint_plus_tgt"
    p99_src = overall[src_key][99]
    p99_tgt = overall[tgt_key][99]
    logger.info(
        "Suggested YAML values (overall p99 rounded up to nearest 32):\n"
        "  tokenization.max_source_length: %d\n"
        "  tokenization.max_target_length: %d",
        _round_up(p99_src, 32),
        _round_up(p99_tgt, 32),
    )


if __name__ == "__main__":
    main()