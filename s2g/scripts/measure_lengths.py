"""
Length-budget scan for S2G encoder inputs and SEL targets.
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
    S2GTokens, add_special_tokens_to_tokenizer,
    build_boundary_encoder_input, build_boundary_joint_encoder_input, build_joint_encoder_input,
    build_ner_encoder_input, build_re_encoder_input, build_boundary_re_encoder_input, build_sel, organize_by_entity,
    VARIANT_TO_TASKS,
)
from s2g.scripts.config_utils import load_config, load_entity_schema, load_schema

logger = logging.getLogger(__name__)

_PCTS = (50, 75, 90, 95, 99, 100)
_PCT_LABELS = tuple("max" if p == 100 else f"p{p}" for p in _PCTS)


def _pct_dict(values: List[int]) -> Dict[int, int]:
    return {p: int(np.percentile(np.array(values, dtype=np.int32), p, method="lower")) for p in _PCTS}


def _scan_pipeline(dataset: S2GDataset, tokenizer, entity_schema: List[str], rel_schema: List[str], tasks: List[str], tokens: S2GTokens, ssi_prompt: str = "ssi") -> Dict[str, Dict[int, int]]:
    use_boundary = "boundary" in tasks
    use_ner = "ner" in tasks
    use_re = "re" in tasks
    use_boundary_re = "boundary_re" in tasks

    src_lengths = {t: [] for t in tasks}
    tgt_lengths = {t: [] for t in tasks}
    
    for i in tqdm(range(len(dataset)), desc="pipeline", leave=False):
        inst = dataset[i]
        ents, toks = inst["entities"], inst["tokens"]
        
        inst_ent_set = set(inst["entity_types"])
        inst_rel_set = set(inst["rel_types"])
        neg_e = [t for t in entity_schema if t not in inst_ent_set]
        neg_r = [t for t in rel_schema if t not in inst_rel_set]
        
        blocks = organize_by_entity(ents, inst["relations"])

        if use_boundary:
            src_lengths["boundary"].append(len(tokenizer.encode(build_boundary_encoder_input(inst["text"], tok=tokens, ssi_prompt=ssi_prompt), add_special_tokens=True)))
            tgt_lengths["boundary"].append(len(tokenizer.encode(build_sel(blocks, "boundary", tokens), add_special_tokens=True)))

        if use_ner:
            spans = [(int(e["offset"][0]), int(e["offset"][1])) for e in ents]
            src_lengths["ner"].append(len(tokenizer.encode(build_ner_encoder_input(entity_schema, toks, spans, tok=tokens, ssi_prompt=ssi_prompt), add_special_tokens=True)))
            tgt_lengths["ner"].append(len(tokenizer.encode(build_sel(blocks, "ner", tokens, rejected_ent_types=neg_e), add_special_tokens=True)))

        if use_re:
            data = [(int(e["offset"][0]), int(e["offset"][1]), e["type"]) for e in ents]
            src_lengths["re"].append(len(tokenizer.encode(build_re_encoder_input(entity_schema, rel_schema, inst["text"], tok=tokens, ssi_prompt=ssi_prompt), add_special_tokens=True)))
            tgt_lengths["re"].append(len(tokenizer.encode(build_sel(blocks, "re", tokens, rejected_rel_types=neg_r), add_special_tokens=True)))

        if use_boundary_re:
            data = [(int(e["offset"][0]), int(e["offset"][1]), "") for e in ents]
            src_lengths["boundary_re"].append(len(tokenizer.encode(build_boundary_re_encoder_input(rel_schema, inst["text"], tok=tokens, ssi_prompt=ssi_prompt), add_special_tokens=True)))
            tgt_lengths["boundary_re"].append(len(tokenizer.encode(build_sel(blocks, "boundary_re", tokens, rejected_rel_types=neg_r), add_special_tokens=True)))

    res = {}
    for t in tasks:
        res[f"{t}_src"] = _pct_dict(src_lengths[t])
        res[f"{t}_tgt"] = _pct_dict(tgt_lengths[t])
    return res


def _scan_boundary_joint(dataset: S2GDataset, tokenizer, entity_schema: List[str], rel_schema: List[str], tasks: List[str], tokens: S2GTokens, ssi_prompt: str = "ssi") -> Dict[str, Dict[int, int]]:
    use_boundary_joint = "boundary_joint" in tasks
    use_joint = "joint" in tasks

    src_lengths = {t: [] for t in tasks}
    tgt_lengths = {t: [] for t in tasks}

    for i in tqdm(range(len(dataset)), desc="boundary_joint", leave=False):
        inst = dataset[i]
        
        inst_ent_set = set(inst["entity_types"])
        inst_rel_set = set(inst["rel_types"])
        neg_e = [t for t in entity_schema if t not in inst_ent_set]
        neg_r = [t for t in rel_schema if t not in inst_rel_set]
        
        blocks = organize_by_entity(inst["entities"], inst["relations"])

        if use_boundary_joint:
            src_lengths["boundary_joint"].append(len(tokenizer.encode(build_boundary_joint_encoder_input(rel_schema, inst["text"], tok=tokens, ssi_prompt=ssi_prompt), add_special_tokens=True)))
            tgt_lengths["boundary_joint"].append(len(tokenizer.encode(build_sel(blocks, "boundary_joint", tokens, rejected_rel_types=neg_r), add_special_tokens=True)))

        if use_joint:
            src_lengths["joint"].append(len(tokenizer.encode(build_joint_encoder_input(entity_schema, rel_schema, inst["text"], tok=tokens, ssi_prompt=ssi_prompt), add_special_tokens=True)))
            tgt_lengths["joint"].append(len(tokenizer.encode(build_sel(blocks, "joint", tokens, rejected_ent_types=neg_e, rejected_rel_types=neg_r), add_special_tokens=True)))

    res = {}
    for t in tasks:
        res[f"{t}_src"] = _pct_dict(src_lengths[t])
        res[f"{t}_tgt"] = _pct_dict(tgt_lengths[t])
    return res


def _print_table(title: str, rows: Dict[str, Dict[int, int]]) -> None:
    header = "".join(f"{lbl:>8}" for lbl in _PCT_LABELS)
    sep = "=" * (20 + 8 * len(_PCTS))
    logger.info(sep)
    logger.info(title)
    logger.info("-" * (20 + 8 * len(_PCTS)))
    logger.info(f"{'task/split':<20}{header}")
    for name, pcts in rows.items():
        logger.info(f"{name:<20}" + "".join(f"{pcts[p]:>8d}" for p in _PCTS))
    logger.info(sep)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    variant = cfg.model.model_variant
    
    tokens = S2GTokens(variant, use_rejection=cfg.ssi.use_rejection)
    add_special_tokens_to_tokenizer(tokenizer, tokens)

    entity_schema, rel_schema = load_entity_schema(cfg.data.entity_schema_file), load_schema(cfg.data.schema_file)
    
    tasks = VARIANT_TO_TASKS[variant]

    is_pipeline_style = variant in {
        "pipeline", "boundary_pipeline", "boundary", "ner", "re", "boundary_re"
    }

    scan_fn = lambda d, tok, es, rs: (
        _scan_pipeline(d, tok, es, rs, tasks, tokens, ssi_prompt=cfg.ssi.ssi_prompt) if is_pipeline_style else _scan_boundary_joint(d, tok, es, rs, tasks, tokens, ssi_prompt=cfg.ssi.ssi_prompt)
    )
    per_split = {n: scan_fn(S2GDataset(Path(cfg.data.data_dir) / f"{n}.jsonl", seed=cfg.train.seed), tokenizer, entity_schema, rel_schema) 
                 for n in ("train", "val", "test") if (Path(cfg.data.data_dir) / f"{n}.jsonl").exists()}

    if not per_split:
        raise RuntimeError("No splits scanned; check data.data_dir.")

    for split_name, stats in per_split.items():
        _print_table(f"Split: {split_name}", stats)

    overall = {}
    for stats in per_split.values():
        for task, pcts in stats.items():
            overall[task] = {p: max(overall.get(task, {}).get(p, 0), v) for p, v in pcts.items()}

    _print_table("Overall (element-wise max across splits)", overall)

    for t in tasks:
        p99_src = overall[f"{t}_src"][99]
        p99_tgt = overall[f"{t}_tgt"][99]
        logger.info(f"--- Suggested Lengths for task: {t} ---")
        logger.info("Suggested Max Source Length (p99 rounded up to 32): %d", ((p99_src + 31) // 32) * 32)
        logger.info("Suggested Max Target Length (p99 rounded up to 32): %d", ((p99_tgt + 31) // 32) * 32)


if __name__ == "__main__":
    main()