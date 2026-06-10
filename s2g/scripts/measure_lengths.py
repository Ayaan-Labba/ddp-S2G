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
    JOINT_TOKENS, PIPELINE_TOKENS, add_special_tokens_to_tokenizer,
    build_boundary_encoder_input, build_joint_encoder_input, build_joint_plus_encoder_input,
    build_ner_encoder_input, build_re_encoder_input, build_sel, organize_by_entity,
)
from s2g.scripts.config_utils import load_config, load_entity_schema, load_schema

logger = logging.getLogger(__name__)


def _pct_dict(values: List[int]) -> Dict[int, int]:
    return {p: int(np.percentile(np.array(values, dtype=np.int32), p, method="lower")) for p in (50, 75, 90, 95, 99, 100)}


def _scan_pipeline(dataset: S2GDataset, tokenizer, entity_schema: List[str], rel_schema: List[str]) -> Dict[str, Dict[int, int]]:
    l: Dict[str, List[int]] = {k: [] for k in ("boundary_src", "ner_src", "re_src", "boundary_tgt", "ner_tgt", "re_tgt")}
    
    for i in tqdm(range(len(dataset)), desc="pipeline", leave=False):
        inst, ents, toks = dataset[i], dataset[i]["entities"], dataset[i]["tokens"]
        
        # EFFICIENCY FIX: Resolve sets exactly once before schema loops to prevent O(N*M) creation allocations
        inst_ent_set = set(inst["entity_types"])
        inst_rel_set = set(inst["rel_types"])
        neg_e = [t for t in entity_schema if t not in inst_ent_set]
        neg_r = [t for t in rel_schema if t not in inst_rel_set]
        
        blocks = organize_by_entity(ents, inst["relations"])

        l["boundary_src"].append(len(tokenizer.encode(build_boundary_encoder_input(inst["text"], tok=PIPELINE_TOKENS), add_special_tokens=True)))
        l["ner_src"].append(len(tokenizer.encode(build_ner_encoder_input(entity_schema, toks, [(int(e["offset"][0]), int(e["offset"][1])) for e in ents], tok=PIPELINE_TOKENS), add_special_tokens=True)))
        l["re_src"].append(len(tokenizer.encode(build_re_encoder_input(rel_schema, toks, [(int(e["offset"][0]), int(e["offset"][1]), e["type"]) for e in ents], tok=PIPELINE_TOKENS), add_special_tokens=True)))
        l["boundary_tgt"].append(len(tokenizer.encode(build_sel(blocks, "boundary", PIPELINE_TOKENS), add_special_tokens=True)))
        l["ner_tgt"].append(len(tokenizer.encode(build_sel(blocks, "ner", PIPELINE_TOKENS, rejected_ent_types=neg_e), add_special_tokens=True)))
        l["re_tgt"].append(len(tokenizer.encode(build_sel(blocks, "re", PIPELINE_TOKENS, rejected_rel_types=neg_r), add_special_tokens=True)))

    return {k: _pct_dict(v) for k, v in l.items()}


def _scan_joint(dataset: S2GDataset, tokenizer, entity_schema: List[str], rel_schema: List[str]) -> Dict[str, Dict[int, int]]:
    l: Dict[str, List[int]] = {k: [] for k in ("joint_src", "joint_plus_src", "joint_tgt", "joint_plus_tgt")}

    for i in tqdm(range(len(dataset)), desc="joint", leave=False):
        inst = dataset[i]
        
        # EFFICIENCY FIX: Resolve sets exactly once before loops
        inst_ent_set = set(inst["entity_types"])
        inst_rel_set = set(inst["rel_types"])
        neg_e = [t for t in entity_schema if t not in inst_ent_set]
        neg_r = [t for t in rel_schema if t not in inst_rel_set]
        
        blocks = organize_by_entity(inst["entities"], inst["relations"])

        l["joint_src"].append(len(tokenizer.encode(build_joint_encoder_input(rel_schema, inst["text"], tok=JOINT_TOKENS), add_special_tokens=True)))
        l["joint_plus_src"].append(len(tokenizer.encode(build_joint_plus_encoder_input(entity_schema, rel_schema, inst["text"], tok=JOINT_TOKENS), add_special_tokens=True)))
        l["joint_tgt"].append(len(tokenizer.encode(build_sel(blocks, "joint", JOINT_TOKENS, rejected_rel_types=neg_r), add_special_tokens=True)))
        l["joint_plus_tgt"].append(len(tokenizer.encode(build_sel(blocks, "joint+", JOINT_TOKENS, rejected_ent_types=neg_e, rejected_rel_types=neg_r), add_special_tokens=True)))

    return {k: _pct_dict(v) for k, v in l.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    add_special_tokens_to_tokenizer(tokenizer, PIPELINE_TOKENS if cfg.model.model_variant == "pipeline" else JOINT_TOKENS)

    entity_schema, rel_schema = load_entity_schema(cfg.data.entity_schema_file), load_schema(cfg.data.schema_file)
    scan_fn = _scan_pipeline if cfg.model.model_variant == "pipeline" else _scan_joint

    per_split = {n: scan_fn(S2GDataset(Path(cfg.data.data_dir) / f"{n}.jsonl", seed=cfg.train.seed), tokenizer, entity_schema, rel_schema) 
                 for n in ("train", "val", "test") if (Path(cfg.data.data_dir) / f"{n}.jsonl").exists()}

    overall = {}
    for stats in per_split.values():
        for task, pcts in stats.items():
            overall[task] = {p: max(overall.get(task, {}).get(p, 0), v) for p, v in pcts.items()}

    logger.info("Suggested Max Source Length: %d", ((overall["re_src" if cfg.model.model_variant == "pipeline" else "joint_plus_src"][99] + 31) // 32) * 32)
    logger.info("Suggested Max Target Length: %d", ((overall["re_tgt" if cfg.model.model_variant == "pipeline" else "joint_plus_tgt"][99] + 31) // 32) * 32)

if __name__ == "__main__":
    main()