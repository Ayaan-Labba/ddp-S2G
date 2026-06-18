"""
Length-budget scan for S2G encoder inputs and SEL targets.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from s2g.data import S2GDataset, S2GCollator
from s2g.linearisation import (
    S2GTokens, add_special_tokens_to_tokenizer,
    organize_by_entity, VARIANT_TO_TASKS,
)
from s2g.scripts.config_utils import load_config, load_entity_schema, load_schema

logger = logging.getLogger(__name__)

_PCTS = (50, 75, 90, 95, 99, 100)
_PCT_LABELS = tuple("max" if p == 100 else f"p{p}" for p in _PCTS)


def _pct_dict(values: List[int]) -> Dict[int, int]:
    return {p: int(np.percentile(np.array(values, dtype=np.int32), p, method="lower")) for p in _PCTS}


def _scan_dataset(
    dataset: S2GDataset, 
    tokenizer: AutoTokenizer, 
    collator: S2GCollator, 
    tasks: List[str]
) -> Dict[str, Dict[int, int]]:
    src_lengths = {t: [] for t in tasks}
    tgt_lengths = {t: [] for t in tasks}
    
    # We want to measure under worst-case/representative conditions. 
    # If the collator is in bernoulli mode, the number of negative samples schedules up to `max_steps`.
    # To get a conservative/safe maximum length budget, we set the collator step to `max_steps`.
    collator.current_step = collator._cfg.get("max_steps", 0)

    for i in tqdm(range(len(dataset)), desc="scanning", leave=False):
        inst = dataset[i]
        blocks = organize_by_entity(inst["entities"], inst["relations"])
        
        for t in tasks:
            prep_fn = getattr(collator, f"_prepare_{t}")
            src_str, tgt_str = prep_fn(inst, blocks)
            
            src_lengths[t].append(len(tokenizer.encode(src_str, add_special_tokens=True)))
            tgt_lengths[t].append(len(tokenizer.encode(tgt_str, add_special_tokens=True)))

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
    set_seed(cfg.train.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    variant = cfg.model.model_variant
    
    tokens = S2GTokens(variant, use_rejection=cfg.sel.use_rejection)
    add_special_tokens_to_tokenizer(tokenizer, tokens, warm=cfg.sel.warm_start)

    entity_schema, rel_schema = load_entity_schema(cfg.data.entity_schema_file), load_schema(cfg.data.schema_file)
    
    tasks = VARIANT_TO_TASKS[variant]

    collator = S2GCollator(tokenizer, entity_schema, rel_schema, {
        "model_variant": variant, 
        "max_source_length": cfg.tokenization.max_source_length, 
        "max_target_length": cfg.tokenization.max_target_length,
        "max_ent_types": cfg.ssi.max_ent_types or len(entity_schema), 
        "max_rel_types": cfg.ssi.max_rel_types or len(rel_schema),
        "random_prompt": cfg.ssi.random_prompt, 
        "random_sel": cfg.sel.random_sel,
        "positive_rate_start": getattr(cfg.ssi, "positive_rate_start", 0.9),
        "positive_rate_end": getattr(cfg.ssi, "positive_rate_end", 0.9),
        "negative_rate_start": getattr(cfg.ssi, "negative_rate_start", 0.1),
        "negative_rate_end": getattr(cfg.ssi, "negative_rate_end", 0.1),
        "pos_max_start": getattr(cfg.ssi, "pos_max_start", 1),
        "pos_max_end": getattr(cfg.ssi, "pos_max_end", 20),
        "negative_max_start": getattr(cfg.ssi, "negative_max_start", 1),
        "negative_max_end": getattr(cfg.ssi, "negative_max_end", 20),
        "tasks": tasks, 
        "mode": cfg.ssi.mode, 
        "max_steps": cfg.train.max_steps,
        "use_rejection": cfg.sel.use_rejection,
        "use_nesting": cfg.sel.use_nesting,
        "ssi_prompt": cfg.ssi.ssi_prompt,
        "data_dir": cfg.data.data_dir,
    })

    per_split = {}
    for n in ("train", "val", "test"):
        filepath = Path(cfg.data.data_dir) / f"{n}.jsonl"
        if filepath.exists():
            set_seed(cfg.train.seed)
            dataset = S2GDataset(filepath, seed=cfg.train.seed)
            per_split[n] = _scan_dataset(dataset, tokenizer, collator, tasks)

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