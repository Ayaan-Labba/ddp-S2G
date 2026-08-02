"""
Length-budget scan for S2G encoder inputs and SEL targets.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from s2g.data import S2GCollator, S2GDataset
from s2g.linearisation import S2GTokens, add_special_tokens_to_tokenizer
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
    variant: str
) -> Dict[str, Dict[int, int]]:
    src_lengths: List[int] = []
    tgt_lengths: List[int] = []
    
    # Set step to max_steps to evaluate worst-case negative sample inclusion
    collator.current_step = collator._cfg.get("max_steps", 0)
    prep_fn = getattr(collator, f"_prepare_{variant}")

    for i in tqdm(range(len(dataset)), desc=f"scanning ({variant})", leave=False):
        inst = dataset[i]
        src_str, tgt_str = prep_fn(inst)
        
        src_lengths.append(len(tokenizer.encode(src_str, add_special_tokens=True)))
        tgt_lengths.append(len(tokenizer.encode(tgt_str, add_special_tokens=True)))

    return {
        f"{variant}_src": _pct_dict(src_lengths),
        f"{variant}_tgt": _pct_dict(tgt_lengths),
    }


def _print_table(title: str, rows: Dict[str, Dict[int, int]]) -> None:
    header = "".join(f"{lbl:>8}" for lbl in _PCT_LABELS)
    sep = "=" * (20 + 8 * len(_PCTS))
    logger.info(sep)
    logger.info(title)
    logger.info("-" * (20 + 8 * len(_PCTS)))
    logger.info(f"{'split/stream':<20}{header}")
    for name, pcts in rows.items():
        logger.info(f"{name:<20}" + "".join(f"{pcts[p]:>8d}" for p in _PCTS))
    logger.info(sep)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()
    set_seed(cfg.train.seed)

    variant = cfg.model.model_variant
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_checkpoint or cfg.model.name)
    
    tokens = S2GTokens(variant, use_rejection=cfg.graph.use_rejection, prompt=cfg.prompt.type)
    add_special_tokens_to_tokenizer(tokenizer, tokens)

    rel_schema, entity_schema = load_schema(cfg.data.rel_schema), load_entity_schema(cfg.data.ent_schema)

    collator = S2GCollator(
        tokenizer,
        entity_schema,
        rel_schema,
        {
            "model_variant": variant,
            "max_source_length": cfg.tokenizer.max_source_length,
            "max_target_length": cfg.tokenizer.max_target_length,
            "max_ent_types": cfg.prompt.max_ent_types or len(entity_schema),
            "max_rel_types": cfg.prompt.max_rel_types or len(rel_schema),
            "random_prompt": cfg.prompt.random_prompt,
            "random_graph": cfg.graph.random_graph,
            "positive_rate_start": getattr(cfg.prompt, "positive_rate_start", 0.5),
            "positive_rate_end": getattr(cfg.prompt, "positive_rate_end", 0.5),
            "negative_rate_start": getattr(cfg.prompt, "negative_rate_start", 0.5),
            "negative_rate_end": getattr(cfg.prompt, "negative_rate_end", 0.5),
            "pos_max_start": getattr(cfg.prompt, "pos_max_start", 1),
            "pos_max_end": getattr(cfg.prompt, "pos_max_end", 10),
            "negative_max_start": getattr(cfg.prompt, "negative_max_start", 1),
            "negative_max_end": getattr(cfg.prompt, "negative_max_end", 10),
            "mode": cfg.prompt.mode,
            "max_steps": cfg.train.max_steps,
            "use_rejection": cfg.graph.use_rejection,
            "use_nesting": cfg.graph.use_nesting,
            "prompt_type": cfg.prompt.type,
            "data_dir": cfg.data.data_dir,
        },
    )

    per_split = {}
    for n in ("train", "val", "test"):
        filepath = Path(cfg.data.data_dir) / f"{n}.jsonl"
        if filepath.exists():
            set_seed(cfg.train.seed)
            dataset = S2GDataset(filepath, seed=cfg.train.seed)
            per_split[n] = _scan_dataset(dataset, tokenizer, collator, variant)

    if not per_split:
        raise RuntimeError("No splits scanned; check data.data_dir.")

    for split_name, stats in per_split.items():
        _print_table(f"Split: {split_name}", stats)

    overall = {}
    for stats in per_split.values():
        for key, pcts in stats.items():
            overall[key] = {p: max(overall.get(key, {}).get(p, 0), v) for p, v in pcts.items()}

    _print_table("Overall (element-wise max across splits)", overall)

    p99_src = overall[f"{variant}_src"][99]
    p99_tgt = overall[f"{variant}_tgt"][99]
    logger.info(f"--- Suggested Lengths for variant: {variant} ---")
    logger.info("Suggested Max Source Length (p99 rounded up to 32): %d", ((p99_src + 31) // 32) * 32)
    logger.info("Suggested Max Target Length (p99 rounded up to 32): %d", ((p99_tgt + 31) // 32) * 32)


if __name__ == "__main__":
    main()