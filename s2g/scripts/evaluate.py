"""
Standalone evaluation script for S2G using streaming DataLoader and S2GEvaluator.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from s2g.data import S2GCollator, S2GDataset, set_parent_death_signal
from s2g.evaluation import S2GEvaluator
from s2g.linearisation import S2GTokens, VALID_VARIANTS, verify_token_integrity
from s2g.scripts.config_utils import load_config, load_ent_schema, load_schema
from s2g.scripts.train import configure_dataloader_start_method, preload_forkserver_modules

logger = logging.getLogger(__name__)


def check_format_support(ckpt, variant: str, fmt: Dict[str, Any]) -> None:
    """
    Refuse a checkpoint whose sidecar records a format this revision cannot emit.

    Axis 2 retired the RE variants and made inline tail types and the ``<no_rel>``
    head marker unconditional, so the losing arms' checkpoints describe targets
    that ``build_graph`` no longer produces. Scoring one would rebuild its gold in
    the *winning* format and report plausible numbers against the wrong reference,
    which is precisely the failure the sidecar exists to prevent.
    """
    retired = {}
    if variant not in VALID_VARIANTS:
        retired['variant'] = variant
    # Neither key is written any more — that is exactly why they are checked here.
    # ``d023c99`` wrote both, so the Axis-2 arms' checkpoints carry them, and their
    # ``token_strs`` cannot betray the difference: it was saved from the *class*
    # map, which held every role whether or not the run emitted it. This is the
    # only thing standing between a losing arm and a plausible wrong number.
    # Absent means a sidecar predating the key; only an explicit False is a format
    # this code cannot reproduce.
    for key in ('joint_tail_type', 'inline_none'):
        if fmt.get(key) is False:
            retired[key] = False

    if retired:
        raise RuntimeError(
            f"{ckpt} was trained under a retired format: {retired}. Axis 2 removed "
            "the RE variants and made inline tail types and the <no_rel> head marker "
            "unconditional, so this revision cannot rebuild that gold. Score it with "
            "the revision it was trained on."
        )


def check_token_map(ckpt, saved: Optional[Dict[str, str]], tokens: S2GTokens) -> None:
    """
    Refuse a checkpoint whose token map disagrees with the one it would be scored
    under. A mismatch mis-parses every target rather than failing, so there is no
    setting here that could recover it.

    Compared on **shared keys**, not whole maps: a role added to
    ``ALL_TOKEN_NAMES`` since the checkpoint was written leaves the older roles on
    their own sentinels, so the emitted format is unchanged and the checkpoint is
    still scoreable. A role that is *active now* but absent from the saved map is a
    real difference, and is rejected separately.
    """
    if not saved:
        return

    saved = dict(saved)
    conflicts = {k: (saved[k], v) for k, v in tokens.token_strs.items() if k in saved and saved[k] != v}
    absent = sorted(name for name in tokens.active_tokens if name not in saved)

    if conflicts or absent:
        raise RuntimeError(
            f"Token map mismatch: {ckpt} was trained with {saved} but this code "
            f"emits {dict(tokens.token_strs)}. "
            f"Conflicting roles: {conflicts or 'none'}; "
            f"active roles the checkpoint never carried: {absent or 'none'}. "
            "Any metrics would be meaningless; score it with the revision it was "
            "trained on."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()

    # Must happen before any DataLoader spins up workers (see train.py).
    configure_dataloader_start_method(cfg.hardware.dataloader_start_method)
    preload_forkserver_modules()

    if cfg.hardware.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, cfg.hardware.gpu_ids))

    set_seed(cfg.train.seed)

    ckpt = cfg.model.pretrained_checkpoint
    if not ckpt:
        raise ValueError("model.pretrained_checkpoint is required for evaluation.")

    ckpt_path = Path(ckpt)

    # Settings that determine how targets are linearised must match the training
    # run exactly, otherwise the gold graphs are rebuilt in a different format and
    # the reported scores are meaningless. Prefer the sidecar written by train.py.
    fmt_file = ckpt_path / "s2g_format.json"
    fmt = {}
    if fmt_file.exists():
        with open(fmt_file, 'r', encoding='utf-8') as f:
            fmt = json.load(f)
        logger.info("Loaded linearisation format from %s: %s", fmt_file, fmt)
    else:
        logger.warning(
            "%s not found; falling back to the evaluation config. Verify that "
            "graph.use_rejection / graph.nesting / graph.dedup / "
            "prompt.type / prompt.style "
            "match training.",
            fmt_file,
        )

    variant_file = ckpt_path / "variant.txt"
    variant = fmt.get('variant') or (
        variant_file.read_text(encoding='utf-8').strip() if variant_file.exists() else cfg.model.variant
    )

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

    use_rejection = fmt.get('use_rejection', cfg.graph.use_rejection)
    nesting = fmt.get('nesting', cfg.graph.nesting)
    dedup = fmt.get('dedup', cfg.graph.dedup)
    prompt_type = fmt.get('prompt_type', cfg.prompt.type)
    prompt_style = fmt.get('style', cfg.prompt.style)
    tokens = S2GTokens(variant=variant, use_rejection=use_rejection)

    check_format_support(ckpt, variant, fmt)
    check_token_map(ckpt, fmt.get('token_strs'), tokens)
    verify_token_integrity(tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    rel_schema_path = cfg.data.rel_schema
    ent_schema_path = cfg.data.ent_schema

    rel_schema = load_schema(rel_schema_path) if rel_schema_path else []
    ent_schema = load_ent_schema(ent_schema_path) if ent_schema_path else []

    split = cfg.evaluation.split
    dataset_path = Path(cfg.data.data_dir) / f"{split}.jsonl"
    eval_dataset = S2GDataset(dataset_path, seed=cfg.train.seed)

    base_collator = S2GCollator(
        tokenizer=tokenizer,
        ent_schema=ent_schema,
        rel_schema=rel_schema,
        config={
            'variant': variant,
            'max_source_length': cfg.tokenizer.max_source_length,
            'max_target_length': cfg.tokenizer.max_target_length,
            'max_ent_types': fmt.get('max_ent_types', cfg.prompt.max_ent_types) or len(ent_schema),
            'max_rel_types': fmt.get('max_rel_types', cfg.prompt.max_rel_types) or len(rel_schema),
            'mode': cfg.prompt.mode,
            'prompt_type': prompt_type,
            'prompt_style': prompt_style,
            'random_prompt': cfg.prompt.random_prompt,
            'random_graph': cfg.graph.random_graph,
            'use_rejection': use_rejection,
            'nesting': nesting,
            'dedup': dedup,
            'seed': cfg.train.seed,
        }
    )

    eval_collator = base_collator.to_eval_mode()

    dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.hardware.num_workers,
        collate_fn=eval_collator,
        worker_init_fn=set_parent_death_signal,
    )

    evaluator = S2GEvaluator(
        tokenizer=tokenizer,
        tokens=tokens,
        variant=variant,
        rel_schema=rel_schema,
        ent_schema=ent_schema,
        dedup=dedup,
    )

    out_dir = Path(cfg.data.output_dir)
    constraint_decoding = getattr(cfg.generation, 'constraint_decoding', False)
    evaluator.run_evaluation(
        dataset=eval_dataset,
        split=split,
        dataloader=dataloader,
        out_dir=out_dir,
        model=model,
        max_target_length=cfg.tokenizer.max_target_length,
        num_beams=cfg.generation.num_beams,
        constraint_decoding=constraint_decoding,
        device=device,
        length_penalty=getattr(cfg.generation, 'length_penalty', None),
        no_repeat_ngram_size=getattr(cfg.generation, 'no_repeat_ngram_size', None),
        early_stopping=getattr(cfg.generation, 'early_stopping', None)
    )


if __name__ == "__main__":
    main()