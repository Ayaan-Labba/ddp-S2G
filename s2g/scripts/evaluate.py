"""
Standalone evaluation script for S2G using streaming DataLoader and S2GEvaluator.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from s2g.data import S2GCollator, S2GDataset
from s2g.evaluation import S2GEvaluator
from s2g.linearisation import S2GTokens, add_special_tokens_to_tokenizer
from s2g.scripts.config_utils import load_config, load_ent_schema, load_schema
from s2g.scripts.train import configure_dataloader_start_method

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()

    # Must happen before any DataLoader spins up workers (see train.py).
    configure_dataloader_start_method(cfg.hardware.dataloader_start_method)

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
            "graph.use_rejection / graph.markers / graph.nesting / graph.joint_tail_type / "
            "graph.dedup / prompt.type / prompt.style "
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
    markers = fmt.get('markers', cfg.graph.markers)
    nesting = fmt.get('nesting', cfg.graph.nesting)
    joint_tail_type = fmt.get('joint_tail_type', cfg.graph.joint_tail_type)
    dedup = fmt.get('dedup', cfg.graph.dedup)
    prompt_type = fmt.get('prompt_type', cfg.prompt.type)
    prompt_style = fmt.get('style', cfg.prompt.style)
    tokens = S2GTokens(variant=variant, use_rejection=use_rejection, markers=markers)
    add_special_tokens_to_tokenizer(tokenizer, tokens, model, warm_start=False)

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
            'markers': markers,
            'nesting': nesting,
            'joint_tail_type': joint_tail_type,
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