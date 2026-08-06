"""
Standalone evaluation script for S2G using streaming DataLoader and S2GEvaluator.
"""
from __future__ import annotations

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

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()

    if cfg.hardware.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, cfg.hardware.gpu_ids))

    set_seed(cfg.train.seed)

    ckpt = cfg.model.pretrained_checkpoint
    if not ckpt:
        raise ValueError("model.pretrained_checkpoint is required for evaluation.")

    ckpt_path = Path(ckpt)
    variant_file = ckpt_path / "variant.txt"
    variant = (
        variant_file.read_text(encoding='utf-8').strip() if variant_file.exists() else cfg.model.variant
    )

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

    use_rejection = getattr(cfg.graph, 'use_rejection', True)
    prompt_type = getattr(cfg.prompt, 'type', 'natural')
    tokens = S2GTokens(variant=variant, use_rejection=use_rejection, prompt=prompt_type)
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
            'max_ent_types': len(ent_schema),
            'max_rel_types': len(rel_schema),
            'mode': getattr(cfg.prompt, 'mode', 'budget'),
            'prompt_type': prompt_type,
            'random_prompt': getattr(cfg.prompt, 'random_prompt', False),
            'random_graph': getattr(cfg.graph, 'random_graph', False),
            'use_rejection': use_rejection,
            'use_nesting': getattr(cfg.graph, 'use_nesting', True),
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