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
from s2g.scripts.config_utils import load_config, load_entity_schema, load_schema

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()

    if cfg.hardware.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.hardware.gpu_ids))

    set_seed(cfg.train.seed)

    ckpt = cfg.model.pretrained_checkpoint
    if not ckpt:
        raise ValueError("model.pretrained_checkpoint is required for evaluation.")

    ckpt_path = Path(ckpt)
    variant_file = ckpt_path / "model_variant.txt"
    model_variant = (
        variant_file.read_text(encoding="utf-8").strip()
        if variant_file.exists()
        else cfg.model.model_variant
    )

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

    use_rejection = getattr(cfg.sel, "use_rejection", False)
    ssi_prompt = getattr(cfg.ssi, "ssi_prompt", "ssi")
    tokens = S2GTokens(variant=model_variant, use_rejection=use_rejection, prompt=ssi_prompt)
    add_special_tokens_to_tokenizer(tokenizer, tokens, model, warm=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rel_schema_path = getattr(cfg.data, "schema_file", None) or getattr(cfg.data, "rel_schema", None)
    ent_schema_path = getattr(cfg.data, "entity_schema_file", None) or getattr(cfg.data, "ent_schema", None)

    rel_schema = load_schema(rel_schema_path) if rel_schema_path else []
    entity_schema = load_entity_schema(ent_schema_path) if ent_schema_path else []

    split = cfg.evaluation.split
    dataset_path = Path(cfg.data.data_dir) / f"{split}.jsonl"
    eval_dataset = S2GDataset(dataset_path, seed=cfg.train.seed)

    base_collator = S2GCollator(
        tokenizer=tokenizer,
        entity_schema=entity_schema,
        rel_schema=rel_schema,
        config={
            "model_variant": model_variant,
            "max_source_length": cfg.tokenization.max_source_length,
            "max_target_length": cfg.tokenization.max_target_length,
            "max_ent_types": len(entity_schema),
            "max_rel_types": len(rel_schema),
            "mode": getattr(cfg.ssi, "mode", "budget"),
            "random_prompt": getattr(cfg.ssi, "random_prompt", False),
            "random_sel": getattr(cfg.sel, "random_sel", False),
            "use_rejection": use_rejection,
            "use_nesting": getattr(cfg.sel, "use_nesting", True),
            "ssi_prompt": ssi_prompt,
        },
    )
    eval_collator = base_collator.to_eval_mode()

    dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.validation.batch_size,
        shuffle=False,
        num_workers=cfg.hardware.num_workers,
        collate_fn=eval_collator,
    )

    evaluator = S2GEvaluator(
        tokenizer=tokenizer,
        tokens=tokens,
        model_variant=model_variant,
        rel_schema=rel_schema,
        entity_schema=entity_schema,
    )

    out_dir = Path(cfg.data.output_dir)
    constraint_decoding = getattr(cfg.generation, "constraint_decoding", False)
    evaluator.run_evaluation(
        model=model,
        dataloader=dataloader,
        dataset=eval_dataset,
        collator=eval_collator,
        out_dir=out_dir,
        split=split,
        device=device,
        max_target_length=cfg.tokenization.max_target_length,
        num_beams=cfg.generation.num_beams,
        constraint_decoding=constraint_decoding,
    )


if __name__ == "__main__":
    main()