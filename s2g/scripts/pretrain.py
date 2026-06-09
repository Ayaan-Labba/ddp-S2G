"""
Pre-training script for the S2G model on REBEL.

Trains a Flan-T5 model on the REBEL relation extraction dataset using
budget-mode SSI construction and the multi-task SEL linearisation.  This
script is the Experiment 3 entry point.  It mirrors ``finetune.py``
exactly; the only differences are the data source (REBEL vs benchmark),
the scheduler type (inverse_sqrt vs cosine), and scale of hyperparameters.

Usage::

    # Fresh start
    torchrun --nproc_per_node=<N> -m s2g.scripts.pretrain \\
        --config configs/pretrain.yaml

    # Resume
    torchrun --nproc_per_node=<N> -m s2g.scripts.pretrain \\
        --config configs/pretrain.yaml \\
        checkpoint.resume_from=outputs/pretrain/checkpoint-last
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    set_seed,
)

from s2g.data import S2GCollator, S2GDataset
from s2g.evaluation import (
    GenerateTextSamplesCallback,
    PeriodicCheckpointCallback,
    StepTrackingCallback,
    load_run_metadata,
)
from s2g.linearisation import (
    JOINT_TOKENS,
    PIPELINE_TOKENS,
    add_special_tokens_to_tokenizer,
)
from s2g.scripts.config_utils import (
    load_config,
    load_entity_schema,
    load_schema,
)
from s2g.training import S2GTrainer

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config()
    logger.info("Configuration loaded: %s", cfg.config_path)

    if cfg.hardware.gpu_ids is not None:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            logger.warning(
                "hardware.gpu_ids set but WORLD_SIZE > 1: ignored in "
                "distributed mode (managed by torchrun)."
            )
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(g) for g in cfg.hardware.gpu_ids
            )

    set_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    wandb_run_id: Optional[str] = None
    wandb_resume: Optional[str] = None
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.checkpoint.resume_from is not None:
        meta = load_run_metadata(cfg.data.output_dir)
        if meta and meta.get("wandb_run_id"):
            wandb_run_id = meta["wandb_run_id"]
            wandb_resume = "must"

    if local_rank == 0:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            id=wandb_run_id,
            resume=wandb_resume,
        )

    rel_schema    = load_schema(cfg.data.schema_file)
    entity_schema = load_entity_schema(cfg.data.entity_schema_file)
    logger.info(
        "Schemas: %d relation types, %d entity types.",
        len(rel_schema), len(entity_schema),
    )

    train_dataset = S2GDataset(
        Path(cfg.data.data_dir) / "train.jsonl",
        seed=cfg.train.seed,
    )
    val_dataset = S2GDataset(
        Path(cfg.data.data_dir) / "val.jsonl",
        subset_fraction=cfg.validation.percent_check,
        seed=cfg.train.seed,
    )
    logger.info(
        "Train: %d  Val: %d", len(train_dataset), len(val_dataset)
    )

    train_eval_dataset = None
    if cfg.validation.train_eval_percent_check:
        n = max(1, int(len(train_dataset) * cfg.validation.train_eval_percent_check))
        idxs = rng.choice(len(train_dataset), size=n, replace=False).tolist()
        from torch.utils.data import Subset
        train_eval_dataset = Subset(train_dataset, idxs)
        logger.info("Train-eval subsample: %d instances", len(train_eval_dataset))

    start_ckpt = cfg.model.pretrained_checkpoint or cfg.model.name
    tokenizer  = AutoTokenizer.from_pretrained(start_ckpt)
    model      = AutoModelForSeq2SeqLM.from_pretrained(start_ckpt)

    tokens = (
        PIPELINE_TOKENS if cfg.model.model_variant == "pipeline"
        else JOINT_TOKENS
    )
    num_added = add_special_tokens_to_tokenizer(tokenizer, tokens, model)
    logger.info(
        "Variant=%s — %d special tokens added.", cfg.model.model_variant, num_added
    )

    collator_cfg: Dict[str, Any] = {
        "model_variant":          cfg.model.model_variant,
        "max_source_length":      cfg.tokenization.max_source_length,
        "max_target_length":      cfg.tokenization.max_target_length,
        "max_ent_types_in_prompt": cfg.ssi.max_ent_types_in_prompt or len(entity_schema),
        "max_rel_types_in_prompt": cfg.ssi.max_rel_types_in_prompt or len(rel_schema),
        "random_prompt":           cfg.ssi.random_prompt,
        "random_sel":              cfg.ssi.random_sel,
    }
    collator = S2GCollator(tokenizer, entity_schema, rel_schema, collator_cfg)

    callbacks = [
        StepTrackingCallback(collator),
        EarlyStoppingCallback(
            early_stopping_patience=cfg.validation.early_stopping_patience,
        ),
        PeriodicCheckpointCallback(
            output_dir=cfg.data.output_dir,
            every_n_steps=cfg.checkpoint.every_n_steps,
            wandb_run_id=wandb.run.id if wandb.run is not None else None,
        ),
    ]

    # Sample generation callback — monitor the primary task for the variant.
    sample_task = "re" if cfg.model.model_variant == "pipeline" else "joint"
    sample_batch = [val_dataset[i] for i in range(min(8, len(val_dataset)))]
    callbacks.append(
        GenerateTextSamplesCallback(
            tokenizer=tokenizer,
            sample_batch=sample_batch,
            collator=collator,
            task=sample_task,
            interval=cfg.callbacks.sample_generation_interval,
            eval_beams=cfg.generation.num_beams,
            max_target_length=cfg.tokenization.max_target_length,
        )
    )

    # prevent the Trainer from building its own scheduler; S2GTrainer
    # creates the inverse_sqrt schedule in create_scheduler().
    hf_scheduler_type = (
        "constant" if cfg.scheduler.type == "inverse_sqrt"
        else cfg.scheduler.type
    )

    # Primary metric key produced by S2GTrainer.evaluate():
    primary_metric = (
        "re_rel_boundary_f1"
        if cfg.model.model_variant == "pipeline"
        else "joint_rel_boundary_f1"
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.data.output_dir,

        # Training loop
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_acc_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=cfg.train.gradient_clip_value,
        fp16=(cfg.train.precision == "16"),
        bf16=(cfg.train.precision == "bf16"),
        dataloader_num_workers=cfg.hardware.num_workers,
        dataloader_persistent_workers=cfg.hardware.persistent_workers,
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,

        # Optimiser
        optim=cfg.optimizer.optim,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        adam_beta1=cfg.optimizer.adam_beta1,
        adam_beta2=cfg.optimizer.adam_beta2,
        adam_epsilon=cfg.optimizer.adam_epsilon,

        # Scheduler
        warmup_steps=cfg.scheduler.warmup_steps,
        lr_scheduler_type=hf_scheduler_type,

        # Evaluation — driven by overridden evaluate(); no predict_with_generate
        eval_strategy="steps",
        eval_steps=cfg.validation.check_interval,
        per_device_eval_batch_size=cfg.validation.batch_size,
        predict_with_generate=False,

        # Checkpointing
        save_strategy="steps",
        save_steps=cfg.validation.check_interval,
        save_total_limit=cfg.checkpoint.save_top_k + 1,
        load_best_model_at_end=True,
        metric_for_best_model=primary_metric,
        greater_is_better=True,

        # Logging
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
        run_name=cfg.wandb.run_name,

        # Misc
        remove_unused_columns=False,
        label_names=[],
    )

    eval_cfg = {
        "max_source_length": cfg.tokenization.max_source_length,
        "max_target_length": cfg.tokenization.max_target_length,
        "eval_batch_size":   cfg.validation.batch_size,
        "eval_beams":        cfg.generation.num_beams,
    }
    trainer = S2GTrainer(
        scheduler_type=cfg.scheduler.type,
        model_variant=cfg.model.model_variant,
        tokens=tokens,
        entity_schema=entity_schema,
        rel_schema=rel_schema,
        eval_cfg=eval_cfg,
        train_eval_dataset=train_eval_dataset,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info("Starting pre-training...")
    trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from)
    logger.info("Pre-training complete.")

    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    # Write variant metadata so from_pretrained() can auto-detect it.
    (best_dir / "model_variant.txt").write_text(
        cfg.model.model_variant, encoding="utf-8"
    )
    logger.info("Best model saved to %s", best_dir)

    full_val = S2GDataset(
        Path(cfg.data.data_dir) / "val.jsonl",
        seed=cfg.train.seed,
    )
    val_metrics = trainer.evaluate(eval_dataset=full_val)
    metrics_path = output_dir / "val_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    logger.info("Final val metrics: %s", val_metrics)


if __name__ == "__main__":
    main()