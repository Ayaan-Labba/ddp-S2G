"""
Training script for the S2G model with integrated streaming evaluation.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import wandb
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, set_seed

from s2g.data import S2GCollator, S2GDataset
from s2g.evaluation import (
    GenerateTextSamplesCallback,
    PeriodicCheckpointCallback,
    S2GEarlyStoppingCallback,
    S2GEvaluator,
    StepTrackingCallback,
    load_run_metadata,
)
from s2g.linearisation import S2GTokens, add_special_tokens_to_tokenizer
from s2g.scripts.config_utils import load_config, load_entity_schema, load_schema
from s2g.training import S2GTrainer


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_config()

    if cfg.hardware.gpu_ids is not None:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            logger.warning("Distributed mode: hardware.gpu_ids ignored.")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.hardware.gpu_ids))

    set_seed(cfg.train.seed)

    out_dir = Path(cfg.data.output_dir)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run_id = (
        (load_run_metadata(cfg.data.output_dir) or {}).get("wandb_run_id")
        if cfg.checkpoint.resume_from
        else None
    )
    if local_rank == 0:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            id=wandb_run_id,
            resume="must" if wandb_run_id else None,
        )

    rel_schema, entity_schema = load_schema(cfg.data.rel_schema), load_entity_schema(
        cfg.data.ent_schema
    )
    train_dataset = S2GDataset(Path(cfg.data.data_dir) / "train.jsonl", seed=cfg.train.seed)
    val_dataset = S2GDataset(
        Path(cfg.data.data_dir) / "val.jsonl",
        subset_fraction=cfg.validation.percent_check if cfg.validation.percent_check < 1.0 else None,
        seed=cfg.train.seed,
    )

    train_eval_dataset = (
        Subset(
            train_dataset,
            np.random.default_rng(cfg.train.seed)
            .choice(
                len(train_dataset),
                size=max(1, int(len(train_dataset) * cfg.validation.train_eval_percent_check)),
                replace=False,
            )
            .tolist(),
        )
        if cfg.validation.train_eval_percent_check
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_checkpoint or cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.pretrained_checkpoint or cfg.model.name)

    tokens = S2GTokens(cfg.model.model_variant, use_rejection=cfg.graph.use_rejection)
    warm_start = cfg.tokenizer.warm_start and (cfg.model.pretrained_checkpoint is None)
    add_special_tokens_to_tokenizer(tokenizer, tokens, model, warm=warm_start)

    collator = S2GCollator(
        tokenizer,
        entity_schema,
        rel_schema,
        {
            "model_variant": cfg.model.model_variant,
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

    callbacks = [
        StepTrackingCallback(collator),
        S2GEarlyStoppingCallback(early_stopping_patience=cfg.validation.early_stopping_patience),
        PeriodicCheckpointCallback(
            output_dir=cfg.data.output_dir,
            every_n_steps=cfg.checkpoint.every_n_steps,
            wandb_run_id=wandb.run.id if wandb.run else None,
        ),
        GenerateTextSamplesCallback(
            tokenizer,
            [val_dataset[i] for i in range(min(8, len(val_dataset)))],
            collator,
            cfg.model.model_variant,
            cfg.callbacks.sample_generation_interval,
            cfg.generation.num_beams,
            cfg.tokenizer.max_target_length,
        ),
    ]

    best_metric = cfg.validation.early_stopping_metric

    trainer = S2GTrainer(
        scheduler_type=cfg.scheduler.type,
        model_variant=cfg.model.model_variant,
        tokens=tokens,
        entity_schema=entity_schema,
        rel_schema=rel_schema,
        train_eval_dataset=train_eval_dataset,
        eval_cfg={
            "max_source_length": cfg.tokenizer.max_source_length,
            "max_target_length": cfg.tokenizer.max_target_length,
            "eval_batch_size": cfg.validation.batch_size,
            "eval_beams": cfg.generation.num_beams,
            "prompt_type": cfg.prompt.type,
        },
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
        args=Seq2SeqTrainingArguments(
            output_dir=cfg.data.output_dir,
            max_steps=cfg.train.max_steps,
            per_device_train_batch_size=cfg.train.batch_size,
            gradient_accumulation_steps=cfg.train.gradient_acc_steps,
            gradient_checkpointing=cfg.train.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=cfg.train.gradient_clip_value,
            fp16=(cfg.train.precision == "fp16"),
            bf16=(cfg.train.precision == "bf16"),
            dataloader_num_workers=cfg.hardware.num_workers,
            dataloader_persistent_workers=cfg.hardware.persistent_workers,
            seed=cfg.train.seed,
            data_seed=cfg.train.seed,
            optim=cfg.optimizer.optim,
            learning_rate=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            adam_beta1=cfg.optimizer.adam_beta1,
            adam_beta2=cfg.optimizer.adam_beta2,
            adam_epsilon=cfg.optimizer.adam_epsilon,
            warmup_steps=cfg.scheduler.warmup_steps,
            lr_scheduler_type="constant" if cfg.scheduler.type == "inverse_sqrt" else cfg.scheduler.type,
            eval_strategy="steps",
            eval_steps=cfg.validation.check_interval,
            per_device_eval_batch_size=cfg.validation.batch_size,
            predict_with_generate=False,
            save_strategy="steps",
            save_steps=cfg.validation.check_interval,
            save_total_limit=cfg.checkpoint.save_top_k + 1,
            load_best_model_at_end=True,
            metric_for_best_model=best_metric,
            greater_is_better=True,
            logging_strategy="steps",
            logging_steps=cfg.train.steps_per_log,
            report_to="wandb",
            run_name=cfg.wandb.run_name,
            remove_unused_columns=False,
            label_names=[],
        ),
    )

    trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from)

    # Post-training evaluation strictly on Rank 0
    if trainer.is_world_process_zero():
        best_dir = out_dir / "best_model"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        (best_dir / "model_variant.txt").write_text(cfg.model.model_variant, encoding="utf-8")

        model.eval()
        device = next(model.parameters()).device

        eval_collator = collator.to_eval_mode()
        evaluator = S2GEvaluator(
            tokenizer=tokenizer,
            tokens=tokens,
            model_variant=cfg.model.model_variant,
            rel_schema=rel_schema,
            entity_schema=entity_schema,
        )

        # 1. Full Validation Set Evaluation
        val_dataset_full = S2GDataset(Path(cfg.data.data_dir) / "val.jsonl", seed=cfg.train.seed)
        val_dataloader = DataLoader(
            val_dataset_full,
            batch_size=cfg.validation.batch_size,
            shuffle=False,
            num_workers=cfg.hardware.num_workers,
            collate_fn=eval_collator,
        )

        constraint_decoding = getattr(cfg.generation, "constraint_decoding", False)
        logger.info("Running post-training validation evaluation...")
        val_metrics = evaluator.run_evaluation(
            model=model,
            dataloader=val_dataloader,
            dataset=val_dataset_full,
            collator=eval_collator,
            out_dir=out_dir,
            split="val",
            device=device,
            max_target_length=cfg.tokenizer.max_target_length,
            num_beams=cfg.generation.num_beams,
            constraint_decoding=constraint_decoding,
        )

        # 2. Test Set Evaluation
        test_path = Path(cfg.data.data_dir) / "test.jsonl"
        test_metrics = {}
        if test_path.exists():
            test_dataset = S2GDataset(test_path, seed=cfg.train.seed)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=cfg.validation.batch_size,
                shuffle=False,
                num_workers=cfg.hardware.num_workers,
                collate_fn=eval_collator,
            )

            logger.info("Running post-training test evaluation...")
            test_metrics = evaluator.run_evaluation(
                model=model,
                dataloader=test_dataloader,
                dataset=test_dataset,
                collator=eval_collator,
                out_dir=out_dir / "eval_test",
                split="test",
                device=device,
                max_target_length=cfg.tokenizer.max_target_length,
                num_beams=cfg.generation.num_beams,
                constraint_decoding=constraint_decoding,
            )

        # 3. Log Final Metrics to W&B
        if wandb.run is not None:
            wandb_logs = {f"final_val/{k}": v for k, v in val_metrics.items()}
            wandb_logs.update({f"final_test/{k}": v for k, v in test_metrics.items()})
            wandb.log(wandb_logs)
            logger.info("Logged final validation and test metrics to W&B.")


if __name__ == "__main__":
    main()