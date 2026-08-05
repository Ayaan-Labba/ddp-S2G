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
from s2g.linearisation import S2GTokens, add_special_tokens_to_tokenizer
from s2g.training import (
    S2GTrainer,
    GenerateTextSamplesCallback,
    PeriodicCheckpointCallback,
    S2GEarlyStoppingCallback,
    StepTrackingCallback,
    load_run_metadata
)
from s2g.evaluation import S2GEvaluator
from s2g.scripts.config_utils import load_config, load_ent_schema, load_schema


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_config()

    # Set cuda devices
    if cfg.hardware.gpu_ids is not None:
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            logger.warning("Distributed mode: hardware.gpu_ids ignored.")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.hardware.gpu_ids))

    # Set random seeds
    set_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)

    # Create output directory
    out_dir = Path(cfg.data.output_dir)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_run_id = (
        (load_run_metadata(cfg.data.output_dir) or {}).get("wandb_run_id") 
        if cfg.checkpoint.resume_from else None
    )
    if local_rank == 0:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            id=wandb_run_id,
            resume="must" if wandb_run_id else None,
        )

    # Load dataset and schema
    train_dataset = S2GDataset(filepath=Path(cfg.data.data_dir) / "train.jsonl")
    eval_train_dataset = (
        Subset(
            train_dataset,
            rng.choice(
                len(train_dataset),
                size=max(1, int(len(train_dataset) * cfg.validation.train_percent_check)),
                replace=False,
            )
            .tolist(),
        )
        if cfg.validation.train_percent_check else None
    )
    val_dataset = S2GDataset(filepath=Path(cfg.data.data_dir) / "val.jsonl")
    eval_val_dataset = (
        Subset(
            val_dataset,
            rng.choice(
                len(val_dataset),
                size=max(1, int(len(val_dataset) * cfg.validation.percent_check)),
                replace=False,
            )
            .tolist(),
        )
        if cfg.validation.percent_check else None
    )
    rel_schema, ent_schema = load_schema(cfg.data.rel_schema), load_ent_schema(cfg.data.ent_schema)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_checkpoint or cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.pretrained_checkpoint or cfg.model.name)

    # Configure tokenizer and model with special tokens
    tokens = S2GTokens(variant=cfg.model.variant, use_rejection=cfg.graph.use_rejection, prompt=cfg.prompt.type)
    warm_start = cfg.train.warm_start and (cfg.model.pretrained_checkpoint is None)
    add_special_tokens_to_tokenizer(tokenizer=tokenizer, tokens=tokens, model=model, warm_start=warm_start)

    # Set up S2G collator
    collator = S2GCollator(
        tokenizer=tokenizer,
        ent_schema=ent_schema,
        rel_schema=rel_schema,
        config={
            'variant': cfg.model.variant,
            'max_source_length': cfg.tokenizer.max_source_length,
            'max_target_length': cfg.tokenizer.max_target_length,
            'mode': cfg.prompt.mode,
            'prompt_type': cfg.prompt.type,
            'max_ent_types': getattr(cfg.prompt, 'max_ent_types', len(ent_schema)),
            'max_rel_types': getattr(cfg.prompt, 'max_rel_types', len(rel_schema)),
            'random_prompt': cfg.prompt.random_prompt,
            'max_steps': cfg.train.max_steps,
            'pos_rate_start': getattr(cfg.prompt, 'pos_rate_start'),
            'pos_rate_end': getattr(cfg.prompt, 'pos_rate_end'),
            'neg_rate_start': getattr(cfg.prompt, 'neg_rate_start'),
            'neg_rate_end': getattr(cfg.prompt, 'neg_rate_end'),
            'pos_max_start': getattr(cfg.prompt, 'pos_max_start'),
            'pos_max_end': getattr(cfg.prompt, 'pos_max_end'),
            'neg_max_start': getattr(cfg.prompt, 'neg_max_start'),
            'neg_max_end': getattr(cfg.prompt, 'neg_max_end'),
            'use_rejection': cfg.graph.use_rejection,
            'use_nesting': cfg.graph.use_nesting,
            'random_graph': cfg.graph.random_graph,
            'seed': cfg.train.seed,
        }
    )

    # Set up training callbacks
    callbacks = [
        StepTrackingCallback(collator),
        S2GEarlyStoppingCallback(early_stopping_patience=cfg.validation.early_stopping_patience),
        PeriodicCheckpointCallback(
            output_dir=cfg.data.output_dir,
            every_n_steps=cfg.checkpoint.every_n_steps,
            wandb_run_id=wandb.run.id if wandb.run else None,
        ),
        GenerateTextSamplesCallback(
            sample_batch = [
                eval_val_dataset[i] 
                for i in rng.choice(len(eval_val_dataset), size=min(8, len(eval_val_dataset)), replace=False)
            ],
            variant = cfg.model.variant,
            tokenizer = tokenizer,
            max_target_length = cfg.tokenizer.max_target_length,
            collator = collator,
            num_beams = cfg.validation.num_beams,
            interval = cfg.callbacks.sample_generation_interval
        )
    ]

    # Set up training arguments and trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.data.output_dir,
        seed=cfg.train.seed,
        data_seed=cfg.train.seed,
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.gradient_acc_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False}, # for DDP
        max_grad_norm=cfg.train.gradient_clip_value,
        fp16=(cfg.train.precision == 'fp16'),
        bf16=(cfg.train.precision == 'bf16'),
        optim=cfg.optimizer.optim,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        adam_beta1=cfg.optimizer.adam_beta1,
        adam_beta2=cfg.optimizer.adam_beta2,
        adam_epsilon=cfg.optimizer.adam_epsilon,
        warmup_steps=cfg.scheduler.warmup_steps,
        lr_scheduler_type='constant' if cfg.scheduler.type == 'inverse_sqrt' else cfg.scheduler.type,
        eval_strategy='steps',
        eval_steps=cfg.validation.check_interval,
        per_device_eval_batch_size=cfg.validation.batch_size,
        predict_with_generate=True,
        generation_max_length=cfg.tokenizer.max_target_length,
        generation_num_beams=cfg.validation.num_beams,
        dataloader_num_workers=cfg.hardware.num_workers,
        dataloader_persistent_workers=cfg.hardware.persistent_workers,
        save_strategy='steps',
        save_steps=cfg.validation.check_interval,
        save_total_limit=cfg.checkpoint.save_top_k + 1,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.validation.early_stopping_metric,
        greater_is_better=True,
        logging_strategy='steps',
        logging_steps=cfg.train.steps_per_log,
        report_to='wandb',
        run_name=cfg.wandb.run_name,
        remove_unused_columns=False
        )
    
    trainer = S2GTrainer(
        train_dataset=train_dataset,
        eval_train_dataset=eval_train_dataset,
        eval_dataset=eval_val_dataset,
        data_collator=collator,
        ent_schema=ent_schema,
        rel_schema=rel_schema,
        model=model,
        variant=cfg.model.variant,
        processing_class=tokenizer,
        tokens=tokens,
        callbacks=callbacks,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from)

    # Post-training evaluation
    if trainer.is_world_process_zero():
        best_dir = out_dir / "best_model"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        (best_dir / "variant.txt").write_text(cfg.model.variant, encoding="utf-8")

        model.eval()
        device = next(model.parameters()).device

        eval_collator = collator.to_eval_mode()
        evaluator = S2GEvaluator(
            tokenizer=tokenizer,
            tokens=tokens,
            variant=cfg.model.variant,
            rel_schema=rel_schema,
            ent_schema=ent_schema,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.validation.batch_size,
            shuffle=False,
            num_workers=cfg.hardware.num_workers,
            collate_fn=eval_collator,
        )

        constraint_decoding = getattr(cfg.generation, 'constraint_decoding', False)
        logger.info("Running post-training validation evaluation...")
        val_metrics = evaluator.run_evaluation(
            model=model,
            dataloader=val_dataloader,
            dataset=val_dataset,
            out_dir=out_dir,
            split='val',
            device=device,
            max_target_length=cfg.tokenizer.max_target_length,
            num_beams=cfg.generation.num_beams,
            constraint_decoding=constraint_decoding,
        )

        # Test Set Evaluation
        test_path = Path(cfg.data.data_dir) / "test.jsonl"
        test_metrics = {}
        if test_path.exists():
            test_dataset = S2GDataset(test_path, seed=cfg.train.seed)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=cfg.evaluation.batch_size,
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
                split='test',
                device=device,
                max_target_length=cfg.tokenizer.max_target_length,
                num_beams=cfg.generation.num_beams,
                constraint_decoding=constraint_decoding,
            )

        # Log Final Metrics to W&B
        if wandb.run is not None:
            wandb_logs = {f"final_val/{k}": v for k, v in val_metrics.items()}
            wandb_logs.update({f"final_test/{k}": v for k, v in test_metrics.items()})
            wandb.log(wandb_logs)
            logger.info("Logged final validation and test metrics to W&B.")


if __name__ == "__main__":
    main()