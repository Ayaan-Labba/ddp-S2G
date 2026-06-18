"""
Training script for the S2G model (used for both Fine-tuning and Pre-training).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import Subset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainingArguments, set_seed,
)

from s2g.data import S2GCollator, S2GDataset
from s2g.evaluation import (
    GenerateTextSamplesCallback, PeriodicCheckpointCallback, 
    StepTrackingCallback, S2GEarlyStoppingCallback, load_run_metadata
)
from s2g.linearisation import S2GTokens, add_special_tokens_to_tokenizer, VARIANT_TO_TASKS
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

    wandb_run_id = (load_run_metadata(cfg.data.output_dir) or {}).get("wandb_run_id") if cfg.checkpoint.resume_from else None

    if local_rank == 0:
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.wandb.run_name, id=wandb_run_id, resume="must" if wandb_run_id else None)

    rel_schema, entity_schema = load_schema(cfg.data.schema_file), load_entity_schema(cfg.data.entity_schema_file)
    train_dataset = S2GDataset(Path(cfg.data.data_dir) / "train.jsonl", seed=cfg.train.seed)
    val_dataset = S2GDataset(Path(cfg.data.data_dir) / "val.jsonl", subset_fraction=cfg.validation.percent_check if cfg.validation.percent_check < 1.0 else None, seed=cfg.train.seed)
    
    train_eval_dataset = Subset(train_dataset, np.random.default_rng(cfg.train.seed).choice(len(train_dataset), size=max(1, int(len(train_dataset) * cfg.validation.train_eval_percent_check)), replace=False).tolist()) if cfg.validation.train_eval_percent_check else None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_checkpoint or cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.pretrained_checkpoint or cfg.model.name)
    # Ensure model parameters are explicitly cast to the configured precision
    if cfg.train.precision == "fp32":
        model = model.float()
    elif cfg.train.precision == "bf16":
        model = model.to(torch.bfloat16)
    elif cfg.train.precision == "fp16":
        model = model.half()
        
    tokens = S2GTokens(cfg.model.model_variant, use_rejection=cfg.sel.use_rejection)
    add_special_tokens_to_tokenizer(tokenizer, tokens, model, warm=cfg.sel.warm_start)

    tasks = VARIANT_TO_TASKS[cfg.model.model_variant]

    collator = S2GCollator(tokenizer, entity_schema, rel_schema, {
        "model_variant": cfg.model.model_variant, "max_source_length": cfg.tokenization.max_source_length, "max_target_length": cfg.tokenization.max_target_length,
        "max_ent_types": cfg.ssi.max_ent_types or len(entity_schema), "max_rel_types": cfg.ssi.max_rel_types or len(rel_schema),
        "random_prompt": cfg.ssi.random_prompt, "random_sel": cfg.sel.random_sel,
        "positive_rate_start": getattr(cfg.ssi, "positive_rate_start", 0.9),
        "positive_rate_end": getattr(cfg.ssi, "positive_rate_end", 0.9),
        "negative_rate_start": getattr(cfg.ssi, "negative_rate_start", 0.1),
        "negative_rate_end": getattr(cfg.ssi, "negative_rate_end", 0.1),
        "pos_max_start": getattr(cfg.ssi, "pos_max_start", 1),
        "pos_max_end": getattr(cfg.ssi, "pos_max_end", 20),
        "negative_max_start": getattr(cfg.ssi, "negative_max_start", 1),
        "negative_max_end": getattr(cfg.ssi, "negative_max_end", 20),
        "tasks": tasks, "mode": cfg.ssi.mode, "max_steps": cfg.train.max_steps,
        "use_rejection": cfg.sel.use_rejection,
        "use_nesting": cfg.sel.use_nesting,
        "ssi_prompt": cfg.ssi.ssi_prompt,
        "data_dir": cfg.data.data_dir,
    })

    callbacks = [
        StepTrackingCallback(collator), S2GEarlyStoppingCallback(early_stopping_patience=cfg.validation.early_stopping_patience),
        PeriodicCheckpointCallback(output_dir=cfg.data.output_dir, every_n_steps=cfg.checkpoint.every_n_steps, wandb_run_id=wandb.run.id if wandb.run else None),
        GenerateTextSamplesCallback(tokenizer, [val_dataset[i] for i in range(min(8, len(val_dataset)))], collator, cfg.model.model_variant, cfg.callbacks.sample_generation_interval, cfg.generation.num_beams, cfg.tokenization.max_target_length)
    ]

    best_metric = cfg.validation.early_stopping_metric

    trainer = S2GTrainer(
        scheduler_type=cfg.scheduler.type, model_variant=cfg.model.model_variant, tokens=tokens, entity_schema=entity_schema, rel_schema=rel_schema, train_eval_dataset=train_eval_dataset,
        eval_cfg={"max_source_length": cfg.tokenization.max_source_length, "max_target_length": cfg.tokenization.max_target_length, "eval_batch_size": cfg.validation.batch_size, "eval_beams": cfg.generation.num_beams, "ssi_prompt": cfg.ssi.ssi_prompt},
        model=model, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=collator, processing_class=tokenizer, callbacks=callbacks,
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
            logging_steps=10, 
            report_to="wandb", 
            run_name=cfg.wandb.run_name, 
            remove_unused_columns=False, 
            label_names=[]
        )
    )

    trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from)
    
    # Secure post-training operations strictly to Rank 0 to prevent DDP JSON corruption
    if trainer.is_world_process_zero():
        best_dir = out_dir / "best_model"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        (best_dir / "model_variant.txt").write_text(cfg.model.model_variant, encoding="utf-8")
        with open(best_dir / "tasks.json", "w", encoding="utf-8") as f:
            json.dump(tasks, f)

    val_metrics = trainer.evaluate(eval_dataset=S2GDataset(Path(cfg.data.data_dir) / "val.jsonl", seed=cfg.train.seed))

    if trainer.is_world_process_zero():
        with open(out_dir / "val_metrics.json", "w", encoding="utf-8") as f: 
            json.dump(val_metrics, f, indent=2)

        if getattr(cfg.evaluation, "evaluate_config", None):
            logger.info(f"Running post-training evaluation using config: {cfg.evaluation.evaluate_config}")
            
            # Explicitly free GPU memory to prevent OOM in the subprocess
            del trainer
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            import subprocess
            import sys
            eval_cmd = [
                sys.executable, "-m", "s2g.scripts.evaluate",
                "--config", cfg.evaluation.evaluate_config,
                f"model.pretrained_checkpoint={str(best_dir)}",
                f"data.data_dir={cfg.data.data_dir}",
                f"data.schema_file={cfg.data.schema_file}",
                f"data.entity_schema_file={cfg.data.entity_schema_file}",
                f"data.output_dir={str(out_dir / 'eval_test')}",
                "evaluation.split=test"
            ]
            try:
                subprocess.run(eval_cmd, check=True)
                logger.info("Post-training evaluation completed successfully.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Post-training evaluation failed with exit code {e.returncode}")

if __name__ == "__main__":
    main()