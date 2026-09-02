"""
Training script for the S2G model with integrated streaming evaluation.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Optional

import numpy as np
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, set_seed
from s2g.data import S2GCollator, S2GDataset, set_parent_death_signal
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


def configure_dataloader_start_method(method: Optional[str]) -> None:
    """
    Pin the multiprocessing start method used to launch DataLoader workers.

    Defaults to ``'fork'``, which is what this pipeline ran on until Python 3.14
    flipped the Linux default to ``'forkserver'`` (gh-84559).

    The two differ enormously in what a worker costs, and it is the *interpreter*
    that dominates, not the data. A forked worker shares the parent's already
    imported torch and transformers pages copy-on-write; a forkserver worker is a
    fresh interpreter that re-imports them privately, roughly 250 MB apiece.
    Measured over three loaders of four workers, whole-tree PSS:

        fork                    +   81 MB
        forkserver              + 3035 MB
        forkserver + preload    + 1033 MB

    On a memory-tight host those extra gigabytes arrive exactly when the first
    validation check builds its loaders, and the OOM killer takes the parent
    mid-spawn — which the forkserver child then reports as a truncated payload:

        _pickle.UnpicklingError: pickle data was truncated

    ``None`` leaves the interpreter default alone. ``'forkserver'`` / ``'spawn'``
    pin one explicitly; ``preload_forkserver_modules`` recovers part of the cost
    when forkserver is in force. The price of ``fork`` is a ``DeprecationWarning``
    about forking a multi-threaded process.
    """
    if not method:
        return

    available = multiprocessing.get_all_start_methods()
    if method not in available:
        logger.warning(
            "Start method %r is unavailable on this platform (have %s); keeping %r.",
            method, available, multiprocessing.get_start_method(),
        )
        return

    if multiprocessing.get_start_method(allow_none=True) != method:
        try:
            multiprocessing.set_start_method(method, force=True)
        except RuntimeError as exc:
            logger.warning("Could not set the start method to %r: %s", method, exc)
            return
        logger.info("DataLoader worker start method set to %r.", method)


def preload_forkserver_modules() -> None:
    """
    Import the heavy modules once in the forkserver process, if that is the method
    in force.

    A forkserver child is forked from the server process, so anything the *server*
    has already imported is shared copy-on-write instead of being re-imported per
    worker. For 12 workers this measured 3035 MB without preloading against
    1033 MB with it — still short of ``fork``'s 81 MB, but enough to matter on a
    memory-tight host.
    """
    try:
        effective = multiprocessing.get_start_method()
    except RuntimeError:
        return

    if effective != 'forkserver':
        return

    try:
        multiprocessing.set_forkserver_preload(['torch', 'transformers', 's2g.data'])
    except Exception:
        logger.debug("Could not set forkserver preload modules.", exc_info=True)
        return

    logger.info("Preloading torch / transformers into the forkserver process.")


def log_host_memory(required_gb: float = 4.0) -> None:
    """
    Report host memory before training, and warn when there is not enough of it.

    A run needs roughly 3 GB of host RAM (torch and CUDA allocations, the model, and
    the state-dict copy taken during each checkpoint save). When the box is already
    full the kernel's OOM killer takes the process with no traceback — just
    ``Killed``, usually mid-save because that is the next large allocation. Printing
    the numbers up front turns that silence into something diagnosable.

    A common cause of a mysteriously full box is orphaned DataLoader workers left
    behind by an earlier run: SIGKILL runs no cleanup, so the workers are reparented
    to init and keep their memory. ``pgrep -fc s2g.scripts.train`` counts them.
    """
    try:
        meminfo = {}
        with open('/proc/meminfo', encoding='utf-8') as f:
            for line in f:
                key, _, rest = line.partition(':')
                meminfo[key] = int(rest.split()[0]) / 1024 ** 2      # kB -> GiB
    except (OSError, ValueError, IndexError):
        return                                                       # not Linux, or unreadable

    total = meminfo.get('MemTotal')
    available = meminfo.get('MemAvailable')
    if total is None or available is None:
        return

    logger.info("Host memory: %.1f GiB total, %.1f GiB available.", total, available)
    if available < required_gb:
        logger.warning(
            "Only %.1f GiB of host memory is available but a run needs roughly %.1f GiB. "
            "Training will most likely be OOM-killed (a bare 'Killed', no traceback). "
            "Check for orphaned workers from an earlier run: "
            "`ps aux --sort=-rss | head -20`, `pgrep -fc s2g.scripts.train`.",
            available, required_gb,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_config()
    log_host_memory()

    # Must happen before any DataLoader spins up workers.
    configure_dataloader_start_method(cfg.hardware.dataloader_start_method)
    preload_forkserver_modules()

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

    # Initialize wandb and log YAML configuration
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
            resume="must" if wandb_run_id else None
        )
        cfg_yaml_path = out_dir / "config.yaml"
        cfg_yaml_path.write_text(OmegaConf.to_yaml(cfg), encoding='utf-8')
        wandb.save(str(cfg_yaml_path), base_path=str(out_dir))

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
    tokens = S2GTokens(
        variant=cfg.model.variant, use_rejection=cfg.graph.use_rejection, markers=cfg.graph.markers
    )
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
            'prompt_style': cfg.prompt.style,
            'use_rejection': cfg.graph.use_rejection,
            'markers': cfg.graph.markers,
            'nesting': cfg.graph.nesting,
            'joint_tail_type': cfg.graph.joint_tail_type,
            'dedup': cfg.graph.dedup,
            'random_graph': cfg.graph.random_graph,
            'seed': cfg.train.seed,
        }
    )

    # Validation emits micro text metrics only (see S2GTrainer.compute_metrics_hf),
    # so selecting the best model on an offset or macro metric can no longer work.
    best_metric = cfg.validation.early_stopping_metric
    if best_metric.startswith(('offset_', 'macro_', 'eval_offset_', 'eval_macro_')):
        raise ValueError(
            f"validation.early_stopping_metric={best_metric!r} is not computed during "
            "training: offset_* and macro_* metrics are reported only by the "
            "end-of-run evaluation. Use a micro text metric such as 'strict_f1' or "
            "'boundary_f1'."
        )

    if cfg.checkpoint.save_only_model and cfg.checkpoint.resume_from:
        logger.warning(
            "checkpoint.save_only_model drops optimizer / scheduler / RNG state, so "
            "resuming from %s will restart the optimizer rather than continue it.",
            cfg.checkpoint.resume_from,
        )

    # Set up training callbacks
    callbacks = [
        StepTrackingCallback(collator),
        S2GEarlyStoppingCallback(early_stopping_patience=cfg.validation.early_stopping_patience),
        # `checkpoint.every_n_steps: null` leaves checkpointing to the validation
        # checks; the callback still records run metadata for W&B resumption.
        PeriodicCheckpointCallback(
            output_dir=cfg.data.output_dir,
            every_n_steps=cfg.checkpoint.every_n_steps,
            wandb_run_id=wandb.run.id if wandb.run else None,
        ),
    ]

    # Samples are drawn from the evaluation split itself and held as a Subset rather
    # than copied out as instances, so the logged table stays tied to the dataset the
    # trainer validates on and passes through the same indexing.
    if eval_val_dataset is not None and len(eval_val_dataset) > 0:
        sample_indices = rng.choice(
            len(eval_val_dataset), size=min(8, len(eval_val_dataset)), replace=False
        ).tolist()
        callbacks.append(
            GenerateTextSamplesCallback(
                sample_dataset=Subset(eval_val_dataset, sample_indices),
                variant=cfg.model.variant,
                tokenizer=tokenizer,
                collator=collator,
                interval=cfg.callbacks.sample_generation_interval,
            )
        )
    else:
        logger.warning(
            "validation.percent_check yields no evaluation subset; "
            "skipping the sample generation callback."
        )

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
        save_only_model=cfg.checkpoint.save_only_model,
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
        scheduler_type=cfg.scheduler.type,
        dedup=cfg.graph.dedup,
    )

    trainer.train(resume_from_checkpoint=cfg.checkpoint.resume_from)

    # Post-training evaluation
    if trainer.is_world_process_zero():
        best_dir = out_dir / "best_model"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        (best_dir / "variant.txt").write_text(cfg.model.variant, encoding="utf-8")

        # Persist every setting that changes how targets are linearised, so that
        # standalone evaluation cannot silently score against a different format.
        (best_dir / "s2g_format.json").write_text(
            json.dumps({
                'variant':         cfg.model.variant,
                'prompt_type':     cfg.prompt.type,
                'style':           cfg.prompt.style,
                'use_rejection':   cfg.graph.use_rejection,
                'markers':         cfg.graph.markers,
                'nesting':         cfg.graph.nesting,
                'joint_tail_type': cfg.graph.joint_tail_type,
                'dedup':           cfg.graph.dedup,
                # The role tokens are dedicated vocabulary now, so a checkpoint
                # trained under one map cannot be scored under another.
                'token_strs':      dict(tokens.token_strs),
                'max_ent_types':   cfg.prompt.max_ent_types,
                'max_rel_types':   cfg.prompt.max_rel_types,
            }, indent=2),
            encoding="utf-8",
        )

        model.eval()
        # Training with gradient checkpointing leaves use_cache disabled, which
        # makes post-training generation needlessly slow.
        model.config.use_cache = True
        device = next(model.parameters()).device

        eval_collator = collator.to_eval_mode()
        evaluator = S2GEvaluator(
            tokenizer=tokenizer,
            tokens=tokens,
            variant=cfg.model.variant,
            rel_schema=rel_schema,
            ent_schema=ent_schema,
            dedup=cfg.graph.dedup,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.validation.batch_size,
            shuffle=False,
            num_workers=cfg.hardware.num_workers,
            collate_fn=eval_collator,
            worker_init_fn=set_parent_death_signal,
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
            length_penalty=getattr(cfg.generation, 'length_penalty', None),
            early_stopping=getattr(cfg.generation, 'early_stopping', None),
            no_repeat_ngram_size=getattr(cfg.generation, 'no_repeat_ngram_size', None)
        )

        # Test Set Evaluation
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
                worker_init_fn=set_parent_death_signal,
            )

            logger.info("Running post-training test evaluation...")
            test_metrics = evaluator.run_evaluation(
                model=model,
                dataloader=test_dataloader,
                dataset=test_dataset,
                out_dir=out_dir / "eval_test",
                split='test',
                device=device,
                max_target_length=cfg.tokenizer.max_target_length,
                num_beams=cfg.generation.num_beams,
                constraint_decoding=constraint_decoding,
                length_penalty=getattr(cfg.generation, 'length_penalty', None),
                early_stopping=getattr(cfg.generation, 'early_stopping', None),
                no_repeat_ngram_size=getattr(cfg.generation, 'no_repeat_ngram_size', None)
            )

        # Log Final Metrics to W&B
        if wandb.run is not None:
            wandb_logs = {f"final_val/{k}": v for k, v in val_metrics.items()}
            wandb_logs.update({f"final_test/{k}": v for k, v in test_metrics.items()})
            wandb.log(wandb_logs)
            logger.info("Logged final validation and test metrics to W&B.")


if __name__ == "__main__":
    main()