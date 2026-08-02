"""
S2G custom Seq2SeqTrainer for single-variant fine-tuning and evaluation.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput

from s2g.evaluation import S2GEvaluator
from s2g.linearisation import EntityBlock

logger = logging.getLogger(__name__)


class S2GTrainer(Seq2SeqTrainer):
    def __init__(self, **kwargs: Any) -> None:
        self._variant = kwargs.pop("model_variant")
        if self._variant not in {'boundary_re', 'boundary_joint', 're', 'joint'}:
            raise ValueError(
                f"model_variant must be one of ['boundary_re', 'boundary_joint', 're', 'joint'], got {self._variant!r}."
            )

        self._tokens             = kwargs.pop("tokens")
        self._entity_schema      = kwargs.pop("entity_schema", [])
        self._rel_schema         = kwargs.pop("rel_schema", [])
        self._eval_cfg           = kwargs.pop("eval_cfg")
        self._train_eval_dataset = kwargs.pop("train_eval_dataset", None)
        self._scheduler_type     = kwargs.pop("scheduler_type", "inverse_sqrt")

        super().__init__(compute_metrics=self._compute_metrics_hf, **kwargs)

        self._max_tgt    = self._eval_cfg["max_target_length"]
        self._eval_beams = self._eval_cfg["eval_beams"]

        self._evaluator = S2GEvaluator(
            tokenizer=self.processing_class,
            tokens=self._tokens,
            model_variant=self._variant,
            rel_schema=self._rel_schema,
            entity_schema=self._entity_schema,
        )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        if self.lr_scheduler is not None:
            return

        if self._scheduler_type == "inverse_sqrt":
            opt = optimizer or self.optimizer
            warmup = self.args.get_warmup_steps(num_training_steps)
            self.lr_scheduler = LambdaLR(
                opt,
                lambda step: (
                    max(step, 1) / max(warmup, 1)
                    if max(step, 1) < warmup
                    else math.sqrt(warmup / max(step, 1))
                ),
            )
        else:
            super().create_scheduler(num_training_steps, optimizer)

    # ------------------------------------------------------------------
    # High-Level Evaluation & Prediction
    # ------------------------------------------------------------------

    def get_eval_dataloader(self, eval_dataset: Optional[Any] = None) -> DataLoader:
        """
        Ensures evaluation dataloaders use deterministic budget-mode collation.
        """
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_collator = (
            self.data_collator.to_eval_mode()
            if hasattr(self.data_collator, "to_eval_mode")
            else self.data_collator
        )

        return DataLoader(
            dataset,
            batch_size=self._eval_cfg["eval_batch_size"],
            collate_fn=eval_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: Any = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs: Any,
    ) -> Dict[str, float]:
        self.args.predict_with_generate = True
        self.args.generation_max_length = self._max_tgt
        self.args.generation_num_beams  = self._eval_beams

        all_metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **gen_kwargs,
        )
        if self._train_eval_dataset and metric_key_prefix == "eval":
            early_stopping_callbacks = [cb for cb in self.callback_handler.callbacks if isinstance(cb, EarlyStoppingCallback)]
            for cb in early_stopping_callbacks:
                self.callback_handler.callbacks.remove(cb)

            try:
                train_metrics = super().evaluate(
                    eval_dataset=self._train_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix="train",
                    **gen_kwargs,
                )

            finally:
                for cb in early_stopping_callbacks:
                    self.callback_handler.callbacks.append(cb)

                all_metrics.update(train_metrics)

        return all_metrics

    def predict(
        self,
        test_dataset: Any,
        ignore_keys: Any = None,
        metric_key_prefix: str = "test",
        **gen_kwargs: Any,
    ) -> PredictionOutput:
        self.args.predict_with_generate = True
        self.args.generation_max_length = self._max_tgt
        self.args.generation_num_beams  = self._eval_beams
        return super().predict(
            test_dataset=test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **gen_kwargs,
        )

    # ------------------------------------------------------------------
    # HF Compute Metrics
    # ------------------------------------------------------------------

    def _compute_metrics_hf(self, eval_preds: Any) -> Dict[str, float]:
        preds, label_ids = eval_preds.predictions, eval_preds.label_ids

        if isinstance(preds, tuple):
            preds = preds[0]

        tokenizer = self.processing_class
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

        pred_blocks: List[List[EntityBlock]] = [self._evaluator.parse_text(p)[0] for p in decoded_preds]
        gold_blocks: List[List[EntityBlock]] = [self._evaluator.parse_text(g)[0] for g in decoded_labels]

        return self._evaluator.compute_final_metrics(pred_blocks, gold_blocks)