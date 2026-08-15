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

from s2g.linearisation import EntityBlock, VALID_VARIANTS
from s2g.evaluation import S2GEvaluator

logger = logging.getLogger(__name__)


class S2GTrainer(Seq2SeqTrainer):
    def __init__(self, **kwargs: Any) -> None:
        self.variant = kwargs.pop('variant')
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"Model variant must be one of {VALID_VARIANTS}, got {self.variant!r}.")

        self.eval_train_dataset = kwargs.pop('eval_train_dataset', None)
        self.ent_schema         = kwargs.pop('ent_schema', [])
        self.rel_schema         = kwargs.pop('rel_schema', [])
        self.tokens             = kwargs.pop('tokens')
        self.scheduler_type     = kwargs.pop('scheduler_type', None)

        super().__init__(compute_metrics=self.compute_metrics_hf, **kwargs)

        self.evaluator = S2GEvaluator(
            variant=self.variant,
            tokenizer=self.processing_class,
            tokens=self.tokens,
            rel_schema=self.rel_schema,
            ent_schema=self.ent_schema,
        )

    def create_scheduler(self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """
        Custom Inverse Square Root Scheduler
        """
        if self.lr_scheduler is not None:
            return

        if self.scheduler_type == 'inverse_sqrt' or self.args.lr_scheduler_type == 'inverse_sqrt':
            opt = optimizer if optimizer else self.optimizer
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

    def get_eval_dataloader(self, eval_dataset: Optional[Any] = None) -> DataLoader:
        """
        Override evaluation dataloader to use budget-mode collation.
        """
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        collator = self.data_collator.to_eval_mode()

        return DataLoader(
            dataset=dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=self.args.dataloader_num_workers,
            shuffle=False,
        )

    def evaluate(self, eval_dataset: Any = None, **gen_kwargs: Any) -> Dict[str, float]:
        """
        Override evaluate function to also evaluate on a subset of the train set.
        """
        all_metrics = super().evaluate(eval_dataset=eval_dataset, **gen_kwargs)
        if self.eval_train_dataset:
            early_stopping_callbacks = [cb for cb in self.callback_handler.callbacks if isinstance(cb, EarlyStoppingCallback)]
            for cb in early_stopping_callbacks:
                self.callback_handler.callbacks.remove(cb)

            try:
                train_metrics = super().evaluate(
                    eval_dataset=self.eval_train_dataset,
                    metric_key_prefix='train',
                    **gen_kwargs,
                )

            finally:
                for cb in early_stopping_callbacks:
                    self.callback_handler.callbacks.append(cb)

                all_metrics.update(train_metrics)

        return all_metrics

    def compute_metrics_hf(self, eval_preds: Any) -> Dict[str, float]:
        preds, label_ids = eval_preds.predictions, eval_preds.label_ids
        if isinstance(preds, tuple): # for predict_with_generate=True
            preds = preds[0]

        tokenizer = self.processing_class
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

        pred_blocks: List[List[EntityBlock]] = [self.evaluator.parse_text(p)[0] for p in decoded_preds]
        gold_blocks: List[List[EntityBlock]] = [self.evaluator.parse_text(g)[0] for g in decoded_labels]

        return self.evaluator.compute_final_metrics(pred_blocks, gold_blocks)