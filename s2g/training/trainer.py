"""
S2G custom Seq2SeqTrainer for single-variant fine-tuning and evaluation.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

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
        self.dedup              = kwargs.pop('dedup', True)
        self._s2g_gold_dataset  = None
        # One persistent-worker loader per eval dataset, keyed by id(dataset) with
        # the dataset held alongside so the id cannot be recycled.
        self._s2g_eval_loaders: Dict[int, Tuple[Any, DataLoader]] = {}

        super().__init__(compute_metrics=self.compute_metrics_hf, **kwargs)

        self.evaluator = S2GEvaluator(
            variant=self.variant,
            tokenizer=self.processing_class,
            tokens=self.tokens,
            rel_schema=self.rel_schema,
            ent_schema=self.ent_schema,
            dedup=self.dedup,
        )

    def create_scheduler(self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
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

        Swaps the collator and defers to ``Trainer`` so that distributed sharding,
        pinned memory and prefetching are preserved.

        ``Trainer`` caches every non-string eval dataset under the single key
        ``"eval"``, so with two eval datasets in play — the validation set and the
        train subset, which ``evaluate`` visits back to back on every check — its
        cache always holds the wrong one. Clearing that cache per call made the
        loaders correct but rebuilt **both of them at every validation check**,
        which defeats ``dataloader_persistent_workers`` entirely: a 100-check run
        spawned several hundred short-lived worker generations instead of reusing
        two sets.
        Under the Python 3.14 ``forkserver`` default each of those spawns also
        pickles the dataset, tokenizer and collator into every worker, so the churn
        is paid in memory as well as in time.

        The fix is a cache keyed by the dataset itself, so each eval dataset keeps
        its own loader and its own persistent workers for the run's lifetime.
        """
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        # ``compute_metrics_hf`` receives only tensors, so remember which dataset
        # this loop is about to iterate — that is where gold now comes from.
        self._s2g_gold_dataset = dataset

        persistent = getattr(self.args, 'dataloader_persistent_workers', False)
        cached = self._s2g_eval_loaders.get(id(dataset))
        if persistent and cached is not None and cached[0] is dataset:
            return cached[1]

        train_collator = self.data_collator
        self.data_collator = train_collator.to_eval_mode()
        try:
            # Evict HF's single-key cache so it builds for *this* dataset rather
            # than handing back the other one's loader.
            getattr(self, '_eval_dataloaders', {}).pop('eval', None)
            dataloader = super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = train_collator

        if persistent:
            # Hold the dataset alongside the loader: it keeps the object alive, so
            # its id cannot be recycled by another dataset later in the run.
            self._s2g_eval_loaders[id(dataset)] = (dataset, dataloader)
        return dataloader

    def evaluate(self, eval_dataset: Any = None, **gen_kwargs: Any) -> Dict[str, float]:
        """
        Override evaluate function to also evaluate on a subset of the train set.
        """
        all_metrics = super().evaluate(eval_dataset=eval_dataset, **gen_kwargs)
        if self.eval_train_dataset:
            early_stopping_callbacks = [cb for cb in self.callback_handler.callbacks if isinstance(cb, EarlyStoppingCallback)]
            for cb in early_stopping_callbacks:
                self.callback_handler.callbacks.remove(cb)

            train_metrics: Dict[str, float] = {}
            try:
                train_metrics = super().evaluate(
                    eval_dataset=self.eval_train_dataset,
                    metric_key_prefix='eval_train',
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
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)

        pred_blocks: List[List[EntityBlock]] = [self.evaluator.parse_text(p)[0] for p in decoded_preds]

        instances = self.gold_instances(len(decoded_preds))
        if instances is None:
            # Fall back to the old label round trip: without the instances there is
            # no offset annotation and no source tokens, so only text metrics are
            # available. Better a narrower report than a silently misaligned one.
            labels = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)
            gold_blocks = [self.evaluator.parse_text(g)[0] for g in decoded_labels]
            return self.evaluator.compute_final_metrics(pred_blocks, gold_blocks, include_macro=False)

        # Validation reports micro text metrics only. The full set is ~40 keys per
        # check, which swamps a W&B run without adding any training signal; the
        # offset and macro tracks are computed once, at end-of-run evaluation.
        gold_blocks = [self.evaluator.build_gold(inst, with_offsets=False)[0] for inst in instances]

        return self.evaluator.compute_final_metrics(pred_blocks, gold_blocks, include_macro=False)

    def gold_instances(self, num_preds: int) -> Optional[List[Dict[str, Any]]]:
        """
        The instances backing the eval loop that just ran, in prediction order.

        Returns ``None`` whenever that order cannot be trusted. Under DDP the
        distributed eval sampler shards strided and the gathered predictions no
        longer follow dataset order, so positional pairing would score every
        prediction against the wrong gold.
        """
        dataset = self._s2g_gold_dataset
        if dataset is None:
            logger.warning("No eval dataset recorded; falling back to label-parsed gold.")
            return None

        if self.args.world_size > 1:
            logger.warning(
                "Distributed evaluation (world_size=%d): prediction order does not follow "
                "dataset order, falling back to label-parsed gold and text-only metrics.",
                self.args.world_size,
            )
            return None

        if len(dataset) != num_preds:
            logger.warning(
                "Eval dataset holds %d instances but %d predictions were returned; "
                "falling back to label-parsed gold.", len(dataset), num_preds,
            )
            return None

        return [dataset[i] for i in range(num_preds)]