"""
S2G custom Seq2SeqTrainer for multi-task fine-tuning.

Notes
-----
- compute_metrics_for_task is called with rel_schema / entity_schema so that
  (a) hallucinated types are discarded and (b) macro F1 is computed per-type in
  REBEL style rather than per-instance.
- _compute_metrics_hf dispatches on the model variant: "re" uses task="re"
  (boundary + strict F1), "boundary_re" uses task="boundary_re" (boundary only),
  and the joint variants compute their full metric sets.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import EarlyStoppingCallback, Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput

from s2g.evaluation.metrics import compute_metrics_for_task
from s2g.linearisation import (
    EntityBlock, VARIANT_TO_TASKS, extract_triplets, parse_sel,
)

logger = logging.getLogger(__name__)

_PIPELINE_TASK_KEYS       = ("re", "boundary_re")
_BOUNDARY_JOINT_TASK_KEYS = ("boundary_joint", "joint")
_ALL_TASK_KEYS            = _PIPELINE_TASK_KEYS + _BOUNDARY_JOINT_TASK_KEYS


def _assemble_joint_quintuples(entities: List[EntityBlock]) -> List[Tuple[str, str, str, str, str]]:
    t_map = {e["text"]: e.get("type", "") for e in entities}
    return [
        (
            ent["text"],
            ent.get("type", ""),
            rel["type"],
            rel["tail"],
            rel.get("tail_type") or t_map.get(rel["tail"], ""),
        )
        for ent in entities for rel in ent["relations"]
    ]


class S2GTrainer(Seq2SeqTrainer):
    def __init__(self, **kwargs: Any) -> None:
        self._variant            = kwargs.pop("model_variant")
        if self._variant not in VARIANT_TO_TASKS:
            raise ValueError(
                f"model_variant must be one of {list(VARIANT_TO_TASKS)}, got {self._variant!r}."
            )

        self._tokens             = kwargs.pop("tokens")
        self._entity_schema      = kwargs.pop("entity_schema", [])
        self._rel_schema         = kwargs.pop("rel_schema", [])
        self._eval_cfg           = kwargs.pop("eval_cfg")
        self._train_eval_dataset = kwargs.pop("train_eval_dataset", None)
        self._scheduler_type     = kwargs.pop("scheduler_type", "inverse_sqrt")
        kwargs.pop("tasks", None)

        super().__init__(compute_metrics=self._compute_metrics_hf, **kwargs)

        self._max_tgt    = self._eval_cfg["max_target_length"]
        self._eval_beams = self._eval_cfg["eval_beams"]

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
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        active_keys = [k for k in _ALL_TASK_KEYS if f"{k}_input_ids" in inputs]
        if not active_keys:
            raise ValueError(f"training_step: no task keys found. Expected from: {_ALL_TASK_KEYS}.")

        total_loss = 0.0

        for k in active_keys:
            task_inputs = {
                "input_ids":      inputs[f"{k}_input_ids"],
                "attention_mask": inputs[f"{k}_attention_mask"],
                "labels":         inputs[f"{k}_labels"],
            }

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, task_inputs)

            loss = loss / len(active_keys)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if self.args.gradient_accumulation_steps > 1 and not getattr(self, "deepspeed", False):
                loss = loss / self.args.gradient_accumulation_steps

            if hasattr(self, "accelerator"):
                self.accelerator.backward(loss)
            elif getattr(self, "do_grad_scaling", False):
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.detach()

        return total_loss

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    # ------------------------------------------------------------------
    # Prediction / eval step
    # ------------------------------------------------------------------

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        active_keys = [k for k in _ALL_TASK_KEYS if f"{k}_input_ids" in inputs]
        if not active_keys:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **kwargs)

        k = active_keys[0]
        standard_inputs = {
            "input_ids":      inputs[f"{k}_input_ids"],
            "attention_mask": inputs[f"{k}_attention_mask"],
            "labels":         inputs.get(f"{k}_labels"),
        }

        return super().prediction_step(model, standard_inputs, prediction_loss_only, ignore_keys, **kwargs)

    # ------------------------------------------------------------------
    # High-level evaluate / predict
    # ------------------------------------------------------------------

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: Any = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs: Any,
    ) -> Dict[str, float]:
        self.args.predict_with_generate  = True
        self.args.generation_max_length  = self._max_tgt
        self.args.generation_num_beams   = self._eval_beams

        all_metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **gen_kwargs,
        )
        if self._train_eval_dataset and metric_key_prefix == "eval":
            # Temporarily remove EarlyStoppingCallback so that logging with
            # prefix="train" doesn't interfere with early-stopping logic.
            early_stopping_callbacks = [
                cb for cb in self.callback_handler.callbacks
                if isinstance(cb, EarlyStoppingCallback)
            ]
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
    # HF-native compute_metrics
    # ------------------------------------------------------------------

    def _compute_metrics_hf(self, eval_preds: Any) -> Dict[str, float]:
        import numpy as np
        preds, label_ids = eval_preds.predictions, eval_preds.label_ids

        if isinstance(preds, tuple):
            preds = preds[0]

        tokenizer      = self.processing_class
        preds          = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=False)

        labels         = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

        specials = [tok for tok in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token) if tok]

        def clean_text(text: str) -> str:
            for tok in specials:
                text = text.replace(tok, "")
            return " ".join(text.split())

        pred_entities: List[List[EntityBlock]] = []
        gold_entities: List[List[EntityBlock]] = []
        for p_text, g_text in zip(decoded_preds, decoded_labels):
            p_ents, _ = parse_sel(clean_text(p_text), tok=self._tokens)
            g_ents, _ = parse_sel(clean_text(g_text), tok=self._tokens)
            pred_entities.append(p_ents)
            gold_entities.append(g_ents)

        m: Dict[str, float] = {}

        if self._variant == "boundary_joint":
            m.update(compute_metrics_for_task(
                "boundary_joint",
                rel_schema=self._rel_schema,
                all_pred_triplets=[extract_triplets(p) for p in pred_entities],
                all_gold_triplets=[extract_triplets(g) for g in gold_entities],
            ))

        elif self._variant == "joint":
            m.update(compute_metrics_for_task(
                "joint",
                rel_schema=self._rel_schema,
                entity_schema=self._entity_schema,
                all_pred_triplets=[extract_triplets(p) for p in pred_entities],
                all_gold_triplets=[extract_triplets(g) for g in gold_entities],
                all_pred_quintuples=[_assemble_joint_quintuples(p) for p in pred_entities],
                all_gold_quintuples=[_assemble_joint_quintuples(g) for g in gold_entities],
                all_pred_entities=[[e["text"] for e in p] for p in pred_entities],
                all_gold_entities=[[e["text"] for e in g] for g in gold_entities],
                all_pred_entity_mentions=[
                    [(e["text"], e.get("type", "")) for e in p if e.get("type")]
                    for p in pred_entities
                ],
                all_gold_entity_mentions=[
                    [(e["text"], e.get("type", "")) for e in g if e.get("type")]
                    for g in gold_entities
                ],
            ))

        elif self._variant in {"re", "boundary_re"}:
            #   - "re"          → task="re"          (boundary + strict F1)
            #   - "boundary_re" → task="boundary_re" (boundary F1 only)
            if self._variant == "re":
                m.update(compute_metrics_for_task(
                    "re",
                    rel_schema=self._rel_schema,
                    entity_schema=self._entity_schema,
                    all_pred_triplets=[extract_triplets(p) for p in pred_entities],
                    all_gold_triplets=[extract_triplets(g) for g in gold_entities],
                    # _assemble_joint_quintuples reads entity types directly from
                    # the parsed SEL, which is correct for the "re" output format.
                    all_pred_quintuples=[_assemble_joint_quintuples(p) for p in pred_entities],
                    all_gold_quintuples=[_assemble_joint_quintuples(g) for g in gold_entities],
                ))
            else:  # boundary_re
                m.update(compute_metrics_for_task(
                    "boundary_re",
                    rel_schema=self._rel_schema,
                    all_pred_triplets=[extract_triplets(p) for p in pred_entities],
                    all_gold_triplets=[extract_triplets(g) for g in gold_entities],
                ))

        return m
