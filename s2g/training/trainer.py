"""
S2G custom Seq2SeqTrainer for multi-task fine-tuning and pre-training.
"""
from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from transformers import EarlyStoppingCallback, Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput
from tqdm.auto import tqdm

from s2g.evaluation.metrics import compute_metrics_for_task
from s2g.linearisation import (
    S2GTokens, AnyTokens, EntityBlock, VARIANT_TO_TASKS,
    build_boundary_encoder_input, build_boundary_joint_encoder_input,
    build_joint_encoder_input, build_ner_encoder_input,
    build_re_encoder_input, extract_triplets, find_all_token_spans, parse_sel,
)

logger = logging.getLogger(__name__)

_PIPELINE_TASK_KEYS = ("boundary", "ner", "re", "boundary_re")
_BOUNDARY_JOINT_TASK_KEYS = ("boundary_joint", "joint")
_ALL_TASK_KEYS = _PIPELINE_TASK_KEYS + _BOUNDARY_JOINT_TASK_KEYS


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _to_spans(source_tokens: List[str], entities: List[EntityBlock]) -> List[Tuple[int, int]]:
    return list(dict.fromkeys(span for ent in entities for span in find_all_token_spans(source_tokens, ent["text"])))


def _to_entity_data(source_tokens: List[str], entities: List[EntityBlock], use_type: bool = True) -> List[Tuple[int, int, str]]:
    return list(dict.fromkeys(
        (*span, ent["type"] if (use_type and ent.get("type")) else "") 
        for ent in entities if (not use_type or ent.get("type")) 
        for span in find_all_token_spans(source_tokens, ent["text"])
    ))


def _assemble_re_quintuples(re_entities: List[EntityBlock], ner_map: Dict[str, str]) -> List[Tuple[str, str, str, str, str]]:
    return [
        (ent["text"], ner_map.get(ent["text"], ""), rel["type"], rel["tail"], ner_map.get(rel["tail"], "")) 
        for ent in re_entities for rel in ent["relations"]
    ]


def _assemble_joint_quintuples(entities: List[EntityBlock]) -> List[Tuple[str, str, str, str, str]]:
    t_map = {e["text"]: e.get("type", "") for e in entities}
    return [
        (ent["text"], ent.get("type", ""), rel["type"], rel["tail"], rel.get("tail_type") or t_map.get(rel["tail"], "")) 
        for ent in entities for rel in ent["relations"]
    ]


class S2GTrainer(Seq2SeqTrainer):
    def __init__(self, **kwargs: Any) -> None:
        self._variant = kwargs.pop("model_variant")
        self._tokens = kwargs.pop("tokens")
        self._entity_schema = kwargs.pop("entity_schema", [])
        self._rel_schema = kwargs.pop("rel_schema", [])
        self._eval_cfg = kwargs.pop("eval_cfg")
        self._train_eval_dataset = kwargs.pop("train_eval_dataset", None)
        self._scheduler_type = kwargs.pop("scheduler_type", "inverse_sqrt")
        
        kwargs.pop("tasks", None)
        self._tasks = VARIANT_TO_TASKS[self._variant]

        compute_metrics = self._compute_metrics_hf if self._variant not in {"pipeline", "boundary_pipeline"} else None
        super().__init__(compute_metrics=compute_metrics, **kwargs)
        self._max_src = self._eval_cfg["max_source_length"]
        self._max_tgt = self._eval_cfg["max_target_length"]
        self._eval_bs = self._eval_cfg["eval_batch_size"]
        self._eval_beams = self._eval_cfg["eval_beams"]
        self._ssi_prompt = self._eval_cfg.get("ssi_prompt", "ssi")
        
        self._specials_to_remove = [
            tok for tok in (self.processing_class.pad_token, self.processing_class.eos_token, self.processing_class.bos_token) if tok
        ]

    def _clean_decoded(self, text: str) -> str:
        for tok in self._specials_to_remove:
            text = text.replace(tok, "")
        return " ".join(text.split())

    def create_scheduler(self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        if self.lr_scheduler is not None: 
            return

        if self._scheduler_type == "inverse_sqrt":
            opt = optimizer or self.optimizer
            warmup = self.args.get_warmup_steps(num_training_steps)
            self.lr_scheduler = LambdaLR(
                opt, 
                lambda step: max(step, 1) / max(warmup, 1) if max(step, 1) < warmup else math.sqrt(warmup / max(step, 1))
            )
        else:
            super().create_scheduler(num_training_steps, optimizer)

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        active_keys = [k for k in _ALL_TASK_KEYS if f"{k}_input_ids" in inputs]
        if not active_keys:
            raise ValueError(f"training_step: no task keys found. Expected from: {_ALL_TASK_KEYS}.")

        total_loss = 0.0
        
        for k in active_keys:
            task_inputs = {
                "input_ids": inputs[f"{k}_input_ids"],
                "attention_mask": inputs[f"{k}_attention_mask"],
                "labels": inputs[f"{k}_labels"]
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

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        outputs = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            labels=inputs["labels"]
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(
        self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
        prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None, **kwargs
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        active_keys = [k for k in _ALL_TASK_KEYS if f"{k}_input_ids" in inputs]
        if not active_keys:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **kwargs)
            
        k = active_keys[0]
        standard_inputs = {
            "input_ids": inputs[f"{k}_input_ids"],
            "attention_mask": inputs[f"{k}_attention_mask"],
            "labels": inputs.get(f"{k}_labels"),
        }
        
        return super().prediction_step(model, standard_inputs, prediction_loss_only, ignore_keys, **kwargs)

    def evaluate(self, eval_dataset: Any = None, ignore_keys: Any = None, metric_key_prefix: str = "eval", **gen_kwargs: Any) -> Dict[str, float]:
        if self._variant not in {"pipeline", "boundary_pipeline"}:
            self.args.predict_with_generate = True
            self.args.generation_max_length = self._max_tgt
            self.args.generation_num_beams = self._eval_beams
            
            all_metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, **gen_kwargs)
            if self._train_eval_dataset and metric_key_prefix == "eval":
                # Temporarily remove EarlyStoppingCallback to prevent it from disabling early stopping
                # because the metric prefix for train eval is "train" instead of "eval"
                early_stopping_callbacks = [
                    cb for cb in self.callback_handler.callbacks if isinstance(cb, EarlyStoppingCallback)
                ]
                for cb in early_stopping_callbacks:
                    self.callback_handler.callbacks.remove(cb)
                
                try:
                    train_metrics = super().evaluate(eval_dataset=self._train_eval_dataset, ignore_keys=ignore_keys, metric_key_prefix="train", **gen_kwargs)
                finally:
                    # Restore the callbacks
                    for cb in early_stopping_callbacks:
                        self.callback_handler.callbacks.append(cb)
                all_metrics.update(train_metrics)
            return all_metrics

        val_dataset = eval_dataset or self.eval_dataset
        if not val_dataset:
            return {}

        self.model.eval()
        
        all_metrics = self._evaluate_dataset(val_dataset, prefix=metric_key_prefix)
        
        if self._train_eval_dataset and metric_key_prefix == "eval":
            all_metrics.update(self._evaluate_dataset(self._train_eval_dataset, prefix="train"))

        self.model.train()
        
        if self.is_world_process_zero():
            self.log(all_metrics)

        if dist.is_initialized():
            dist.barrier()
            
        if metric_key_prefix == "eval":
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, all_metrics)
            
        return all_metrics
        
    def predict(self, test_dataset: Any, ignore_keys: Any = None, metric_key_prefix: str = "test", **gen_kwargs: Any) -> PredictionOutput:
        if self._variant not in {"pipeline", "boundary_pipeline"}:
            self.args.predict_with_generate = True
            self.args.generation_max_length = self._max_tgt
            self.args.generation_num_beams = self._eval_beams
            return super().predict(test_dataset=test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, **gen_kwargs)

        metrics = self.evaluate(
            eval_dataset=test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix, **gen_kwargs
        )
        return PredictionOutput(predictions=None, label_ids=None, metrics=metrics)

    def _evaluate_dataset(self, dataset: Any, prefix: str) -> Dict[str, float]:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        total_instances = len(dataset)
        chunk_size = math.ceil(total_instances / world_size)
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, total_instances)
        
        local_instances = [dataset[i] for i in range(start_idx, end_idx)]

        if self._variant in {"pipeline", "boundary_pipeline", "boundary", "ner", "re", "boundary_re"}:
            local_data = self._get_pipeline_predictions(local_instances, prefix=prefix)
        else:
            local_data = self._get_boundary_joint_predictions(local_instances, prefix=prefix)

        if world_size > 1:
            gathered_data = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_data, local_data)
            all_data = {
                k: [item for d in gathered_data if d is not None for item in d[k]]
                for k in local_data.keys()
            }
        else:
            all_data = local_data

        if rank == 0:
            m = {}
            if self._variant in {"pipeline", "boundary_pipeline", "boundary", "ner", "re", "boundary_re"}:
                b_per_inst, n_per_inst, r_per_inst = all_data["b_per_inst"], all_data["n_per_inst"], all_data["r_per_inst"]
                g_ents, g_mentions = all_data["g_ents"], all_data["g_mentions"]
                g_trips, g_quints = all_data["g_trips"], all_data["g_quints"]
                ner_maps = all_data["ner_maps"]

                if "boundary" in self._tasks:
                    m.update(compute_metrics_for_task(
                        "boundary", 
                        all_pred_entities=[[e["text"] for e in b] for b in b_per_inst], 
                        all_gold_entities=g_ents
                    ))
                
                if "ner" in self._tasks:
                    m.update(compute_metrics_for_task(
                        "ner", 
                        all_pred_entities=[[e["text"] for e in n] for n in n_per_inst], 
                        all_gold_entities=g_ents, 
                        all_pred_entity_mentions=[[(e["text"], e.get("type", "")) for e in n if e.get("type")] for n in n_per_inst], 
                        all_gold_entity_mentions=g_mentions
                    ))
                
                if "re" in self._tasks:
                    m.update(compute_metrics_for_task(
                        "re", 
                        all_pred_triplets=[extract_triplets(r) for r in r_per_inst], 
                        all_gold_triplets=g_trips, 
                        all_pred_quintuples=[_assemble_re_quintuples(r, nm) for r, nm in zip(r_per_inst, ner_maps)], 
                        all_gold_quintuples=g_quints
                    ))

                if "boundary_re" in self._tasks:
                    m.update(compute_metrics_for_task(
                        "boundary_re", 
                        all_pred_triplets=[extract_triplets(r) for r in r_per_inst], 
                        all_gold_triplets=g_trips, 
                    ))
            else:
                j_per_inst, jp_per_inst = all_data["j_per_inst"], all_data["jp_per_inst"]
                g_ents, g_mentions = all_data["g_ents"], all_data["g_mentions"]
                g_trips, g_quints = all_data["g_trips"], all_data["g_quints"]

                if "boundary_joint" in self._tasks:
                    m.update(compute_metrics_for_task(
                        "boundary_joint", 
                        all_pred_triplets=[extract_triplets(j) for j in j_per_inst], 
                        all_gold_triplets=g_trips
                    ))
                
                if "joint" in self._tasks:
                    m.update(compute_metrics_for_task(
                        "joint", 
                        all_pred_triplets=[extract_triplets(jp) for jp in jp_per_inst], 
                        all_gold_triplets=g_trips, 
                        all_pred_quintuples=[_assemble_joint_quintuples(jp) for jp in jp_per_inst], 
                        all_gold_quintuples=g_quints, 
                        all_pred_entities=[[e["text"] for e in jp] for jp in jp_per_inst], 
                        all_gold_entities=g_ents, 
                        all_pred_entity_mentions=[[(e["text"], e.get("type", "")) for e in jp if e.get("type")] for jp in jp_per_inst], 
                        all_gold_entity_mentions=g_mentions
                    ))
            
            return {f"{prefix}_{k}": v for k, v in m.items()}
            
        return {}

    def _get_pipeline_predictions(self, instances: List[Dict], prefix: str) -> Dict[str, List]:
        use_boundary = "boundary" in self._tasks
        use_ner = "ner" in self._tasks
        use_re = "re" in self._tasks
        use_boundary_re = "boundary_re" in self._tasks

        b_per_inst = []
        if use_boundary:
            b_inputs = [build_boundary_encoder_input(inst["text"], tok=self._tokens, ssi_prompt=self._ssi_prompt) for inst in instances]
            b_per_inst = self._run_generation(b_inputs, desc=f"({prefix}) Boundary")
        else:
            b_per_inst = [[] for _ in instances]

        n_per_inst = []
        if use_ner:
            if use_boundary:
                n_inputs = [
                    build_ner_encoder_input(self._entity_schema, inst["tokens"], _to_spans(inst["tokens"], b), False, self._tokens, ssi_prompt=self._ssi_prompt) 
                    for inst, b in zip(instances, b_per_inst)
                ]
            else:
                n_inputs = [
                    build_ner_encoder_input(self._entity_schema, inst["tokens"], [], False, self._tokens, ssi_prompt=self._ssi_prompt) 
                    for inst in instances
                ]
            n_per_inst = self._run_generation(n_inputs, desc=f"({prefix}) NER")
        else:
            n_per_inst = [[] for _ in instances]

        r_per_inst = []
        ner_maps = []
        if use_re or use_boundary_re:
            r_inputs = []
            if not use_boundary and not use_ner:
                for inst in instances:
                    if use_re:
                        entity_data = [(int(e["offset"][0]), int(e["offset"][1]), e.get("type", "")) for e in inst["entities"]]
                        ner_maps.append({e["text"]: e.get("type", "") for e in inst["entities"]})
                    else:
                        entity_data = [(int(e["offset"][0]), int(e["offset"][1]), "") for e in inst["entities"]]
                        ner_maps.append({e["text"]: "" for e in inst["entities"]})
                    r_inputs.append(build_re_encoder_input(self._rel_schema, inst["tokens"], entity_data, False, self._tokens, ssi_prompt=self._ssi_prompt))
            else:
                for inst, b, n in zip(instances, b_per_inst, n_per_inst):
                    if use_ner:
                        r_inputs.append(build_re_encoder_input(self._rel_schema, inst["tokens"], _to_entity_data(inst["tokens"], n, use_type=True), False, self._tokens, ssi_prompt=self._ssi_prompt))
                        ner_maps.append({e["text"]: e.get("type", "") for e in n})
                    else:
                        r_inputs.append(build_re_encoder_input(self._rel_schema, inst["tokens"], _to_entity_data(inst["tokens"], b, use_type=False), False, self._tokens, ssi_prompt=self._ssi_prompt))
                        ner_maps.append({e["text"]: "" for e in b})
            r_per_inst = self._run_generation(r_inputs, desc=f"({prefix}) RE")
        else:
            r_per_inst = [[] for _ in instances]
            ner_maps = [{} for _ in instances]

        return {
            "b_per_inst": b_per_inst,
            "n_per_inst": n_per_inst,
            "r_per_inst": r_per_inst,
            "ner_maps": ner_maps,
            "g_ents": [[e["text"] for e in inst["entities"]] for inst in instances],
            "g_mentions": [[(e["text"], e.get("type", "")) for e in inst["entities"]] for inst in instances],
            "g_trips": [[(r["head"]["text"], r["type"], r["tail"]["text"]) for r in inst["relations"]] for inst in instances],
            "g_quints": [[(r["head"]["text"], r["head"].get("type", "") if (use_ner or use_re) else "", r["type"], r["tail"]["text"], r["tail"].get("type", "") if (use_ner or use_re) else "") for r in inst["relations"]] for inst in instances]
        }

    def _get_boundary_joint_predictions(self, instances: List[Dict], prefix: str) -> Dict[str, List]:
        use_boundary_joint = "boundary_joint" in self._tasks
        use_joint = "joint" in self._tasks

        j_per_inst = []
        if use_boundary_joint:
            j_inputs = [build_boundary_joint_encoder_input(self._rel_schema, inst["text"], False, self._tokens, ssi_prompt=self._ssi_prompt) for inst in instances]
            j_per_inst = self._run_generation(j_inputs, desc=f"({prefix}) BoundaryJoint")
        else:
            j_per_inst = [[] for _ in instances]

        jp_per_inst = []
        if use_joint:
            jp_inputs = [build_joint_encoder_input(self._entity_schema, self._rel_schema, inst["text"], False, self._tokens, ssi_prompt=self._ssi_prompt) for inst in instances]
            jp_per_inst = self._run_generation(jp_inputs, desc=f"({prefix}) BoundaryJoint+")
        else:
            jp_per_inst = [[] for _ in instances]

        gold_heads_per_inst = [{r["head"]["text"] for r in inst["relations"]} for inst in instances]
        return {
            "j_per_inst": j_per_inst,
            "jp_per_inst": jp_per_inst,
            "g_ents": [[e["text"] for e in inst["entities"] if e["text"] in gold_heads_per_inst[i]] for i, inst in enumerate(instances)],
            "g_mentions": [[(e["text"], e.get("type", "")) for e in inst["entities"] if e["text"] in gold_heads_per_inst[i]] for i, inst in enumerate(instances)],
            "g_trips": [[(r["head"]["text"], r["type"], r["tail"]["text"]) for r in inst["relations"]] for inst in instances],
            "g_quints": [[(r["head"]["text"], r["head"].get("type", ""), r["type"], r["tail"]["text"], r["tail"].get("type", "")) for r in inst["relations"]] for inst in instances]
        }

    def _autocast_ctx(self) -> contextlib.AbstractContextManager:
        if self.args.device.type == "cuda":
            if self.args.bf16: 
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.args.fp16: 
                return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _run_generation(self, encoder_inputs: List[str], desc: str = "Generating") -> List[List[EntityBlock]]:
        all_entities, raw_model = [], _unwrap_model(self.model)

        gen_kwargs = {
            "num_beams": self._eval_beams,
            "max_length": self._max_tgt,
        }
        if self._eval_beams > 1:
            gen_kwargs["length_penalty"] = 0.0
            gen_kwargs["no_repeat_ngram_size"] = 0
            gen_kwargs["early_stopping"] = False

        # tqdm wraps the execution range safely across distributed steps
        for i in tqdm(
            range(0, len(encoder_inputs), self._eval_bs),
            desc=desc,
            disable=not self.is_world_process_zero(),
            leave=False
        ):
            tok_out = self.processing_class(
                encoder_inputs[i:i + self._eval_bs], max_length=self._max_src, 
                truncation=True, padding="longest", return_tensors="pt"
            ).to(self.args.device, non_blocking=True)

            with torch.inference_mode(), self._autocast_ctx():
                generated = raw_model.generate(**tok_out, **gen_kwargs)

            for text in self.processing_class.batch_decode(generated, skip_special_tokens=False):
                entities, _ = parse_sel(self._clean_decoded(text), tok=self._tokens)
                all_entities.append(entities)

        return all_entities

    def _compute_metrics_hf(self, eval_preds: Any) -> Dict[str, float]:
        import numpy as np
        preds, label_ids = eval_preds.predictions, eval_preds.label_ids
        
        if isinstance(preds, tuple):
            preds = preds[0]
            
        tokenizer = self.processing_class
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)

        labels = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

        specials = [tok for tok in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token) if tok]

        def clean_text(text):
            for tok in specials:
                text = text.replace(tok, "")
            return " ".join(text.split())

        pred_entities = []
        gold_entities = []
        for p_text, g_text in zip(decoded_preds, decoded_labels):
            p_ents, _ = parse_sel(clean_text(p_text), tok=self._tokens)
            g_ents, _ = parse_sel(clean_text(g_text), tok=self._tokens)
            pred_entities.append(p_ents)
            gold_entities.append(g_ents)

        m = {}
        if self._variant == "boundary_joint":
            m.update(compute_metrics_for_task(
                "boundary_joint",
                all_pred_triplets=[extract_triplets(p) for p in pred_entities],
                all_gold_triplets=[extract_triplets(g) for g in gold_entities]
            ))
        elif self._variant == "joint":
            m.update(compute_metrics_for_task(
                "joint",
                all_pred_triplets=[extract_triplets(p) for p in pred_entities],
                all_gold_triplets=[extract_triplets(g) for g in gold_entities],
                all_pred_quintuples=[_assemble_joint_quintuples(p) for p in pred_entities],
                all_gold_quintuples=[_assemble_joint_quintuples(g) for g in gold_entities],
                all_pred_entities=[[e["text"] for e in p] for p in pred_entities],
                all_gold_entities=[[e["text"] for e in g] for g in gold_entities],
                all_pred_entity_mentions=[[(e["text"], e.get("type", "")) for e in p if e.get("type")] for p in pred_entities],
                all_gold_entity_mentions=[[(e["text"], e.get("type", "")) for e in g if e.get("type")] for g in gold_entities]
            ))
        elif self._variant == "boundary":
            m.update(compute_metrics_for_task(
                "boundary", 
                all_pred_entities=[[e["text"] for e in p] for p in pred_entities], 
                all_gold_entities=[[e["text"] for e in g] for g in gold_entities]
            ))
        elif self._variant == "ner":
            m.update(compute_metrics_for_task(
                "ner", 
                all_pred_entities=[[e["text"] for e in p] for p in pred_entities], 
                all_gold_entities=[[e["text"] for e in g] for g in gold_entities],
                all_pred_entity_mentions=[[(e["text"], e.get("type", "")) for e in p if e.get("type")] for p in pred_entities], 
                all_gold_entity_mentions=[[(e["text"], e.get("type", "")) for e in g if e.get("type")] for g in gold_entities]
            ))
        elif self._variant in {"re", "boundary_re"}:
            m.update(compute_metrics_for_task(
                "boundary_re", 
                all_pred_triplets=[extract_triplets(p) for p in pred_entities], 
                all_gold_triplets=[extract_triplets(g) for g in gold_entities]
            ))
            
        return m