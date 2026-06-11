"""
S2G custom Seq2SeqTrainer for multi-task fine-tuning and pre-training.
"""
from __future__ import annotations

import contextlib
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput
from tqdm.auto import tqdm

from s2g.evaluation.metrics import compute_metrics_for_task
from s2g.linearisation import (
    AnyTokens, EntityBlock, JOINT_TOKENS, PIPELINE_TOKENS,
    build_boundary_encoder_input, build_joint_encoder_input,
    build_joint_plus_encoder_input, build_ner_encoder_input,
    build_re_encoder_input, extract_triplets, find_all_token_spans, parse_sel,
)

logger = logging.getLogger(__name__)

_PIPELINE_TASK_KEYS = ("boundary", "ner", "re")
_JOINT_TASK_KEYS = ("joint", "joint_plus")
_ALL_TASK_KEYS = _PIPELINE_TASK_KEYS + _JOINT_TASK_KEYS


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


def _assemble_joint_plus_quintuples(entities: List[EntityBlock]) -> List[Tuple[str, str, str, str, str]]:
    t_map = {e["text"]: e.get("type", "") for e in entities}
    return [
        (ent["text"], ent.get("type", ""), rel["type"], rel["tail"], t_map.get(rel["tail"], "")) 
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
        
        self._tasks = kwargs.pop("tasks", None)
        if self._tasks is None:
            self._tasks = ["ner", "re"] if self._variant == "pipeline" else ["joint", "joint+"]
        else:
            self._tasks = list(self._tasks)

        super().__init__(**kwargs)
        self._max_src = self._eval_cfg["max_source_length"]
        self._max_tgt = self._eval_cfg["max_target_length"]
        self._eval_bs = self._eval_cfg["eval_batch_size"]
        self._eval_beams = self._eval_cfg["eval_beams"]
        
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

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
        active_keys = [k for k in _ALL_TASK_KEYS if f"{k}_input_ids" in inputs]
        if not active_keys:
            raise ValueError(f"compute_loss: no task keys found. Expected from: {_ALL_TASK_KEYS}.")

        total_loss, last_outputs = None, None
        
        for k in active_keys:
            outputs = model(
                input_ids=inputs[f"{k}_input_ids"], 
                attention_mask=inputs[f"{k}_attention_mask"], 
                labels=inputs[f"{k}_labels"]
            )
            
            if total_loss is None:
                total_loss = outputs.loss
            else:
                total_loss = total_loss + outputs.loss
                
            last_outputs = outputs
            # EFFICIENCY FIX: Explicitly delete outputs to free VRAM graph nodes before next task pass
            del outputs 

        return (total_loss, last_outputs) if return_outputs else total_loss

    def evaluate(self, eval_dataset: Any = None, ignore_keys: Any = None, metric_key_prefix: str = "eval", **gen_kwargs: Any) -> Dict[str, float]:
        val_dataset = eval_dataset or self.eval_dataset
        if not val_dataset:
            return {}

        self.model.eval()
        
        all_metrics = self._evaluate_dataset(val_dataset, prefix=metric_key_prefix)
        
        if self._train_eval_dataset and metric_key_prefix == "eval":
            all_metrics.update(self._evaluate_dataset(self._train_eval_dataset, prefix="train_eval"))

        self.model.train()
        
        if self.is_world_process_zero():
            self.log(all_metrics)

        if dist.is_initialized():
            dist.barrier()
            
        return all_metrics
        
    def predict(self, test_dataset: Any, ignore_keys: Any = None, metric_key_prefix: str = "test", **gen_kwargs: Any) -> PredictionOutput:
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

        if self._variant == "pipeline":
            local_data = self._get_pipeline_predictions(local_instances, prefix=prefix)
        else:
            local_data = self._get_joint_predictions(local_instances, prefix=prefix)

        if world_size > 1:
            gathered_data = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_data, local_data)
            
            # EFFICIENCY FIX: Fast list comprehension flattens data in C, removing slow .extend() loops
            all_data = {
                k: [item for d in gathered_data if d is not None for item in d[k]]
                for k in local_data.keys()
            }
        else:
            all_data = local_data

        if rank == 0:
            m = {}
            if self._variant == "pipeline":
                b_per_inst, n_per_inst, r_per_inst = all_data["b_per_inst"], all_data["n_per_inst"], all_data["r_per_inst"]
                g_ents, g_mentions = all_data["g_ents"], all_data["g_mentions"]
                g_trips, g_quints = all_data["g_trips"], all_data["g_quints"]
                ner_maps = all_data["ner_maps"]

                if "boundary" in self._tasks:
                    m.update({f"boundary_{k}": v for k, v in compute_metrics_for_task(
                        "boundary", 
                        all_pred_entities=[[e["text"] for e in b] for b in b_per_inst], 
                        all_gold_entities=g_ents
                    ).items()})
                
                if "ner" in self._tasks:
                    m.update({f"ner_{k}": v for k, v in compute_metrics_for_task(
                        "ner", 
                        all_pred_entities=[[e["text"] for e in n] for n in n_per_inst], 
                        all_gold_entities=g_ents, 
                        all_pred_entity_mentions=[[(e["text"], e.get("type", "")) for e in n if e.get("type")] for n in n_per_inst], 
                        all_gold_entity_mentions=g_mentions
                    ).items()})
                
                if "re" in self._tasks:
                    m.update({f"re_{k}": v for k, v in compute_metrics_for_task(
                        "re", 
                        all_pred_triplets=[extract_triplets(r) for r in r_per_inst], 
                        all_gold_triplets=g_trips, 
                        all_pred_quintuples=[_assemble_re_quintuples(r, nm) for r, nm in zip(r_per_inst, ner_maps)], 
                        all_gold_quintuples=g_quints
                    ).items()})
            else:
                j_per_inst, jp_per_inst = all_data["j_per_inst"], all_data["jp_per_inst"]
                g_ents, g_mentions = all_data["g_ents"], all_data["g_mentions"]
                g_trips, g_quints = all_data["g_trips"], all_data["g_quints"]

                if "joint" in self._tasks:
                    m.update({f"joint_{k}": v for k, v in compute_metrics_for_task(
                        "joint", 
                        all_pred_triplets=[extract_triplets(j) for j in j_per_inst], 
                        all_gold_triplets=g_trips
                    ).items()})
                
                if "joint+" in self._tasks:
                    m.update({f"joint_plus_{k}": v for k, v in compute_metrics_for_task(
                        "joint+", 
                        all_pred_triplets=[extract_triplets(jp) for jp in jp_per_inst], 
                        all_gold_triplets=g_trips, 
                        all_pred_quintuples=[_assemble_joint_plus_quintuples(jp) for jp in jp_per_inst], 
                        all_gold_quintuples=g_quints, 
                        all_pred_entities=[[e["text"] for e in jp] for jp in jp_per_inst], 
                        all_gold_entities=g_ents, 
                        all_pred_entity_mentions=[[(e["text"], e.get("type", "")) for e in jp if e.get("type")] for jp in jp_per_inst], 
                        all_gold_entity_mentions=g_mentions
                    ).items()})
            
            return {f"{prefix}_{k}": v for k, v in m.items()}
            
        return {}

    def _get_pipeline_predictions(self, instances: List[Dict], prefix: str) -> Dict[str, List]:
        use_boundary = "boundary" in self._tasks
        use_ner = "ner" in self._tasks
        use_re = "re" in self._tasks

        b_per_inst = []
        if use_boundary:
            b_inputs = [build_boundary_encoder_input(inst["text"], tok=self._tokens) for inst in instances]
            b_per_inst = self._run_generation(b_inputs, desc=f"({prefix}) Boundary")
        else:
            b_per_inst = [[] for _ in instances]

        n_per_inst = []
        if use_ner:
            if use_boundary:
                n_inputs = [
                    build_ner_encoder_input(self._entity_schema, inst["tokens"], _to_spans(inst["tokens"], b), False, self._tokens) 
                    for inst, b in zip(instances, b_per_inst)
                ]
            else:
                n_inputs = [
                    build_ner_encoder_input(self._entity_schema, inst["tokens"], [], False, self._tokens) 
                    for inst in instances
                ]
            n_per_inst = self._run_generation(n_inputs, desc=f"({prefix}) NER")
        else:
            n_per_inst = [[] for _ in instances]

        r_per_inst = []
        ner_maps = []
        if use_re:
            r_inputs = []
            for inst, b, n in zip(instances, b_per_inst, n_per_inst):
                if use_ner:
                    r_inputs.append(build_re_encoder_input(self._rel_schema, inst["tokens"], _to_entity_data(inst["tokens"], n, use_type=True), False, self._tokens))
                    ner_maps.append({e["text"]: e.get("type", "") for e in n})
                else:
                    r_inputs.append(build_re_encoder_input(self._rel_schema, inst["tokens"], _to_entity_data(inst["tokens"], b, use_type=False), False, self._tokens))
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
            "g_quints": [[(r["head"]["text"], r["head"].get("type", "") if use_ner else "", r["type"], r["tail"]["text"], r["tail"].get("type", "") if use_ner else "") for r in inst["relations"]] for inst in instances]
        }

    def _get_joint_predictions(self, instances: List[Dict], prefix: str) -> Dict[str, List]:
        use_joint = "joint" in self._tasks
        use_joint_plus = "joint+" in self._tasks

        j_per_inst = []
        if use_joint:
            j_inputs = [build_joint_encoder_input(self._rel_schema, inst["text"], False, self._tokens) for inst in instances]
            j_per_inst = self._run_generation(j_inputs, desc=f"({prefix}) Joint")
        else:
            j_per_inst = [[] for _ in instances]

        jp_per_inst = []
        if use_joint_plus:
            jp_inputs = [build_joint_plus_encoder_input(self._entity_schema, self._rel_schema, inst["text"], False, self._tokens) for inst in instances]
            jp_per_inst = self._run_generation(jp_inputs, desc=f"({prefix}) Joint+")
        else:
            jp_per_inst = [[] for _ in instances]

        return {
            "j_per_inst": j_per_inst,
            "jp_per_inst": jp_per_inst,
            "g_ents": [[e["text"] for e in inst["entities"]] for inst in instances],
            "g_mentions": [[(e["text"], e.get("type", "")) for e in inst["entities"]] for inst in instances],
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