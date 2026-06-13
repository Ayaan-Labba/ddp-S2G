"""
S2G Data Collator for multi-task Pipeline and BoundaryJoint model fine-tuning.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from transformers import PreTrainedTokenizerBase

from s2g.linearisation import (
    S2GTokens, AnyTokens, VARIANT_TO_TASKS,
    build_boundary_encoder_input, build_boundary_joint_encoder_input,
    build_joint_encoder_input, build_ner_encoder_input,
    build_re_encoder_input, build_sel, organize_by_entity,
    filter_entity_blocks
)


class S2GCollator:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, entity_schema: List[str], 
        rel_schema: List[str], config: Dict[str, Any]
    ) -> None:
        self._variant = config.get("model_variant")
        self._mode = config.get("mode", "budget")
        if self._variant not in VARIANT_TO_TASKS or self._mode not in {"budget", "bernoulli"}:
            raise ValueError(f"Invalid model_variant '{self._variant}' or mode '{self._mode}'.")

        self._tokenizer = tokenizer
        self._entity_schema = list(entity_schema)
        self._entity_schema_set = set(entity_schema)
        self._rel_schema = list(rel_schema)
        self._rel_schema_set = set(rel_schema)
        self._cfg = config
        self._random_prompt = config.get("random_prompt", False)
        self._random_sel = config.get("random_sel", False)
        self._use_rejection = config.get("use_rejection", False)
        self._tok: AnyTokens = S2GTokens(self._variant, use_rejection=self._use_rejection)
        self._step = 0

        self._tasks = VARIANT_TO_TASKS[self._variant]

        task_to_key = {
            "boundary": "boundary",
            "ner": "ner",
            "re": "re",
            "boundary_re": "boundary_re",
            "boundary_joint": "boundary_joint",
            "joint": "joint"
        }
        self._task_keys = [task_to_key.get(t, t) for t in self._tasks]

    @property
    def current_step(self) -> int: 
        return self._step
        
    @current_step.setter
    def current_step(self, value: int) -> None: 
        self._step = value

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        tasks = {tk: ([], []) for tk in self._task_keys}
        for inst in batch:
            blocks = organize_by_entity(inst["entities"], inst["relations"])
            for tk in self._task_keys:
                func = getattr(self, f"_prepare_{tk}")
                enc, dec = func(inst, blocks)
                tasks[tk][0].append(enc)
                tasks[tk][1].append(dec)
                
        return {k: v for tk, (enc, dec) in tasks.items() for k, v in self._tokenize_task(tk, enc, dec).items()}

    def _prepare_boundary(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        return (
            build_boundary_encoder_input(inst["text"], tok=self._tok), 
            build_sel(blocks, "boundary", self._tok, random_sel=self._random_sel)
        )

    def _prepare_ner(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        pos_ent, neg_ent = self._sample_types(
            inst["entity_types"], self._entity_schema, self._cfg.get("max_ent_types_in_prompt")
        )
        enc = build_ner_encoder_input(
            pos_ent + neg_ent, inst["tokens"], [], random_order=self._random_prompt, tok=self._tok
        )
        allowed_ents = set(pos_ent)
        filtered_blocks = [b for b in blocks if b.get("type") in allowed_ents]
        
        return enc, build_sel(filtered_blocks, "ner", self._tok, rejected_ent_types=neg_ent, random_sel=self._random_sel, use_rejection=self._use_rejection)

    def _prepare_re(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types_in_prompt")
        )
        data = [(int(e["offset"][0]), int(e["offset"][1]), e["type"]) for e in inst["entities"]]
        enc = build_re_encoder_input(
            pos_rel + neg_rel, inst["tokens"], data, random_order=self._random_prompt, tok=self._tok
        )
        filtered_blocks = filter_entity_blocks(blocks, set(pos_rel))
        
        return enc, build_sel(filtered_blocks, "re", self._tok, rejected_rel_types=neg_rel, random_sel=self._random_sel, use_rejection=self._use_rejection)

    def _prepare_boundary_re(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types_in_prompt")
        )
        data = [(int(e["offset"][0]), int(e["offset"][1]), "") for e in inst["entities"]]
        enc = build_re_encoder_input(
            pos_rel + neg_rel, inst["tokens"], data, random_order=self._random_prompt, tok=self._tok
        )
        filtered_blocks = filter_entity_blocks(blocks, set(pos_rel))
        
        return enc, build_sel(filtered_blocks, "boundary_re", self._tok, rejected_rel_types=neg_rel, random_sel=self._random_sel, use_rejection=self._use_rejection)

    def _prepare_boundary_joint(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types_in_prompt")
        )
        enc = build_boundary_joint_encoder_input(
            pos_rel + neg_rel, inst["text"], random_order=self._random_prompt, tok=self._tok
        )
        filtered_blocks = filter_entity_blocks(blocks, set(pos_rel))
        
        return enc, build_sel(filtered_blocks, "boundary_joint", self._tok, rejected_rel_types=neg_rel, random_sel=self._random_sel, use_rejection=self._use_rejection)

    def _prepare_joint(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        pos_ent, neg_ent = self._sample_types(
            inst["entity_types"], self._entity_schema, self._cfg.get("max_ent_types_in_prompt")
        )
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types_in_prompt")
        )
        enc = build_joint_encoder_input(
            pos_ent + neg_ent, pos_rel + neg_rel, inst["text"], random_order=self._random_prompt, tok=self._tok
        )
        allowed_ents = set(pos_ent)
        filtered_blocks = filter_entity_blocks(
            [b for b in blocks if b.get("type") in allowed_ents],
            set(pos_rel)
        )
        
        return enc, build_sel(
            filtered_blocks, "joint", self._tok, rejected_ent_types=neg_ent, 
            rejected_rel_types=neg_rel, random_sel=self._random_sel, use_rejection=self._use_rejection
        )

    def _sample_types(
            self, instance_types: List[str], schema: List[str], max_types: Optional[int]
        ) -> Tuple[List[str], List[str]]:
        inst_set = set(instance_types)
        
        if self._mode == "budget":
            neg_pool = [t for t in schema if t not in inst_set]
            sampled_neg = random.sample(
                neg_pool, min(max(0, max_types - len(instance_types)), len(neg_pool))
            ) if max_types is not None else neg_pool
            return list(instance_types), sampled_neg

        pos_rate, neg_rate, k = self._schedule_values()
        included_pos = [t for t in instance_types if random.random() < pos_rate]
        candidate_neg = [t for t in schema if t not in inst_set and random.random() < neg_rate]
        
        if len(candidate_neg) > k: 
            candidate_neg = random.sample(candidate_neg, k)
        if max_types is not None and len(candidate_neg) > (rem := max(0, max_types - len(included_pos))):
            candidate_neg = random.sample(candidate_neg, rem)
            
        return included_pos, candidate_neg

    def _schedule_values(self) -> Tuple[float, float, int]:
        T = max(int(self._cfg.get("max_steps", 1)), 1)
        frac = min(self._step, T) / T
        
        def lerp(start: float, end: float) -> float:
            return start + frac * (end - start)
            
        return (
            lerp(self._cfg.get("positive_rate_start", 0.9), self._cfg.get("positive_rate_end", 0.9)),
            lerp(self._cfg.get("negative_rate_start", 0.1), self._cfg.get("negative_rate_end", 0.1)),
            round(lerp(float(self._cfg.get("negative_max_start", 1)), float(self._cfg.get("negative_max_end", 20))))
        )

    def _tokenize_task(self, task_key: str, encoder_inputs: List[str], decoder_targets: List[str]) -> Dict[str, Any]:
        model_inputs = self._tokenizer(
            encoder_inputs, max_length=self._cfg["max_source_length"], 
            truncation=True, padding="longest", return_tensors="pt"
        )
        label_enc = self._tokenizer(
            decoder_targets, max_length=self._cfg["max_target_length"], 
            truncation=True, padding="longest", return_tensors="pt"
        )
        
        label_ids = label_enc["input_ids"].clone()
        label_ids.masked_fill_(label_ids == self._tokenizer.pad_token_id, -100)

        return {
            f"{task_key}_input_ids": model_inputs["input_ids"],
            f"{task_key}_attention_mask": model_inputs["attention_mask"],
            f"{task_key}_labels": label_ids,
        }