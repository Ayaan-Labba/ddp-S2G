"""
S2G Data Collator for multi-task Pipeline and Joint model fine-tuning.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from transformers import PreTrainedTokenizerBase

from s2g.linearisation import (
    JOINT_TOKENS, PIPELINE_TOKENS, AnyTokens,
    build_boundary_encoder_input, build_joint_encoder_input,
    build_joint_plus_encoder_input, build_ner_encoder_input,
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
        if self._variant not in {"pipeline", "joint"} or self._mode not in {"budget", "bernoulli"}:
            raise ValueError(f"Invalid model_variant '{self._variant}' or mode '{self._mode}'.")

        self._tokenizer = tokenizer
        self._entity_schema = list(entity_schema)
        self._entity_schema_set = set(entity_schema)
        self._rel_schema = list(rel_schema)
        self._rel_schema_set = set(rel_schema)
        self._cfg = config
        self._tok: AnyTokens = PIPELINE_TOKENS if self._variant == "pipeline" else JOINT_TOKENS
        
        self._random_prompt = config.get("random_prompt", False)
        self._random_sel = config.get("random_sel", False)
        self._step = 0

    @property
    def current_step(self) -> int: 
        return self._step
        
    @current_step.setter
    def current_step(self, value: int) -> None: 
        self._step = value

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        return self._collate_pipeline(batch) if self._variant == "pipeline" else self._collate_joint(batch)

    def _collate_pipeline(self, batch: List[Dict]) -> Dict[str, Any]:
        tasks = {"boundary": ([], []), "ner": ([], []), "re": ([], [])}
        for inst in batch:
            blocks = organize_by_entity(inst["entities"], inst["relations"])
            for task, func in [("boundary", self._prepare_boundary), ("ner", self._prepare_ner), ("re", self._prepare_re)]:
                enc, dec = func(inst, blocks)
                tasks[task][0].append(enc)
                tasks[task][1].append(dec)
                
        return {k: v for t, (enc, dec) in tasks.items() for k, v in self._tokenize_task(t, enc, dec).items()}

    def _collate_joint(self, batch: List[Dict]) -> Dict[str, Any]:
        tasks = {"joint": ([], []), "joint_plus": ([], [])}
        for inst in batch:
            blocks = organize_by_entity(inst["entities"], inst["relations"])
            for task, func in [("joint", self._prepare_joint), ("joint_plus", self._prepare_joint_plus)]:
                enc, dec = func(inst, blocks)
                tasks[task][0].append(enc)
                tasks[task][1].append(dec)
                
        return {k: v for t, (enc, dec) in tasks.items() for k, v in self._tokenize_task(t, enc, dec).items()}

    def _prepare_boundary(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        return (
            build_boundary_encoder_input(inst["text"], tok=self._tok), 
            build_sel(blocks, "boundary", self._tok, random_sel=self._random_sel)
        )

    def _prepare_ner(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        pos_ent, neg_ent = self._sample_types(
            inst["entity_types"], self._entity_schema, self._cfg.get("max_ent_types_in_prompt")
        )
        spans = [(int(e["offset"][0]), int(e["offset"][1])) for e in inst["entities"]]
        enc = build_ner_encoder_input(
            pos_ent + neg_ent, inst["tokens"], spans, random_order=self._random_prompt, tok=self._tok
        )
        allowed_ents = set(pos_ent)
        filtered_blocks = [b for b in blocks if b.get("type") in allowed_ents]
        
        return enc, build_sel(filtered_blocks, "ner", self._tok, rejected_ent_types=neg_ent, random_sel=self._random_sel)

    def _prepare_re(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types_in_prompt")
        )
        data = [(int(e["offset"][0]), int(e["offset"][1]), e["type"]) for e in inst["entities"]]
        enc = build_re_encoder_input(
            pos_rel + neg_rel, inst["tokens"], data, random_order=self._random_prompt, tok=self._tok
        )
        filtered_blocks = filter_entity_blocks(blocks, set(pos_rel))
        
        return enc, build_sel(filtered_blocks, "re", self._tok, rejected_rel_types=neg_rel, random_sel=self._random_sel)

    def _prepare_joint(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types_in_prompt")
        )
        enc = build_joint_encoder_input(
            pos_rel + neg_rel, inst["text"], random_order=self._random_prompt, tok=self._tok
        )
        filtered_blocks = filter_entity_blocks(blocks, set(pos_rel))
        
        return enc, build_sel(filtered_blocks, "joint", self._tok, rejected_rel_types=neg_rel, random_sel=self._random_sel)

    def _prepare_joint_plus(self, inst: Dict, blocks: List) -> Tuple[str, str]:
        pos_ent, neg_ent = self._sample_types(
            inst["entity_types"], self._entity_schema, self._cfg.get("max_ent_types_in_prompt")
        )
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types_in_prompt")
        )
        enc = build_joint_plus_encoder_input(
            pos_ent + neg_ent, pos_rel + neg_rel, inst["text"], random_order=self._random_prompt, tok=self._tok
        )
        allowed_ents = set(pos_ent)
        filtered_blocks = filter_entity_blocks(
            [b for b in blocks if b.get("type") in allowed_ents],
            set(pos_rel)
        )
        
        return enc, build_sel(
            filtered_blocks, "joint+", self._tok, rejected_ent_types=neg_ent, 
            rejected_rel_types=neg_rel, random_sel=self._random_sel
        )

    def _sample_types(
            self, instance_types: List[str], schema: List[str], max_types: Optional[int]
        ) -> Tuple[List[str], List[str]]:
        
        # EFFICIENCY FIX: Evaluate set once, not O(N*M) times inside comprehension
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
        
        # EFFICIENCY FIX: Masked fill avoids allocating intermediate boolean tensor arrays
        label_ids.masked_fill_(label_ids == self._tokenizer.pad_token_id, -100)

        return {
            f"{task_key}_input_ids": model_inputs["input_ids"],
            f"{task_key}_attention_mask": model_inputs["attention_mask"],
            f"{task_key}_labels": label_ids,
        }