"""
S2G Data Collator.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from s2g.linearisation import (
    S2GTokens,
    build_graph,
    build_boundary_joint_encoder_input,
    build_joint_encoder_input,
    build_re_encoder_input, 
    build_boundary_re_encoder_input, 
    organise_filter_and_block, 
    get_tok
)


class S2GCollator:
    VALID_VARIANTS = {"re", "boundary_re", "boundary_joint", "joint"}

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, entity_schema: List[str], 
        rel_schema: List[str], config: Dict[str, Any]
    ) -> None:
        self._variant = config.get("model_variant")
        self._mode = config.get("mode", "budget")
        if self._variant not in self.VALID_VARIANTS or self._mode not in {"budget", "bernoulli"}:
            raise ValueError(f"Invalid model_variant '{self._variant}' or mode '{self._mode}'.")

        self._tokenizer = tokenizer
        self._entity_schema = list(entity_schema)
        self._entity_schema_set = set(entity_schema)
        self._rel_schema = list(rel_schema)
        self._rel_schema_set = set(rel_schema)
        self._cfg = config
        self._random_prompt = config.get("random_prompt", False)
        self._random_graph = config.get("random_graph", False)
        self._use_rejection = config.get("use_rejection", False)
        self._use_nesting = config.get("use_nesting", True)
        self._prompt_type = config.get("prompt_type", "natural")
        self._tok: S2GTokens = S2GTokens(self._variant, use_rejection=self._use_rejection, prompt=self._prompt_type)

        # Pre-populate TOK_CACHE in prompt.py to prevent lookup errors when prompt == 'ssi'
        get_tok(self._variant, prompt=self._prompt_type)
        
        self._step = 0

    @property
    def current_step(self) -> int: 
        return self._step
        
    @current_step.setter
    def current_step(self, value: int) -> None: 
        self._step = value
        self._cached_schedule = self._schedule_values()

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        prepare_func = getattr(self, f"_prepare_{self._variant}")
        encoder_inputs: List[str] = []
        decoder_targets: List[str] = []

        for inst in batch:
            enc, dec = prepare_func(inst)
            encoder_inputs.append(enc)
            decoder_targets.append(dec)

        return self._tokenize(encoder_inputs, decoder_targets)

    def _prepare_re(self, inst: Dict) -> Tuple[str, str]:
        pos_ent, neg_ent = self._sample_types(
            inst["entity_types"], self._entity_schema, self._cfg.get("max_ent_types")
        )
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types")
        )
        enc = build_re_encoder_input(
            pos_ent + neg_ent, pos_rel + neg_rel, inst["text"], 
            random_order=self._random_prompt, prompt=self._prompt_type
        )
        blocks = organise_filter_and_block(
            inst["entities"], inst["relations"], set(pos_ent), set(pos_rel)
        )
        dec = build_graph(
            blocks, "re", self._tok, 
            use_nesting=self._use_nesting, random_graph=self._random_graph, 
            use_rejection=self._use_rejection, rejected_ent_types=neg_ent, 
            rejected_rel_types=neg_rel
        )
        return enc, dec

    def _prepare_boundary_re(self, inst: Dict) -> Tuple[str, str]:
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types")
        )
        enc = build_boundary_re_encoder_input(
            pos_rel + neg_rel, inst["text"], 
            random_order=self._random_prompt, prompt=self._prompt_type
        )
        blocks = organise_filter_and_block(
            inst["entities"], inst["relations"], self._entity_schema_set, set(pos_rel)
        )
        dec = build_graph(
            blocks, "boundary_re", self._tok, 
            use_nesting=self._use_nesting, random_graph=self._random_graph, 
            use_rejection=self._use_rejection, rejected_rel_types=neg_rel
        )
        return enc, dec

    def _prepare_boundary_joint(self, inst: Dict) -> Tuple[str, str]:
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types")
        )
        enc = build_boundary_joint_encoder_input(
            pos_rel + neg_rel, inst["text"], 
            random_order=self._random_prompt, prompt=self._prompt_type
        )
        blocks = organise_filter_and_block(
            inst["entities"], inst["relations"], self._entity_schema_set, set(pos_rel)
        )
        dec = build_graph(
            blocks, "boundary_joint", self._tok, 
            use_nesting=self._use_nesting, random_graph=self._random_graph, 
            use_rejection=self._use_rejection, rejected_rel_types=neg_rel
        )
        return enc, dec

    def _prepare_joint(self, inst: Dict) -> Tuple[str, str]:
        pos_ent, neg_ent = self._sample_types(
            inst["entity_types"], self._entity_schema, self._cfg.get("max_ent_types")
        )
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"], self._rel_schema, self._cfg.get("max_rel_types")
        )
        enc = build_joint_encoder_input(
            pos_ent + neg_ent, pos_rel + neg_rel, inst["text"], 
            random_order=self._random_prompt, prompt=self._prompt_type
        )
        blocks = organise_filter_and_block(
            inst["entities"], inst["relations"], set(pos_ent), set(pos_rel)
        )
        dec = build_graph(
            blocks, "joint", self._tok, 
            use_nesting=self._use_nesting, random_graph=self._random_graph, 
            use_rejection=self._use_rejection, rejected_ent_types=neg_ent, 
            rejected_rel_types=neg_rel
        )
        return enc, dec

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

        pos_rate, neg_rate, pos_k, neg_k = getattr(self, "_cached_schedule", self._schedule_values())
        included_pos = [t for t in instance_types if random.random() < pos_rate]
        if len(included_pos) > pos_k:
            included_pos = random.sample(included_pos, pos_k)
        
        candidate_neg = [t for t in schema if t not in inst_set and random.random() < neg_rate]
        if len(candidate_neg) > neg_k: 
            candidate_neg = random.sample(candidate_neg, neg_k)
        
        if max_types is not None and len(candidate_neg) > (rem := max(0, max_types - len(included_pos))):
            candidate_neg = random.sample(candidate_neg, rem)
            
        return included_pos, candidate_neg

    def _schedule_values(self) -> Tuple[float, float, int, int]:
        T = max(int(self._cfg.get("max_steps", 1)), 1)
        frac = min(self._step, T) / T
        
        def lerp(start: float, end: float) -> float:
            return start + frac * (end - start)
            
        return (
            lerp(self._cfg.get("positive_rate_start", 0.9), self._cfg.get("positive_rate_end", 0.9)),
            lerp(self._cfg.get("negative_rate_start", 0.1), self._cfg.get("negative_rate_end", 0.1)),
            round(lerp(float(self._cfg.get("pos_max_start", 1)), float(self._cfg.get("pos_max_end", 20)))),
            round(lerp(float(self._cfg.get("negative_max_start", 1)), float(self._cfg.get("negative_max_end", 20))))
        )

    def _tokenize(self, encoder_inputs: List[str], decoder_targets: List[str]) -> Dict[str, Any]:
        model_inputs = self._tokenizer(
            encoder_inputs, max_length=self._cfg["max_source_length"], 
            truncation=True, padding="longest", return_tensors="pt"
        )
        label_enc = self._tokenizer(
            decoder_targets, max_length=self._cfg["max_target_length"], 
            truncation=True, padding="longest", return_tensors="pt"
        )
        
        label_ids = label_enc["input_ids"]
        label_ids.masked_fill_(label_ids == self._tokenizer.pad_token_id, -100)

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": label_ids,
        }

    def to_eval_mode(self) -> S2GCollator:
            """
            Returns a copy of the collator configured specifically for evaluation:
            enforces 'budget' mode, deterministic type sampling, and fixed token order.
            """
            eval_cfg = dict(self._cfg)
            eval_cfg["mode"] = "budget"
            eval_cfg["random_prompt"] = False
            eval_cfg["random_graph"] = False
            
            return S2GCollator(
                tokenizer=self._tokenizer,
                entity_schema=self._entity_schema,
                rel_schema=self._rel_schema,
                config=eval_cfg,
            )