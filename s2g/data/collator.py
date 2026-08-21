"""
S2G Data Collator.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from s2g.linearisation import (
    S2GTokens,
    build_graph,
    build_boundary_joint_encoder_input,
    build_joint_encoder_input,
    build_re_encoder_input, 
    build_boundary_re_encoder_input, 
    organise_filter_and_block, 
    VALID_VARIANTS
)


class S2GCollator:

    def __init__(self, 
        tokenizer: PreTrainedTokenizerBase, 
        ent_schema: List[str], 
        rel_schema: List[str], 
        config: Dict[str, Any]
    ) -> None:
        self.variant = config.get('variant')
        self.mode = config.get('mode', 'budget')
        if self.variant not in VALID_VARIANTS or self.mode not in {'budget', 'bernoulli'}:
            raise ValueError(f"Invalid model variant '{self.variant}' or mode '{self.mode}'.")

        self.tokenizer = tokenizer
        self.ent_schema = list(ent_schema)
        self.ent_schema_set = set(ent_schema)
        self.rel_schema = list(rel_schema)
        self.rel_schema_set = set(rel_schema)
        self.cfg = config
        self.random_prompt = config.get('random_prompt', False)
        self.random_graph = config.get('random_graph', False)
        self.use_rejection = config.get('use_rejection', True)
        self.use_nesting = config.get('use_nesting', True)
        self.prompt_type = config.get('prompt_type', 'natural')
        self.tok: S2GTokens = S2GTokens(self.variant, use_rejection=self.use_rejection)
        self.rng = random.Random(config.get('seed', 0))

        # Shared-memory counter: ``collate_fn`` runs inside DataLoader worker
        # processes, which hold pickled copies of this object. A plain attribute
        # written by the trainer would never reach them, freezing the bernoulli
        # curriculum at step 0 for the whole run.
        self._step = torch.zeros(1, dtype=torch.long).share_memory_()

    @property
    def current_step(self) -> int:
        return int(self._step.item())

    @current_step.setter
    def current_step(self, value: int) -> None:
        self._step.fill_(int(value))

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        prepare_func = getattr(self, f"prepare_{self.variant}")
        encoder_inputs: List[str] = []
        decoder_targets: List[str] = []

        for inst in batch:
            enc, dec = prepare_func(inst)
            encoder_inputs.append(enc)
            decoder_targets.append(dec)

        return self.tokenize(encoder_inputs, decoder_targets)

    def prepare_re(self, inst: Dict) -> Tuple[str, str]:
        pos_ent, neg_ent = self.sample_types(
            inst['entity_types'], self.ent_schema, self.cfg.get('max_ent_types')
        )
        pos_rel, neg_rel = self.sample_types(
            inst['rel_types'], self.rel_schema, self.cfg.get('max_rel_types')
        )
        enc = build_re_encoder_input(
            pos_ent + neg_ent, pos_rel + neg_rel, inst['text'], 
            random_order=self.random_prompt, prompt=self.prompt_type
        )
        blocks = organise_filter_and_block(
            inst['entities'], inst['relations'], set(pos_ent), set(pos_rel), variant='re', use_types=True
        )
        dec = build_graph(
            blocks, 're', self.tok, 
            use_nesting=self.use_nesting, random_graph=self.random_graph, 
            use_rejection=self.use_rejection, rejected_ent_types=neg_ent, 
            rejected_rel_types=neg_rel
        )
        return enc, dec

    def prepare_boundary_re(self, inst: Dict) -> Tuple[str, str]:
        pos_rel, neg_rel = self.sample_types(
            inst['rel_types'], self.rel_schema, self.cfg.get('max_rel_types')
        )
        enc = build_boundary_re_encoder_input(
            pos_rel + neg_rel, inst['text'], 
            random_order=self.random_prompt, prompt=self.prompt_type
        )
        blocks = organise_filter_and_block(
            inst['entities'], inst['relations'], set(), set(pos_rel), variant='boundary_re', use_types=False
        )
        dec = build_graph(
            blocks, 'boundary_re', self.tok, 
            use_nesting=self.use_nesting, random_graph=self.random_graph, 
            use_rejection=self.use_rejection, rejected_rel_types=neg_rel
        )
        return enc, dec

    def prepare_boundary_joint(self, inst: Dict) -> Tuple[str, str]:
        pos_rel, neg_rel = self.sample_types(
            inst['rel_types'], self.rel_schema, self.cfg.get('max_rel_types')
        )
        enc = build_boundary_joint_encoder_input(
            pos_rel + neg_rel, inst['text'], 
            random_order=self.random_prompt, prompt=self.prompt_type
        )
        blocks = organise_filter_and_block(
            inst['entities'], inst['relations'], set(), set(pos_rel), variant='boundary_joint', use_types=False
        )
        dec = build_graph(
            blocks, 'boundary_joint', self.tok, 
            use_nesting=self.use_nesting, random_graph=self.random_graph, 
            use_rejection=self.use_rejection, rejected_rel_types=neg_rel
        )
        return enc, dec

    def prepare_joint(self, inst: Dict) -> Tuple[str, str]:
        pos_ent, neg_ent = self.sample_types(
            inst['entity_types'], self.ent_schema, self.cfg.get('max_ent_types')
        )
        pos_rel, neg_rel = self.sample_types(
            inst['rel_types'], self.rel_schema, self.cfg.get('max_rel_types')
        )
        enc = build_joint_encoder_input(
            pos_ent + neg_ent, pos_rel + neg_rel, inst['text'], 
            random_order=self.random_prompt, prompt=self.prompt_type
        )
        blocks = organise_filter_and_block(
            inst['entities'], inst['relations'], set(pos_ent), set(pos_rel), variant='joint', use_types=True
        )
        dec = build_graph(
            blocks, 'joint', self.tok, 
            use_nesting=self.use_nesting, random_graph=self.random_graph, 
            use_rejection=self.use_rejection, rejected_ent_types=neg_ent, 
            rejected_rel_types=neg_rel
        )
        return enc, dec

    def sample_types(
            self, 
            instance_types: List[str], 
            schema: List[str], 
            max_types: Optional[int]
        ) -> Tuple[List[str], List[str]]:
        inst_set = set(instance_types)
        if self.mode == 'budget':
            neg_pool = [t for t in schema if t not in inst_set]
            sampled_neg = self.rng.sample(
                neg_pool, min(max(0, max_types - len(instance_types)), len(neg_pool))
            ) if max_types is not None else neg_pool

            return list(instance_types), sampled_neg

        pos_rate, neg_rate, pos_k, neg_k = self.schedule_values()
        included_pos = [t for t in instance_types]
        if len(included_pos) > pos_k:
            included_pos = self.rng.sample(included_pos, pos_k)
        
        candidate_neg = [t for t in schema if t not in inst_set]
        if len(candidate_neg) > neg_k: 
            candidate_neg = self.rng.sample(candidate_neg, neg_k)

        if max_types is not None and len(included_pos) > max_types:
            included_pos = self.rng.sample(included_pos, max_types)

        if max_types is not None and len(candidate_neg) > (rem := max(0, max_types - len(included_pos))):
            candidate_neg = self.rng.sample(candidate_neg, rem)

        included_pos = [t for t in included_pos if self.rng.random() < pos_rate]
        candidate_neg = [t for t in candidate_neg if self.rng.random() < neg_rate]
        
        return included_pos, candidate_neg

    def schedule_values(self) -> Tuple[float, float, int, int]:
        T = max(int(self.cfg.get('max_steps', 1)), 1)
        frac = min(self.current_step, T) / T
        
        def lerp(start: float, end: float) -> float:
            return start + frac * (end - start)
            
        return (
            lerp(self.cfg.get('pos_rate_start', 0.9), self.cfg.get('pos_rate_end', 0.9)),
            lerp(self.cfg.get('neg_rate_start', 0.1), self.cfg.get('neg_rate_end', 0.1)),
            round(lerp(float(self.cfg.get('pos_max_start', 1)), float(self.cfg.get('pos_max_end', 20)))),
            round(lerp(float(self.cfg.get('neg_max_start', 1)), float(self.cfg.get('neg_max_end', 20))))
        )

    def tokenize(self, encoder_inputs: List[str], decoder_targets: List[str]) -> Dict[str, Any]:
        model_inputs = self.tokenizer(
            encoder_inputs, max_length=self.cfg['max_source_length'], 
            truncation=True, padding='longest', return_tensors='pt'
        )
        label_enc = self.tokenizer(
            decoder_targets, max_length=self.cfg['max_target_length'], 
            truncation=True, padding='longest', return_tensors='pt'
        )
        
        label_ids = label_enc['input_ids']
        label_ids.masked_fill_(label_ids == self.tokenizer.pad_token_id, -100)

        return {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': label_ids,
        }

    def to_eval_mode(self) -> S2GCollator:
        """
        Returns a copy of the collator configured specifically for evaluation (enforces budget mode).
        """
        eval_cfg = dict(self.cfg)
        eval_cfg['mode'] = 'budget'
        
        return S2GCollator(
            tokenizer=self.tokenizer,
            ent_schema=self.ent_schema,
            rel_schema=self.rel_schema,
            config=eval_cfg,
        )