"""
Constraint decoder FSM for the S2G model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase

from s2g.linearisation import AnyTokens, PIPELINE_TOKENS, get_token_ids

logger = logging.getLogger(__name__)


class Trie:
    def __init__(self, label_token_ids: Sequence[Sequence[int]], sentinel_ids: Set[int]) -> None:
        self.sentinel_ids = frozenset(sentinel_ids)
        self._root: Dict = {}
        for ids in filter(None, label_token_ids):
            node = self._root
            for tid in ids: 
                node = node.setdefault(int(tid), {})
            node["_end"] = True

    def get_valid_next(self, prefix_ids: List[int]) -> FrozenSet[int]:
        node = self._root
        for tid in prefix_ids:
            if tid not in node: 
                return self.sentinel_ids
            node = node[tid]
        
        valid = {k for k in node if k != "_end"}
        if "_end" in node: 
            valid.update(self.sentinel_ids)
            
        return frozenset(valid) or self.sentinel_ids


def _extract_ssi_labels(
        source_row: List[int], delimiter_to_task: Dict[int, str], 
        type_id: int, rel_id: int, eos_id: int, pad_id: int
    ) -> Tuple[str, List[List[int]], List[List[int]]]:
    ent_seqs, rel_seqs, curr, curr_kind, task = [], [], [], None, "boundary"

    def flush():
        if curr and curr_kind: 
            (ent_seqs if curr_kind == "type" else rel_seqs).append(list(curr))
        curr.clear()

    for tid in source_row:
        if tid in {pad_id, eos_id} | set(delimiter_to_task.keys()):
            flush()
            if tid in delimiter_to_task: 
                task = delimiter_to_task[tid]
            break
        if tid in {type_id, rel_id}:
            flush()
            curr_kind = "type" if tid == type_id else "rel"
        elif curr_kind: 
            curr.append(int(tid))

    return task, ent_seqs, rel_seqs


class FSMState(Enum):
    START, ENT_SPAN, TYPE_LABEL, REL_LABEL, TAIL_SPAN, INTER, NULL_LABEL, END = auto(), auto(), auto(), auto(), auto(), auto(), auto(), auto()

@dataclass
class _HypState:
    fsm_state: FSMState = FSMState.START
    span_tokens: List[int] = field(default_factory=list)
    label_prefix: List[int] = field(default_factory=list)
    
    def clone(self) -> '_HypState':
        """Deep copy for state transition caching."""
        return _HypState(self.fsm_state, self.span_tokens.copy(), self.label_prefix.copy())


class ConstraintDecodingProcessor(LogitsProcessor):
    def __init__(
            self, tokenizer: PreTrainedTokenizerBase, source_ids: torch.Tensor, 
            tokens: AnyTokens, num_beams: int = 1
        ) -> None:
        self.num_beams, self._batch_size = num_beams, source_ids.shape[0]
        
        tid = get_token_ids(tokenizer, tokens)
        self.ent_start_id, self.ent_end_id = tid["ent_start"], tid["ent_end"]
        self.type_id, self.rel_id, self.tail_id, self.null_id = tid["type_"], tid["rel"], tid["tail"], tid["null"]
        self.eos_id, self.pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id or 0

        delimiter_to_task = {
            tid[f]: n for f, n in [
                ("bound", "boundary"), ("ner", "ner"), ("re", "re"), 
                ("joint", "joint"), ("joint_plus", "joint+")
            ] if f in tid
        }
        self._special_ids = frozenset(tid.values()) | {self.eos_id, self.pad_id}

        self._source_token_sets, self._source_lists, self._token_to_positions = [], [], []
        for b in range(self._batch_size):
            src = source_ids[b].tolist()
            self._source_token_sets.append(frozenset(set(src) - self._special_ids))
            self._source_lists.append(src)
            pos_map: Dict[int, List[int]] = {}
            for i, token in enumerate(src): 
                pos_map.setdefault(token, []).append(i)
            self._token_to_positions.append(pos_map)

        self._tasks, self._ent_type_tries, self._rel_tries, self._null_tries = [], [], [], []
        for b in range(self._batch_size):
            task, e_seqs, r_seqs = _extract_ssi_labels(
                source_ids[b].tolist(), delimiter_to_task, 
                self.type_id, self.rel_id, self.eos_id, self.pad_id
            )
            self._tasks.append(task)
            
            self._ent_type_tries.append(
                Trie(e_seqs, {self.ent_end_id} if task == "ner" else {self.rel_id, self.ent_end_id}) 
                if task in {"ner", "joint+"} else None
            )
            self._rel_tries.append(Trie(r_seqs, {self.tail_id}) if task in {"re", "joint", "joint+"} else None)
            
            null_map = {
                "ner": (e_seqs, {self.type_id, self.eos_id}), 
                "re": (r_seqs, {self.rel_id, self.eos_id}), 
                "joint": (r_seqs, {self.rel_id, self.eos_id}), 
                "joint+": (e_seqs + r_seqs, {self.type_id, self.rel_id, self.eos_id})
            }
            n_seqs, n_sents = null_map.get(task, ([], {self.eos_id}))
            self._null_tries.append(Trie(n_seqs, n_sents))

        self._disallowed = None
        # Cache tracks states dynamically, completely eliminating O(L^2) step rebuilds
        self._state_cache: Dict[Tuple[int, ...], _HypState] = {}

    def _batch_idx(self, hyp_idx: int) -> int: 
        return hyp_idx // self.num_beams

    def _source_copy_next(self, batch_idx: int, span_tokens: List[int]) -> FrozenSet[int]:
        if not span_tokens: 
            return self._source_token_sets[batch_idx]
        
        src, n = self._source_lists[batch_idx], len(span_tokens)
        return frozenset(
            src[p + 1] for p in self._token_to_positions[batch_idx].get(span_tokens[-1], [])
            if p - n + 1 >= 0 and src[p - n + 1 : p + 1] == span_tokens 
            and p + 1 < len(src) and src[p + 1] not in self._special_ids
        )

    def _ent_span_exits(self, task: str) -> FrozenSet[int]:
        return frozenset({{
            "ner": self.type_id, "re": self.rel_id, 
            "joint": self.rel_id, "joint+": self.type_id
        }.get(task, self.ent_end_id)} | ({self.ent_end_id} if task == "joint" else set()))

    def _transition(self, state: _HypState, token_id: int) -> None:
        if token_id == self.eos_id: 
            state.fsm_state = FSMState.END
        elif token_id == self.pad_id and state.fsm_state == FSMState.START:
            pass  
        elif token_id == self.ent_start_id: 
            state.fsm_state, state.span_tokens, state.label_prefix = FSMState.ENT_SPAN, [], []
        elif token_id == self.ent_end_id: 
            state.fsm_state, state.span_tokens, state.label_prefix = FSMState.INTER, [], []
        elif token_id == self.tail_id: 
            state.fsm_state, state.span_tokens = FSMState.TAIL_SPAN, []
        elif token_id == self.null_id: 
            state.fsm_state, state.label_prefix = FSMState.NULL_LABEL, []
        elif token_id in {self.type_id, self.rel_id}:
            if state.fsm_state != FSMState.NULL_LABEL: 
                state.fsm_state = FSMState.TYPE_LABEL if token_id == self.type_id else FSMState.REL_LABEL
                if token_id == self.rel_id: 
                    state.span_tokens = []
            state.label_prefix = []
        elif state.fsm_state in {FSMState.ENT_SPAN, FSMState.TAIL_SPAN}: 
            state.span_tokens.append(token_id)
        elif state.fsm_state in {FSMState.TYPE_LABEL, FSMState.REL_LABEL, FSMState.NULL_LABEL}: 
            state.label_prefix.append(token_id)

    def _allowed_tokens(self, state: _HypState, hyp_idx: int) -> FrozenSet[int]:
        task, b_idx = self._tasks[self._batch_idx(hyp_idx)], self._batch_idx(hyp_idx)
        
        if state.fsm_state == FSMState.START: 
            return frozenset({self.ent_start_id, self.eos_id} | ({self.null_id} if task != "boundary" else set()))
            
        if state.fsm_state == FSMState.ENT_SPAN: 
            return frozenset(self._source_copy_next(b_idx, state.span_tokens) | self._ent_span_exits(task)) or frozenset({self.eos_id})
            
        if state.fsm_state == FSMState.TAIL_SPAN: 
            return frozenset(self._source_copy_next(b_idx, state.span_tokens) | {self.rel_id, self.ent_end_id}) or frozenset({self.eos_id})
            
        if state.fsm_state == FSMState.TYPE_LABEL: 
            return self._ent_type_tries[b_idx].get_valid_next(state.label_prefix) if self._ent_type_tries[b_idx] else frozenset({self.ent_end_id, self.eos_id})
            
        if state.fsm_state == FSMState.REL_LABEL: 
            return self._rel_tries[b_idx].get_valid_next(state.label_prefix) if self._rel_tries[b_idx] else frozenset({self.tail_id, self.eos_id})
            
        if state.fsm_state == FSMState.NULL_LABEL: 
            return self._null_tries[b_idx].get_valid_next(state.label_prefix)
            
        if state.fsm_state == FSMState.INTER: 
            return frozenset({self.ent_start_id, self.eos_id} | ({self.null_id} if task != "boundary" else set()))
        
        return frozenset({self.eos_id, self.pad_id})

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_hyps, vocab_size = input_ids.shape[0], scores.shape[1]
        
        if self._disallowed is None or self._disallowed.shape[0] != vocab_size:
            self._disallowed = torch.ones(vocab_size, dtype=torch.bool, device=scores.device)

        new_cache = {}

        for h in range(num_hyps):
            seq = input_ids[h].tolist()
            seq_tuple = tuple(seq)
            prev_seq_tuple = tuple(seq[:-1])
            last_token = seq[-1]
            
            # EFFICIENCY FIX: O(1) state lookup instead of O(L) token replay
            if prev_seq_tuple in self._state_cache:
                state = self._state_cache[prev_seq_tuple].clone()
                self._transition(state, last_token)
            else:
                state = _HypState()
                for token in seq:
                    self._transition(state, token)
                    
            new_cache[seq_tuple] = state

            self._disallowed.fill_(True)
            valid_tokens = [t for t in self._allowed_tokens(state, h) if t < vocab_size]
            
            # EFFICIENCY FIX: Indexed boolean mask mapping avoids explicit tensor constructor overhead
            if valid_tokens:
                self._disallowed[valid_tokens] = False
                
            scores[h].masked_fill_(self._disallowed, float("-inf"))

        # Swap in the new cache to automatically prune pruned beam branches
        self._state_cache = new_cache
        return scores


def build_constraint_processor(
        tokenizer: PreTrainedTokenizerBase, source_ids: torch.Tensor, 
        tokens: AnyTokens = PIPELINE_TOKENS, num_beams: int = 1
    ) -> ConstraintDecodingProcessor:
    return ConstraintDecodingProcessor(tokenizer=tokenizer, source_ids=source_ids, tokens=tokens, num_beams=num_beams)