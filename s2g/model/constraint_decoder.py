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

from s2g.linearisation import AnyTokens, S2GTokens, get_token_ids

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
        source_row: List[int], 
        tokenizer: PreTrainedTokenizerBase,
        tokens: AnyTokens,
        tid: Dict[str, int],
        eos_id: int, 
        pad_id: int
    ) -> Tuple[str, List[List[int]], List[List[int]]]:
    ent_seqs, rel_seqs, curr = [], [], []
    curr_kind = None
    seen_ner = False
    seen_re = False
    seen_bound = False

    bound_id = tid.get("bound")
    ner_id = tid.get("ner")
    re_id = tid.get("re")
    text_id = tid.get("text")

    def flush():
        if curr and curr_kind: 
            (ent_seqs if curr_kind == "ner" else rel_seqs).append(list(curr))
        curr.clear()

    for token_id in source_row:
        if token_id in {pad_id, eos_id, text_id}:
            flush()
            break
        elif bound_id is not None and token_id == bound_id:
            seen_bound = True
            flush()
            break
        elif ner_id is not None and token_id == ner_id:
            flush()
            curr_kind = "ner"
            seen_ner = True
        elif re_id is not None and token_id == re_id:
            flush()
            curr_kind = "re"
            seen_re = True
        elif curr_kind:
            curr.append(int(token_id))

    task = None
    variant = tokens.variant
    if seen_bound:
        task = "boundary"
    elif seen_ner and seen_re:
        task = "joint" if variant not in {"re", "pipeline"} else "re"
    elif seen_ner:
        task = "ner"
    elif seen_re:
        if variant in {"re", "pipeline"}:
            task = "re"
        elif variant in {"boundary_re", "boundary_pipeline"}:
            task = "boundary_re"
        else:
            task = "boundary_joint"
    else:
        # Fallback task determination when no SSI is present (natural or false)
        if variant in {"joint", "boundary_joint", "ner", "re", "boundary", "boundary_re"}:
            task = variant
        elif variant == "pipeline":
            # If <ent> is in source_row, it's RE task (from older behavior if any)
            ent_start_id = tid.get("ent_start")
            if ent_start_id is not None and ent_start_id in source_row:
                task = "pipeline_re"
            else:
                try:
                    dec_text = tokenizer.decode(source_row)
                    if "relations of types" in dec_text or "among entities" in dec_text:
                        task = "pipeline_re"
                    else:
                        task = "ner"
                except Exception:
                    task = "ner"
        elif variant == "boundary_pipeline":
            # If <ent> is in source_row, it's boundary_joint task
            ent_start_id = tid.get("ent_start")
            if ent_start_id is not None and ent_start_id in source_row:
                task = "boundary_joint"
            else:
                try:
                    dec_text = tokenizer.decode(source_row)
                    if "relations of types" in dec_text or "among entities" in dec_text:
                        task = "pipeline_boundary_re"
                    else:
                        task = "boundary"
                except Exception:
                    task = "boundary"
        else:
            task = "pipeline"

    return task, ent_seqs, rel_seqs


class FSMState(Enum):
    START = auto()
    ENT_SPAN = auto()
    TYPE_LABEL = auto()
    REL_LABEL = auto()
    TAIL_SPAN = auto()
    TAIL_TYPE_LABEL = auto()
    NEST = auto()
    INTER = auto()
    NULL_LABEL = auto()
    END = auto()
    ENT_DECL_SPAN = auto()
    TRIPLET_HEAD_SPAN = auto()
    # New states for re/boundary_re
    TRIP_HEAD_SPAN = auto()
    TRIP_HTYPE = auto()
    TRIP_REL = auto()
    TRIP_TAIL_SPAN = auto()
    TRIP_TTYPE = auto()
    TRIP_AND_EXPECTED = auto()
    TRIP_NULL = auto()

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
            tokens: AnyTokens, num_beams: int = 1,
            entity_schema: Optional[List[str]] = None, rel_schema: Optional[List[str]] = None
        ) -> None:
        self.num_beams, self._batch_size = num_beams, source_ids.shape[0]
        
        tid = get_token_ids(tokenizer, tokens)
        self.ent_start_id, self.ent_end_id = tid.get("ent_start"), tid.get("ent_end")
        self.type_id, self.rel_id, self.tail_id, self.null_id = tid.get("type_"), tid.get("rel"), tid.get("tail"), tid.get("null")
        self.head_id, self.nest_id = tid.get("head"), tid.get("nest")
        self.trip_id, self.sep_id = tid.get("trip"), tid.get("sep")
        self.eos_id, self.pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id or 0

        self.and_tokens = frozenset(tokenizer.encode(" and", add_special_tokens=False) + tokenizer.encode("and", add_special_tokens=False))

        self._special_ids = frozenset(v for v in tid.values() if v is not None) | {self.eos_id, self.pad_id}

        self._source_token_sets, self._source_lists, self._token_to_positions = [], [], []
        for b in range(self._batch_size):
            src = source_ids[b].tolist()
            self._source_token_sets.append(frozenset(set(src) - self._special_ids))
            self._source_lists.append(src)
            pos_map: Dict[int, List[int]] = {}
            for i, token in enumerate(src): 
                pos_map.setdefault(token, []).append(i)
            self._token_to_positions.append(pos_map)

        self._tasks, self._ent_type_tries, self._rel_tries, self._null_tries, self._tail_type_tries = [], [], [], [], []
        for b in range(self._batch_size):
            task, e_seqs, r_seqs = _extract_ssi_labels(
                source_row=source_ids[b].tolist(), 
                tokenizer=tokenizer,
                tokens=tokens, 
                tid=tid, 
                eos_id=self.eos_id, 
                pad_id=self.pad_id
            )

            # Load schemas from file if not provided as arguments and they are needed
            if (entity_schema is None or rel_schema is None) and (not e_seqs or not r_seqs):
                import sys
                schema_file = None
                entity_schema_file = None
                for idx, arg in enumerate(sys.argv):
                    if arg == "--schema_file" and idx + 1 < len(sys.argv):
                        schema_file = sys.argv[idx + 1]
                    elif arg.startswith("--schema_file="):
                        schema_file = arg.split("=", 1)[1]
                    elif arg == "--entity_schema_file" and idx + 1 < len(sys.argv):
                        entity_schema_file = sys.argv[idx + 1]
                    elif arg.startswith("--entity_schema_file="):
                        entity_schema_file = arg.split("=", 1)[1]
                
                config_path = None
                for idx, arg in enumerate(sys.argv):
                    if arg == "--config" and idx + 1 < len(sys.argv):
                        config_path = sys.argv[idx + 1]
                    elif arg.startswith("--config="):
                        config_path = arg.split("=", 1)[1]
                if config_path and (not schema_file or not entity_schema_file):
                    try:
                        from omegaconf import OmegaConf
                        cfg = OmegaConf.load(config_path)
                        if not schema_file and hasattr(cfg, "data") and cfg.data.get("schema_file"):
                            schema_file = cfg.data.schema_file
                        if not entity_schema_file and hasattr(cfg, "data") and cfg.data.get("entity_schema_file"):
                            entity_schema_file = cfg.data.entity_schema_file
                    except Exception:
                        pass
                
                from s2g.scripts.config_utils import load_schema, load_entity_schema
                if rel_schema is None and schema_file:
                    try:
                        rel_schema = load_schema(schema_file)
                    except Exception as e:
                        logger.warning("FSM: Failed to load rel_schema from %s: %s", schema_file, e)
                if entity_schema is None and entity_schema_file:
                    try:
                        entity_schema = load_entity_schema(entity_schema_file)
                    except Exception as e:
                        logger.warning("FSM: Failed to load entity_schema from %s: %s", entity_schema_file, e)

            # If e_seqs is empty and entity_schema is available, tokenize the schema
            if not e_seqs and entity_schema and task in {"ner", "joint", "re"}:
                e_seqs = [tokenizer.encode(t, add_special_tokens=False) for t in entity_schema]
                
            # If r_seqs is empty and rel_schema is available, tokenize the schema
            if not r_seqs and rel_schema and task in {"re", "boundary_re", "boundary_joint", "joint", "pipeline_re", "pipeline_boundary_re"}:
                r_seqs = [tokenizer.encode(t, add_special_tokens=False) for t in rel_schema]

            self._tasks.append(task)
            
            self._ent_type_tries.append(
                Trie(e_seqs, {self.rel_id} if task in {"joint", "re", "pipeline_re"} else self._get_inter_tokens(task))
                if task in {"ner", "joint", "re", "pipeline_re"} else None
            )
            self._tail_type_tries.append(
                Trie(e_seqs, {self.sep_id, self.trip_id, self.null_id, self.eos_id} if task == "re" else {self.nest_id, self.head_id, self.null_id, self.eos_id})
                if task in {"joint", "re", "pipeline_re"} else None
            )
            self._rel_tries.append(Trie(r_seqs, {self.tail_id}) if task in {"re", "boundary_re", "boundary_joint", "joint", "pipeline_re", "pipeline_boundary_re"} else None)
            
            null_map = {
                "ner": (e_seqs, {self.null_id, self.eos_id}), 
                "boundary_joint": (r_seqs, {self.null_id, self.eos_id}), 
                "joint": (e_seqs + r_seqs, {self.null_id, self.eos_id}),
                "pipeline_re": (e_seqs + r_seqs, {self.null_id, self.eos_id}),
                "pipeline_boundary_re": (r_seqs, {self.null_id, self.eos_id})
                # re and boundary_re are handled openly after <null> without a Trie
            }
            n_seqs, n_sents = null_map.get(task, ([], {self.eos_id}))
            self._null_tries.append(Trie(n_seqs, n_sents))

        self._disallowed = None
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

    def _get_inter_tokens(self, task: str) -> Set[int]:
        if task == "boundary":
            return {self.ent_start_id, self.eos_id}
        if task in {"re", "boundary_re", "boundary_joint", "joint", "pipeline_re", "pipeline_boundary_re"}:
            return {self.head_id, self.null_id, self.eos_id}
        return {self.ent_start_id, self.null_id, self.eos_id}

    def _ent_span_exits(self, task: str) -> FrozenSet[int]:
        if task == "boundary":
            return frozenset(self._get_inter_tokens(task))
        if task in {"boundary_joint", "boundary_re", "pipeline_boundary_re"}:
            return frozenset({self.rel_id} | self._get_inter_tokens(task))
        if task in {"ner", "joint", "re", "pipeline_re"}:
            return frozenset({self.type_id})
        return frozenset()

    def _transition(self, state: _HypState, token_id: int, task: str) -> None:
        if task in {"re", "boundary_re"}:
            if token_id == self.eos_id:
                state.fsm_state = FSMState.END
            elif token_id == self.pad_id and state.fsm_state == FSMState.START:
                pass
            elif token_id == self.trip_id:
                state.fsm_state, state.span_tokens = FSMState.TRIP_HEAD_SPAN, []
            elif token_id == self.null_id:
                state.fsm_state = FSMState.TRIP_NULL
            elif state.fsm_state == FSMState.TRIP_NULL:
                pass # Accept anything after null until EOS
            elif token_id == self.sep_id:
                if state.fsm_state == FSMState.TRIP_HEAD_SPAN:
                    state.fsm_state, state.label_prefix = FSMState.TRIP_HTYPE if task == "re" else FSMState.TRIP_REL, []
                elif state.fsm_state == FSMState.TRIP_HTYPE:
                    state.fsm_state, state.label_prefix = FSMState.TRIP_REL, []
                elif state.fsm_state == FSMState.TRIP_REL:
                    state.fsm_state, state.span_tokens = FSMState.TRIP_TAIL_SPAN, []
                elif state.fsm_state == FSMState.TRIP_TAIL_SPAN:
                    state.fsm_state, state.label_prefix = FSMState.TRIP_TTYPE if task == "re" else FSMState.TRIP_AND_EXPECTED, []
                elif state.fsm_state == FSMState.TRIP_TTYPE:
                    state.fsm_state = FSMState.TRIP_AND_EXPECTED
                elif state.fsm_state == FSMState.TRIP_AND_EXPECTED:
                    # we just received the <sep> after "and"
                    state.fsm_state, state.label_prefix = FSMState.TRIP_REL, []
            else:
                if state.fsm_state in {FSMState.TRIP_HEAD_SPAN, FSMState.TRIP_TAIL_SPAN}:
                    state.span_tokens.append(token_id)
                elif state.fsm_state in {FSMState.TRIP_HTYPE, FSMState.TRIP_REL, FSMState.TRIP_TTYPE}:
                    state.label_prefix.append(token_id)
            return

        if task in {"joint", "boundary_joint"}:
            if token_id == self.eos_id: 
                state.fsm_state = FSMState.END
            elif token_id == self.pad_id and state.fsm_state == FSMState.START:
                pass  
            elif token_id == self.ent_start_id: 
                state.fsm_state, state.span_tokens, state.label_prefix = FSMState.ENT_DECL_SPAN, [], []
            elif token_id == self.head_id: 
                state.fsm_state, state.span_tokens, state.label_prefix = FSMState.TRIPLET_HEAD_SPAN, [], []
            elif token_id == self.type_id: 
                if state.fsm_state == FSMState.ENT_DECL_SPAN:
                    state.fsm_state, state.label_prefix = FSMState.TYPE_LABEL, []
                elif state.fsm_state == FSMState.NULL_LABEL:
                    pass
                else:
                    pass
            elif token_id == self.rel_id: 
                state.fsm_state, state.span_tokens, state.label_prefix = FSMState.REL_LABEL, [], []
            elif token_id == self.tail_id: 
                state.fsm_state, state.span_tokens = FSMState.TAIL_SPAN, []
            elif token_id == self.nest_id:
                state.fsm_state = FSMState.NEST
            elif token_id == self.null_id: 
                state.fsm_state, state.label_prefix = FSMState.NULL_LABEL, []
            else:
                if state.fsm_state in {FSMState.ENT_DECL_SPAN, FSMState.TRIPLET_HEAD_SPAN, FSMState.TAIL_SPAN}: 
                    state.span_tokens.append(token_id)
                elif state.fsm_state in {FSMState.TYPE_LABEL, FSMState.REL_LABEL, FSMState.NULL_LABEL}: 
                    state.label_prefix.append(token_id)
            return

        if token_id == self.eos_id: 
            state.fsm_state = FSMState.END
        elif token_id == self.pad_id and state.fsm_state == FSMState.START:
            pass  
        elif token_id in {self.ent_start_id, self.head_id}: 
            state.fsm_state, state.span_tokens, state.label_prefix = FSMState.ENT_SPAN, [], []
        elif token_id == self.ent_end_id: 
            state.fsm_state, state.span_tokens, state.label_prefix = FSMState.INTER, [], []
        elif token_id == self.tail_id: 
            state.fsm_state, state.span_tokens = FSMState.TAIL_SPAN, []
        elif token_id == self.null_id: 
            state.fsm_state, state.label_prefix = FSMState.NULL_LABEL, []
        elif token_id == self.nest_id:
            state.fsm_state = FSMState.NEST
        elif token_id in {self.type_id, self.rel_id}:
            if state.fsm_state != FSMState.NULL_LABEL: 
                if token_id == self.type_id:
                    state.fsm_state = FSMState.TAIL_TYPE_LABEL if state.fsm_state == FSMState.TAIL_SPAN else FSMState.TYPE_LABEL
                else:
                    state.fsm_state = FSMState.REL_LABEL
                if token_id == self.rel_id: 
                    state.span_tokens = []
            state.label_prefix = []
        elif state.fsm_state in {FSMState.ENT_SPAN, FSMState.TAIL_SPAN}: 
            state.span_tokens.append(token_id)
        elif state.fsm_state in {FSMState.TYPE_LABEL, FSMState.TAIL_TYPE_LABEL, FSMState.REL_LABEL, FSMState.NULL_LABEL}: 
            state.label_prefix.append(token_id)

    def _allowed_tokens(self, state: _HypState, hyp_idx: int) -> FrozenSet[int]:
        task, b_idx = self._tasks[self._batch_idx(hyp_idx)], self._batch_idx(hyp_idx)
        
        if task in {"re", "boundary_re"}:
            if state.fsm_state == FSMState.START:
                return frozenset({self.trip_id, self.eos_id} | {self.null_id})
                
            if state.fsm_state == FSMState.TRIP_HEAD_SPAN:
                return frozenset(self._source_copy_next(b_idx, state.span_tokens) | {self.sep_id}) or frozenset({self.eos_id})
                
            if state.fsm_state == FSMState.TRIP_HTYPE:
                return self._ent_type_tries[b_idx].get_valid_next(state.label_prefix) if self._ent_type_tries[b_idx] else frozenset({self.sep_id})
                
            if state.fsm_state == FSMState.TRIP_REL:
                return self._rel_tries[b_idx].get_valid_next(state.label_prefix) if self._rel_tries[b_idx] else frozenset({self.sep_id})
                
            if state.fsm_state == FSMState.TRIP_TAIL_SPAN:
                exits = {self.sep_id} if task == "re" else {self.sep_id, self.trip_id, self.null_id, self.eos_id}
                return frozenset(self._source_copy_next(b_idx, state.span_tokens) | exits) or frozenset({self.eos_id})
                
            if state.fsm_state == FSMState.TRIP_TTYPE:
                return self._tail_type_tries[b_idx].get_valid_next(state.label_prefix) if self._tail_type_tries[b_idx] else frozenset({self.sep_id, self.trip_id, self.null_id, self.eos_id})
                
            if state.fsm_state == FSMState.TRIP_AND_EXPECTED:
                # Can be "and", or <trip>, or <null>, or <eos>
                # "and" tokens are allowed. We also need to allow <sep> if the model has generated "and" and is ready to close.
                # Since we don't strictly track the exact sub-tokens of "and", we allow and_tokens + sep_id + trip_id + null_id + eos_id
                return frozenset(self.and_tokens | {self.sep_id, self.trip_id, self.null_id, self.eos_id})
                
            if state.fsm_state == FSMState.TRIP_NULL:
                # Any valid vocab token is allowed until EOS
                return frozenset() # We will handle this by returning a full set of vocab, or just not masking anything in the caller
                
            return frozenset({self.eos_id, self.pad_id})
            
        if task in {"joint", "boundary_joint"}:
            if state.fsm_state == FSMState.START: 
                return frozenset({self.ent_start_id, self.head_id, self.eos_id} | ({self.null_id} if task != "boundary" else set()))
            
            if state.fsm_state == FSMState.ENT_DECL_SPAN:
                exits = {self.type_id} if task == "joint" else {self.ent_start_id, self.head_id, self.null_id, self.eos_id}
                return frozenset(self._source_copy_next(b_idx, state.span_tokens) | exits) or frozenset({self.eos_id})
                
            if state.fsm_state == FSMState.TYPE_LABEL:
                sentinels = {self.ent_start_id, self.head_id, self.null_id, self.eos_id}
                return self._ent_type_tries[b_idx].get_valid_next(state.label_prefix) if self._ent_type_tries[b_idx] else frozenset(sentinels)
                
            if state.fsm_state == FSMState.TRIPLET_HEAD_SPAN:
                return frozenset(self._source_copy_next(b_idx, state.span_tokens) | {self.rel_id}) or frozenset({self.eos_id})
                
            if state.fsm_state == FSMState.REL_LABEL:
                return self._rel_tries[b_idx].get_valid_next(state.label_prefix) if self._rel_tries[b_idx] else frozenset({self.tail_id, self.eos_id})
                
            if state.fsm_state == FSMState.TAIL_SPAN:
                exits = {self.nest_id, self.head_id, self.null_id, self.eos_id}
                return frozenset(self._source_copy_next(b_idx, state.span_tokens) | exits) or frozenset({self.eos_id})
                
            if state.fsm_state == FSMState.NEST:
                return frozenset({self.rel_id})
                
            if state.fsm_state == FSMState.NULL_LABEL:
                return self._null_tries[b_idx].get_valid_next(state.label_prefix)
                
            return frozenset({self.eos_id, self.pad_id})

        if state.fsm_state == FSMState.START: 
            if task in {"re", "boundary_joint", "joint", "pipeline_re", "pipeline_boundary_re"}:
                return frozenset({self.head_id, self.eos_id} | ({self.null_id} if task != "boundary" else set()))
            return frozenset({self.ent_start_id, self.eos_id} | ({self.null_id} if task != "boundary" else set()))
            
        if state.fsm_state == FSMState.ENT_SPAN: 
            return frozenset(self._source_copy_next(b_idx, state.span_tokens) | self._ent_span_exits(task)) or frozenset({self.eos_id})
            
        if state.fsm_state == FSMState.TAIL_SPAN:
            exits = set()
            if task in {"joint", "re", "pipeline_re"}:
                exits.add(self.type_id)
            elif task in {"boundary_re", "boundary_joint", "pipeline_boundary_re"}:
                exits.update({self.nest_id, self.head_id, self.null_id, self.eos_id})
            return frozenset(self._source_copy_next(b_idx, state.span_tokens) | exits) or frozenset({self.eos_id})
            
        if state.fsm_state == FSMState.TYPE_LABEL: 
            return self._ent_type_tries[b_idx].get_valid_next(state.label_prefix) if self._ent_type_tries[b_idx] else frozenset(self._get_inter_tokens(task))
            
        if state.fsm_state == FSMState.TAIL_TYPE_LABEL:
            return self._tail_type_tries[b_idx].get_valid_next(state.label_prefix) if self._tail_type_tries[b_idx] else frozenset({self.nest_id, self.head_id, self.null_id, self.eos_id})

        if state.fsm_state == FSMState.REL_LABEL: 
            return self._rel_tries[b_idx].get_valid_next(state.label_prefix) if self._rel_tries[b_idx] else frozenset({self.tail_id, self.eos_id})
            
        if state.fsm_state == FSMState.NEST:
            return frozenset({self.rel_id})

        if state.fsm_state == FSMState.NULL_LABEL: 
            return self._null_tries[b_idx].get_valid_next(state.label_prefix)
            
        if state.fsm_state == FSMState.INTER: 
            if task in {"re", "boundary_joint", "joint", "pipeline_re", "pipeline_boundary_re"}:
                return frozenset({self.head_id, self.eos_id} | ({self.null_id} if task != "boundary" else set()))
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
            task = self._tasks[self._batch_idx(h)]
            if prev_seq_tuple in self._state_cache:
                state = self._state_cache[prev_seq_tuple].clone()
                self._transition(state, last_token, task)
            else:
                state = _HypState()
                for token in seq:
                    self._transition(state, token, task)
                    
            new_cache[seq_tuple] = state

            self._disallowed.fill_(True)
            valid_tokens = [t for t in self._allowed_tokens(state, h) if t < vocab_size]
            if state.fsm_state == FSMState.TRIP_NULL:
                # Don't mask anything during natural language rejection
                self._disallowed.fill_(False)
            elif valid_tokens:
                self._disallowed[valid_tokens] = False
                
            scores[h].masked_fill_(self._disallowed, float("-inf"))

        # Swap in the new cache to automatically prune pruned beam branches
        self._state_cache = new_cache
        return scores


def build_constraint_processor(
        tokenizer: PreTrainedTokenizerBase, source_ids: torch.Tensor, 
        tokens: AnyTokens = S2GTokens("pipeline"), num_beams: int = 1,
        entity_schema: Optional[List[str]] = None, rel_schema: Optional[List[str]] = None
    ) -> ConstraintDecodingProcessor:
    return ConstraintDecodingProcessor(tokenizer=tokenizer, source_ids=source_ids, tokens=tokens, num_beams=num_beams, entity_schema=entity_schema, rel_schema=rel_schema)