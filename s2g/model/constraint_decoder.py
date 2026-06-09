"""
Constraint decoder for the S2G model.

Implements the finite-state machine (FSM) described in Section 8.1 of
the S2G specification.  The FSM is applied during beam search at test
time; it is **not** used during training or validation.

The processor enforces three constraints simultaneously:

1. **Copy constraint.** Entity spans and tail spans are restricted to
   exact substrings of the source text.

2. **Label trie.** Type and relation type labels are generated through a
   prefix trie built from the SSI labels prompted for that instance.
   Terminal transitions use task-specific sentinels (e.g. ``<tail>`` for
   relation labels, ``</ent>`` for entity type labels in NER).

3. **Null trie.** Rejected labels in the ``<null>`` block use a
   separate trie whose sentinels are the type/relation separator tokens
   and EOS — never ``<tail>``.

Task identification
-------------------
The task is identified per batch item by scanning ``source_ids`` for the
first task delimiter token (``<bound>``, ``<ner>``, ``<re>``, ``<joint>``,
or ``<joint+>``).  The delimiter determines:

- Which FSM states and transitions are active.
- Which SSI labels are extracted (entity types from ``<type>``-prefixed
  spans, relation types from ``<rel>``-prefixed spans).
- Which sentinel sets are used for each trie.

SSI extraction
--------------
The extractor walks each source row left-to-right.  Tokens are
accumulated into the current label span when preceded by ``<type>`` or
``<rel>``; the completed span is appended to the entity-type list or the
relation-type list respectively.  Scanning stops at the task delimiter.

State machine overview
----------------------
A unified FSM covers all five tasks.  Not all states are reachable for
every task — the ``_allowed_tokens`` method enforces task-specific
transitions.  States:

    START       → <ent>            → ENT_SPAN
    ENT_SPAN    → source copy      → ENT_SPAN   (accumulate span)
                → <type>          → TYPE_LABEL  (NER, Joint+)
                → <rel>           → REL_LABEL   (RE, Joint)
                → </ent>          → INTER       (Boundary, Joint no-rels)
    TYPE_LABEL  → trie             → TYPE_LABEL
                → sentinel        → REL_LABEL / INTER
    REL_LABEL   → trie             → REL_LABEL
                → <tail> sentinel  → TAIL_SPAN
    TAIL_SPAN   → source copy      → TAIL_SPAN
                → <rel>           → REL_LABEL
                → </ent>          → INTER
    INTER       → <ent>            → ENT_SPAN
                → <null>          → NULL_LABEL  (not Boundary)
                → EOS             → END
    NULL_LABEL  → null trie        → NULL_LABEL
                → sentinel        → NULL_LABEL  (separator; reset prefix)
                → EOS             → END
    END         → EOS / pad only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import torch
from transformers import LogitsProcessor, PreTrainedTokenizerBase

from s2g.linearisation import (
    AnyTokens,
    PIPELINE_TOKENS,
    get_token_ids,
)

logger = logging.getLogger(__name__)


# ---- TRIE ----


class Trie:
    """Prefix tree over pre-tokenised label ID sequences.

    Each label is a sequence of subword token IDs.  At query time, given
    the already-generated prefix, ``get_valid_next`` returns the set of
    valid next token IDs.  At leaf nodes (complete labels) the sentinel
    IDs are also valid.

    Args:
        label_token_ids: Sequences of token IDs, one per label.
        sentinel_ids:    Token IDs that terminate a complete label.
    """

    def __init__(
        self,
        label_token_ids: Sequence[Sequence[int]],
        sentinel_ids: Set[int],
    ) -> None:
        self.sentinel_ids: FrozenSet[int] = frozenset(sentinel_ids)
        self._root: Dict = {}
        for ids in label_token_ids:
            if not ids:
                continue
            node = self._root
            for tid in ids:
                tid = int(tid)
                if tid not in node:
                    node[tid] = {}
                node = node[tid]
            node["_end"] = True

    def get_valid_next(self, prefix_ids: List[int]) -> FrozenSet[int]:
        """Return valid next token IDs given an already-generated prefix.

        If the prefix exhausts a complete label (leaf), sentinels are
        included.  If the prefix falls off the trie (malformed output),
        only sentinels are returned as a recovery.

        Args:
            prefix_ids: Token IDs generated so far for this label.
        """
        node = self._root
        for tid in prefix_ids:
            if tid not in node:
                return self.sentinel_ids   # fell off — allow sentinels
            node = node[tid]

        valid: Set[int] = set()
        for key in node:
            if key != "_end":
                valid.add(key)
        if "_end" in node:
            valid.update(self.sentinel_ids)

        return frozenset(valid) if valid else self.sentinel_ids


# ---- SSI LABEL EXTRACTION ----


def _extract_ssi_labels(
    source_row: List[int],
    delimiter_id_to_task: Dict[int, str],
    type_id: int,
    rel_id: int,
    eos_id: int,
    pad_id: int,
) -> Tuple[str, List[List[int]], List[List[int]]]:
    """Scan one encoder source row and extract the SSI labels.

    Walks left-to-right, collecting token IDs between ``<type>`` markers
    into the entity-type list and between ``<rel>`` markers into the
    relation-type list.  Stops at the first task delimiter token.

    Args:
        source_row:           Token ID list for one source instance.
        delimiter_id_to_task: Maps task delimiter token ID → task string.
        type_id:              Token ID of ``<type>``.
        rel_id:               Token ID of ``<rel>``.
        eos_id, pad_id:       Terminal tokens.

    Returns:
        ``(task, ent_type_seqs, rel_type_seqs)``

        - *task*: one of ``"boundary"``, ``"ner"``, ``"re"``, ``"joint"``,
          ``"joint+"``; defaults to ``"boundary"`` if no delimiter found.
        - *ent_type_seqs*: list of token-ID sequences for entity types.
        - *rel_type_seqs*: list of token-ID sequences for relation types.
    """
    ent_type_seqs: List[List[int]] = []
    rel_type_seqs: List[List[int]] = []
    task = "boundary"

    current:      List[int]      = []
    current_kind: Optional[str]  = None   # "type" or "rel"

    def _flush() -> None:
        """Commit the current label span to the appropriate list."""
        if current and current_kind == "type":
            ent_type_seqs.append(list(current))
        elif current and current_kind == "rel":
            rel_type_seqs.append(list(current))
        current.clear()

    for tid in source_row:
        if tid in (pad_id, eos_id):
            _flush()
            break
        if tid in delimiter_id_to_task:
            _flush()
            task = delimiter_id_to_task[tid]
            break
        if tid == type_id:
            _flush()
            current_kind = "type"
        elif tid == rel_id:
            _flush()
            current_kind = "rel"
        else:
            if current_kind is not None:
                current.append(int(tid))

    return task, ent_type_seqs, rel_type_seqs


# ---- FSM STATES ----


class FSMState(Enum):
    """Unified FSM state set covering all five task grammars."""
    START       = auto()
    ENT_SPAN    = auto()   # generating entity / head span (source copy)
    TYPE_LABEL  = auto()   # generating entity type label (NER, Joint+)
    REL_LABEL   = auto()   # generating relation type label (RE, Joint, Joint+)
    TAIL_SPAN   = auto()   # generating tail span (source copy)
    INTER       = auto()   # between entity blocks (after </ent>)
    NULL_LABEL  = auto()   # inside rejection block
    END         = auto()


@dataclass
class _HypState:
    """Mutable per-hypothesis FSM state."""
    fsm_state:    FSMState = FSMState.START
    span_tokens:  List[int] = field(default_factory=list)
    label_prefix: List[int] = field(default_factory=list)


# ---- CONSTRAINT LOGITS PROCESSOR ----


class ConstraintDecodingProcessor(LogitsProcessor):
    """HuggingFace ``LogitsProcessor`` enforcing the S2G SEL grammar.

    One instance is created per ``generate()`` call.  It is stateful:
    it tracks one FSM state per beam hypothesis and holds per-batch-item
    label tries (beams sharing the same source share the same tries).

    Per-item tries are built from the SSI labels extracted from each row
    of ``source_ids``; they are therefore consistent with exactly the
    labels the encoder was shown.

    Args:
        tokenizer:  HuggingFace tokeniser with S2G tokens registered.
        source_ids: Encoder input IDs ``(batch, src_len)``.
        tokens:     Token registry — ``PIPELINE_TOKENS`` or
                    ``JOINT_TOKENS`` matching the loaded model.
        num_beams:  Beams per batch item.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        source_ids: torch.Tensor,
        tokens: AnyTokens,
        num_beams: int = 1,
    ) -> None:
        self.num_beams  = num_beams

        # ---- Resolve token IDs ----
        tid = get_token_ids(tokenizer, tokens)
        self.ent_start_id: int = tid["ent_start"]
        self.ent_end_id:   int = tid["ent_end"]
        self.type_id:      int = tid["type_"]
        self.rel_id:       int = tid["rel"]
        self.tail_id:      int = tid["tail"]
        self.null_id:      int = tid["null"]
        self.eos_id:       int = tokenizer.eos_token_id
        self.pad_id:       int = tokenizer.pad_token_id or 0

        # Delimiter → task mapping (only keys present in the registry).
        delimiter_id_to_task: Dict[int, str] = {}
        for field_name, task_name in [
            ("bound",      "boundary"),
            ("ner",        "ner"),
            ("re",         "re"),
            ("joint",      "joint"),
            ("joint_plus", "joint+"),
        ]:
            if field_name in tid:
                delimiter_id_to_task[tid[field_name]] = task_name

        batch_size = source_ids.shape[0]
        self._batch_size = batch_size

        # All structural and delimiter token IDs.  These are already encoded
        # as task-specific exits in _ent_span_exits; they must NOT also bleed
        # into _source_copy_next, or a state like ENT_SPAN (NER) could expose
        # </ent> as a copy-next candidate — allowing the entity block to close
        # without a type label, violating the NER grammar.
        self._special_ids: FrozenSet[int] = frozenset(tid.values()) | {
            self.eos_id, self.pad_id
        }

        # ---- Per-batch-item source token sets (for copy constraint) ----
        # Pre-filtered: special tokens excluded so only content tokens are
        # available as span-copy candidates for the empty-prefix case.
        self._source_token_sets: List[FrozenSet[int]] = []
        for b in range(batch_size):
            src_set = set(source_ids[b].tolist()) - self._special_ids
            self._source_token_sets.append(frozenset(src_set))

        # ---- Pre-computed source lists and position index ----
        # Avoids repeated tensor→list conversion and O(src_len) linear scans
        # in _source_copy_next during beam search.  The position index maps
        # each token ID to the list of positions where it appears in the
        # source; _source_copy_next uses it to look up only positions where
        # the last token of the current span prefix appears.
        self._source_lists: List[List[int]] = []
        self._token_to_positions: List[Dict[int, List[int]]] = []
        for b in range(batch_size):
            src = source_ids[b].tolist()
            self._source_lists.append(src)
            pos_map: Dict[int, List[int]] = {}
            for i, tid in enumerate(src):
                pos_map.setdefault(tid, []).append(i)
            self._token_to_positions.append(pos_map)

        # ---- Per-batch-item tasks and tries ----
        self._tasks:         List[str]            = []
        self._ent_type_tries: List[Optional[Trie]] = []
        self._rel_tries:      List[Optional[Trie]] = []
        self._null_tries:     List[Trie]            = []

        for b in range(batch_size):
            task, ent_seqs, rel_seqs = _extract_ssi_labels(
                source_row=source_ids[b].tolist(),
                delimiter_id_to_task=delimiter_id_to_task,
                type_id=self.type_id,
                rel_id=self.rel_id,
                eos_id=self.eos_id,
                pad_id=self.pad_id,
            )
            self._tasks.append(task)

            # Entity type trie — NER (sentinel: </ent>) and
            # Joint+ (sentinels: <rel> | </ent>).
            if task == "ner":
                ent_trie: Optional[Trie] = Trie(
                    ent_seqs,
                    sentinel_ids={self.ent_end_id},
                )
            elif task == "joint+":
                ent_trie = Trie(
                    ent_seqs,
                    sentinel_ids={self.rel_id, self.ent_end_id},
                )
            else:
                ent_trie = None
            self._ent_type_tries.append(ent_trie)

            # Relation type trie — RE, Joint, Joint+ (sentinel: <tail>).
            if task in ("re", "joint", "joint+"):
                rel_trie: Optional[Trie] = Trie(
                    rel_seqs,
                    sentinel_ids={self.tail_id},
                )
            else:
                rel_trie = None
            self._rel_tries.append(rel_trie)

            # Null trie — sentinels are the type/rel separators and EOS.
            # Labels are the union of whichever SSI types can appear in
            # this task's null block.
            if task == "ner":
                null_seqs    = ent_seqs
                null_sentinel = {self.type_id, self.eos_id}
            elif task in ("re", "joint"):
                null_seqs    = rel_seqs
                null_sentinel = {self.rel_id, self.eos_id}
            elif task == "joint+":
                null_seqs    = ent_seqs + rel_seqs
                null_sentinel = {self.type_id, self.rel_id, self.eos_id}
            else:                              # boundary — no null block
                null_seqs    = []
                null_sentinel = {self.eos_id}
            self._null_tries.append(Trie(null_seqs, sentinel_ids=null_sentinel))

        # Per-hypothesis states; initialised lazily on first __call__.
        self._states:    Optional[List[_HypState]] = None
        self._disallowed: Optional[torch.Tensor]   = None

    # --- Initialisation ---

    def _init_states(self, total_hypotheses: int, vocab_size: int, device: torch.device) -> None:
        self._states = [_HypState() for _ in range(total_hypotheses)]
        # Reusable boolean mask buffer (True = token is blocked).
        # Allocated once per generate() call; reset via fill_ each step.
        self._disallowed: torch.Tensor = torch.ones(
            vocab_size, dtype=torch.bool, device=device
        )

    def _batch_idx(self, hyp_idx: int) -> int:
        """Map flat hypothesis index to the original batch-item index."""
        return hyp_idx // self.num_beams

    # --- Source-copy next tokens ---

    def _source_copy_next(
        self,
        batch_idx: int,
        span_tokens: List[int],
    ) -> FrozenSet[int]:
        """Return content tokens that can extend the current span by one position.

        Structural special tokens are excluded from the returned set; they are
        already encoded as task-specific exits in ``_ent_span_exits`` and must
        not bleed in via the copy mechanism (e.g. ``</ent>`` following a span
        in the NER augmented source must not be reachable from ENT_SPAN as a
        copy candidate — only ``<type>`` is a valid exit for NER).

        When *span_tokens* is empty, any source content token is valid (first
        token of the span).  When non-empty, finds all positions where
        *span_tokens* matches in the source and returns the following
        non-special token at each match.

        Args:
            batch_idx:   Index into the original (unexpanded) batch.
            span_tokens: Token IDs generated so far for this span.
        """
        if not span_tokens:
            return frozenset(self._source_token_sets[batch_idx])

        src      = self._source_lists[batch_idx]
        n        = len(span_tokens)
        last_tid = span_tokens[-1]
        # Only check positions where the last token of the prefix appears —
        # avoids an O(src_len) scan on every beam step.
        valid: Set[int] = set()
        for p in self._token_to_positions[batch_idx].get(last_tid, []):
            start = p - n + 1
            if start >= 0 and src[start : p + 1] == span_tokens:
                nxt_pos = p + 1
                if nxt_pos < len(src):
                    nxt = src[nxt_pos]
                    if nxt not in self._special_ids:
                        valid.add(nxt)

        return frozenset(valid)

    # --- Task-specific ENT_SPAN exit tokens ---

    def _ent_span_exits(self, task: str) -> FrozenSet[int]:
        """Structural tokens that can immediately follow an entity span.

        Boundary  : </ent>
        NER       : <type>
        RE        : <rel>
        Joint     : <rel> | </ent>
        Joint+    : <type>
        """
        if task == "boundary":
            return frozenset({self.ent_end_id})
        if task == "ner":
            return frozenset({self.type_id})
        if task == "re":
            return frozenset({self.rel_id})
        if task == "joint":
            return frozenset({self.rel_id, self.ent_end_id})
        if task == "joint+":
            return frozenset({self.type_id})
        return frozenset({self.ent_end_id})  # safety fallback

    # --- Allowed tokens per state ---

    def _allowed_tokens(self, hyp_idx: int) -> FrozenSet[int]:
        """Compute the set of valid next token IDs for one hypothesis."""
        state     = self._states[hyp_idx]   # type: ignore[index]
        task      = self._tasks[self._batch_idx(hyp_idx)]
        batch_idx = self._batch_idx(hyp_idx)

        if state.fsm_state == FSMState.START:
            return frozenset({self.ent_start_id})

        elif state.fsm_state == FSMState.ENT_SPAN:
            copy_next = self._source_copy_next(batch_idx, state.span_tokens)
            exits     = self._ent_span_exits(task)
            valid     = frozenset(copy_next | exits)
            return valid if valid else frozenset({self.eos_id})

        elif state.fsm_state == FSMState.TYPE_LABEL:
            trie = self._ent_type_tries[batch_idx]
            if trie is None:
                return frozenset({self.ent_end_id, self.eos_id})
            return trie.get_valid_next(state.label_prefix)

        elif state.fsm_state == FSMState.REL_LABEL:
            trie = self._rel_tries[batch_idx]
            if trie is None:
                return frozenset({self.tail_id, self.eos_id})
            return trie.get_valid_next(state.label_prefix)

        elif state.fsm_state == FSMState.TAIL_SPAN:
            copy_next = self._source_copy_next(batch_idx, state.span_tokens)
            exits     = frozenset({self.rel_id, self.ent_end_id})
            valid     = frozenset(copy_next | exits)
            return valid if valid else frozenset({self.eos_id})

        elif state.fsm_state == FSMState.INTER:
            valid = {self.ent_start_id, self.eos_id}
            if task != "boundary":
                valid.add(self.null_id)
            return frozenset(valid)

        elif state.fsm_state == FSMState.NULL_LABEL:
            return self._null_tries[batch_idx].get_valid_next(
                state.label_prefix
            )

        elif state.fsm_state == FSMState.END:
            return frozenset({self.eos_id, self.pad_id})

        return frozenset({self.eos_id})  # unreachable — safety valve

    # --- State transition ---

    def _transition(self, hyp_idx: int, token_id: int) -> None:
        """Update the FSM state for *hyp_idx* after emitting *token_id*."""
        state = self._states[hyp_idx]   # type: ignore[index]

        if token_id in (self.eos_id, self.pad_id):
            state.fsm_state = FSMState.END
            return

        if token_id == self.ent_start_id:
            state.fsm_state   = FSMState.ENT_SPAN
            state.span_tokens  = []
            state.label_prefix = []
            return

        if token_id == self.ent_end_id:
            # </ent> closes the current entity block → INTER.
            state.fsm_state   = FSMState.INTER
            state.span_tokens  = []
            state.label_prefix = []
            return

        if token_id == self.type_id:
            if state.fsm_state == FSMState.NULL_LABEL:
                # Inside the null block: <type> is a separator.
                # Flush the current label prefix; stay in NULL_LABEL.
                state.label_prefix = []
                return
            # Outside the null block: transition to TYPE_LABEL.
            state.fsm_state    = FSMState.TYPE_LABEL
            state.label_prefix = []
            return

        if token_id == self.rel_id:
            if state.fsm_state == FSMState.NULL_LABEL:
                # Inside the null block: <rel> is a separator.
                state.label_prefix = []
                return
            # Outside: open a new relation label.
            state.fsm_state    = FSMState.REL_LABEL
            state.label_prefix = []
            state.span_tokens  = []
            return

        if token_id == self.tail_id:
            state.fsm_state   = FSMState.TAIL_SPAN
            state.span_tokens  = []
            return

        if token_id == self.null_id:
            state.fsm_state    = FSMState.NULL_LABEL
            state.label_prefix = []
            return

        # Regular content token — accumulate in the active buffer.
        if state.fsm_state in (FSMState.ENT_SPAN, FSMState.TAIL_SPAN):
            state.span_tokens.append(token_id)
        elif state.fsm_state in (
            FSMState.TYPE_LABEL, FSMState.REL_LABEL, FSMState.NULL_LABEL
        ):
            state.label_prefix.append(token_id)

    # --- LogitsProcessor interface ---

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply FSM constraints to *scores* at the current decoding step.

        ``input_ids`` has shape ``(batch_size × num_beams, seq_len)``;
        ``scores`` has shape ``(batch_size × num_beams, vocab_size)``.

        States are initialised on the first call.  At subsequent calls
        the last emitted token drives each hypothesis's FSM transition,
        then the allowed-token set is computed and all disallowed tokens
        are masked to ``-inf``.
        """
        num_hyps = input_ids.shape[0]

        if self._states is None:
            self._init_states(num_hyps, scores.shape[1], scores.device)

        seq_len = input_ids.shape[1]
        if seq_len > 1:
            last_tokens = input_ids[:, -1].tolist()
            for h in range(num_hyps):
                self._transition(h, last_tokens[h])

        vocab_size = scores.shape[1]
        for h in range(num_hyps):
            allowed = self._allowed_tokens(h)
            # Reuse pre-allocated bool buffer: fill True (blocked), then unblock
            # allowed tokens in one vectorized index op, finally masked_fill_.
            # This avoids a float32 vocab-sized allocation + Python loop per step.
            self._disallowed.fill_(True)
            if allowed:
                valid_ids = [t for t in allowed if t < vocab_size]
                if valid_ids:
                    self._disallowed[
                        torch.tensor(valid_ids, dtype=torch.long, device=scores.device)
                    ] = False
            scores[h].masked_fill_(self._disallowed, float("-inf"))

        return scores


# ---- BUILDER FUNCTION ----


def build_constraint_processor(
    tokenizer: PreTrainedTokenizerBase,
    source_ids: torch.Tensor,
    tokens: AnyTokens = PIPELINE_TOKENS,
    num_beams: int = 1,
) -> ConstraintDecodingProcessor:
    """Construct a :class:`ConstraintDecodingProcessor` ready for use.

    The processor builds its per-item tries internally by extracting the
    prompted labels from each row of ``source_ids``.

    Args:
        tokenizer:  HuggingFace tokeniser with S2G tokens registered.
        source_ids: Encoder input IDs ``(batch, src_len)``.
        tokens:     Token registry matching the model variant.
        num_beams:  Beams per batch item.

    Returns:
        A configured :class:`ConstraintDecodingProcessor`.
    """
    return ConstraintDecodingProcessor(
        tokenizer=tokenizer,
        source_ids=source_ids,
        tokens=tokens,
        num_beams=num_beams,
    )