"""
Structured Extraction Language (SEL) — construction and parsing.

Provides both directions of the SEL conversion for all five task settings:
Boundary, NER, RE, Joint, and Joint+.

1. **Construction** (:func:`build_sel`) — given entity blocks, rejected
   types, and a task name, produce the flat target string that the
   decoder is trained to generate.

2. **Parsing** (:func:`parse_sel`) — given a flat decoded string, recover
   structured entity blocks and a rejected-type list.

SEL grammars
------------
All grammars use ``SPAN ::= token+`` (exact substring of source) and
``LABEL ::= token+`` (natural-language type name).

Boundary::

    SEL    ::= ENT+
    ENT    ::= <ent> SPAN </ent>

NER::

    SEL    ::= ENT+ REJECT?
    ENT    ::= <ent> SPAN <type> LABEL </ent>
    REJECT ::= <null> (<type> LABEL)+

RE::

    SEL    ::= ENT* REJECT?
    ENT    ::= <ent> SPAN REL+ </ent>          # tail-only entities omitted
    REL    ::= <rel> LABEL <tail> SPAN
    REJECT ::= <null> (<rel> LABEL)+

Joint::

    SEL    ::= ENT+ REJECT?
    ENT    ::= <ent> SPAN REL+ </ent>
             | <ent> SPAN </ent>               # all entities included
    REL    ::= <rel> LABEL <tail> SPAN
    REJECT ::= <null> (<rel> LABEL)+

Joint+::

    SEL    ::= ENT+ REJECT?
    ENT    ::= <ent> SPAN <type> LABEL REL+ </ent>
             | <ent> SPAN <type> LABEL </ent>
    REL    ::= <rel> LABEL <tail> SPAN
    REJECT ::= <null> (<type> LABEL)+ (<rel> LABEL)+

Parser design
-------------
Single-pass left-to-right scan.  The parser is task-agnostic: it handles
all five grammars uniformly.  The key disambiguation rule is that
``<type>`` and ``<rel>`` tokens inside the ``<null>`` block (NULL_LABEL
state) do **not** change the parser state — they only update
``last_null_token`` and flush the accumulating rejected label.  In all
other states, ``<type>`` transitions to TYPE_LABEL and ``<rel>``
transitions to REL_LABEL as normal.
"""

from __future__ import annotations

import random as _random
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from .special_tokens import AnyTokens, PIPELINE_TOKENS

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EntityBlock = Dict[str, Any]
# {
#   "text":      str           — surface text of the entity span
#   "type":      Optional[str] — entity type label (None for Boundary/RE/Joint)
#   "relations": List[{"type": str, "tail": str}]
#   "offset":    List[int]     — token offsets; present after organize_by_entity,
#                                stripped in parse_sel output
# }

RejectedItem = Dict[str, str]
# {"kind": "type" | "rel", "label": str}

Triplet = Tuple[str, str, str]  # (head_text, relation_type, tail_text)


# ---- SEL CONSTRUCTION ----


def organize_by_entity(
    entities: List[Dict],
    relations: List[Dict],
) -> List[EntityBlock]:
    """Group flat entity and relation lists into entity-centric blocks.

    Entities are sorted by their start-token offset.  Relations within
    each block are sorted by the tail entity's start-token offset.
    Every entity in *entities* receives a block regardless of whether it
    appears as a relation head or tail; task-specific filtering (e.g.
    excluding tail-only entities for RE) is handled in :func:`build_sel`.

    Args:
        entities:  List of entity dicts.  Required keys: ``text``,
                   ``offset`` (``[start, end)``).  Optional: ``type``.
        relations: List of relation dicts.  Required keys: ``head``
                   (entity dict), ``tail`` (entity dict), ``type`` (str).

    Returns:
        List of :data:`EntityBlock` dicts ordered by start offset.

    Example::

        >>> organize_by_entity(
        ...     [{"text": "Obama", "offset": [0, 1], "type": "person"},
        ...      {"text": "Hawaii", "offset": [4, 5], "type": "location"}],
        ...     [{"head": {"offset": [0, 1]}, "tail": {"text": "Hawaii",
        ...       "offset": [4, 5]}, "type": "born in"}],
        ... )
        [{"text": "Obama", "type": "person", "offset": [0, 1],
          "relations": [{"type": "born in", "tail": "Hawaii"}]},
         {"text": "Hawaii", "type": "location", "offset": [4, 5],
          "relations": []}]
    """
    sorted_entities = sorted(entities, key=lambda e: e["offset"][0])

    entity_blocks: List[EntityBlock] = []
    offset_to_idx: Dict[Tuple[int, int], int] = {}

    for ent in sorted_entities:
        key = (int(ent["offset"][0]), int(ent["offset"][1]))
        block: EntityBlock = {
            "text":      ent["text"],
            "type":      ent.get("type"),   # None if absent
            "offset":    list(ent["offset"]),
            "relations": [],
        }
        entity_blocks.append(block)
        offset_to_idx[key] = len(entity_blocks) - 1

    for rel in relations:
        head_key = (
            int(rel["head"]["offset"][0]),
            int(rel["head"]["offset"][1]),
        )
        if head_key not in offset_to_idx:
            continue  # orphaned relation — skip defensively
        idx = offset_to_idx[head_key]
        entity_blocks[idx]["relations"].append(
            {
                "type":          rel["type"],
                "tail":          rel["tail"]["text"],
                "_tail_offset":  int(rel["tail"]["offset"][0]),
            }
        )

    for block in entity_blocks:
        block["relations"].sort(key=lambda r: r["_tail_offset"])
        for rel in block["relations"]:
            del rel["_tail_offset"]

    return entity_blocks


def filter_entity_blocks(
    entity_blocks: List[EntityBlock],
    allowed_rel_types: Set[str],
) -> List[EntityBlock]:
    """Remove relation children whose type is not in *allowed_rel_types*.

    Entity blocks themselves are never removed — only their ``<rel>``
    children are pruned.  Used when positive relation types are withheld
    from the SSI (Bernoulli-mode pre-training, Experiment 3).  Not
    called in Experiment 1 (budget mode always includes all gold
    positives).

    Args:
        entity_blocks:     Output of :func:`organize_by_entity`.
        allowed_rel_types: Set of relation-type strings to retain.

    Returns:
        New list of entity blocks with filtered relation lists.
    """
    filtered: List[EntityBlock] = []
    for block in entity_blocks:
        new_block: EntityBlock = {
            "text":      block["text"],
            "type":      block.get("type"),
            "relations": [
                rel for rel in block["relations"]
                if rel["type"] in allowed_rel_types
            ],
        }
        if "offset" in block:
            new_block["offset"] = block["offset"]
        filtered.append(new_block)
    return filtered


def build_sel(
    entity_blocks: List[EntityBlock],
    task: str,
    tok: AnyTokens = PIPELINE_TOKENS,
    rejected_ent_types: Optional[List[str]] = None,
    rejected_rel_types: Optional[List[str]] = None,
    random_sel: bool = False,
) -> str:
    """Build the flat SEL target string for *task*.

    Implements the task-specific grammar defined in the module docstring.
    Key behavioural differences across tasks:

    - **Boundary**: entity span only; no type, no relations.
    - **NER**: entity span + type label; no relations.
    - **RE**: only entities with ≥ 1 outgoing relation receive a block.
    - **Joint**: all entities receive a block; relations are optional.
    - **Joint+**: all entities with type label; relations are optional.

    The rejection block (``<null> …``) is omitted when no types are
    absent from the instance.  Boundary never has a rejection block.

    Args:
        entity_blocks:      Output of :func:`organize_by_entity`.
        task:               One of ``"boundary"``, ``"ner"``, ``"re"``,
                            ``"joint"``, ``"joint+"``.
        tok:                Token registry (Pipeline or Joint).
        rejected_ent_types: Entity types absent from this instance
                            (used for NER and Joint+ null blocks).
        rejected_rel_types: Relation types absent from this instance
                            (used for RE, Joint, and Joint+ null blocks).
        random_sel:         Shuffle entity and relation order if ``True``.
                            Rejected-type order is also shuffled.

    Returns:
        Flat SEL string ready for tokenisation as the decoder target.

    Examples::

        >>> # Boundary
        >>> build_sel(blocks, "boundary", tok)
        '<ent> Barack Obama </ent> <ent> Honolulu </ent>'

        >>> # NER with rejection
        >>> build_sel(blocks, "ner", tok,
        ...           rejected_ent_types=["organization", "artifact"])
        '<ent> Barack Obama <type> person </ent> ... <null> <type> artifact <type> organization'

        >>> # RE — tail-only entity (United States) has no block
        >>> build_sel(blocks, "re", tok,
        ...           rejected_rel_types=["founded", "killed"])
        '<ent> Barack Obama <rel> place of birth <tail> Honolulu ... </ent> <null> <rel> founded <rel> killed'
    """
    if task not in ("boundary", "ner", "re", "joint", "joint+"):
        raise ValueError(
            f"Unknown task {task!r}. "
            "Expected one of: 'boundary', 'ner', 're', 'joint', 'joint+'."
        )

    rejected_ent_types = list(rejected_ent_types or [])
    rejected_rel_types = list(rejected_rel_types or [])

    blocks = list(entity_blocks)
    if random_sel:
        _random.shuffle(blocks)
    else:
        # Preserve offset ordering from organize_by_entity (already sorted).
        pass

    parts: List[str] = []

    # --- Boundary ---
    if task == "boundary":
        for ent in blocks:
            parts.append(f"{tok.ent_start} {ent['text']} {tok.ent_end}")

    # --- NER ---
    elif task == "ner":
        for ent in blocks:
            parts.append(
                f"{tok.ent_start} {ent['text']} {tok.type_} {ent.get('type') or ''} {tok.ent_end}"
            )
        _append_null_block(
            parts, tok,
            ent_types=rejected_ent_types,
            rel_types=[],
            random_sel=random_sel,
        )

    # --- RE ---
    elif task == "re":
        for ent in blocks:
            rels = ent["relations"]
            if not rels:
                # Tail-only or relation-less entities: no block in RE.
                continue
            if random_sel:
                rels = list(rels)
                _random.shuffle(rels)
            ent_str = tok.ent_start + " " + ent["text"]
            for rel in rels:
                ent_str += (
                    f" {tok.rel} {rel['type']} {tok.tail} {rel['tail']}"
                )
            ent_str += f" {tok.ent_end}"
            parts.append(ent_str)
        _append_null_block(
            parts, tok,
            ent_types=[],
            rel_types=rejected_rel_types,
            random_sel=random_sel,
        )

    # --- Joint ---
    elif task == "joint":
        for ent in blocks:
            rels = list(ent["relations"])
            if random_sel:
                _random.shuffle(rels)
            ent_str = tok.ent_start + " " + ent["text"]
            for rel in rels:
                ent_str += (
                    f" {tok.rel} {rel['type']} {tok.tail} {rel['tail']}"
                )
            ent_str += f" {tok.ent_end}"
            parts.append(ent_str)
        _append_null_block(
            parts, tok,
            ent_types=[],
            rel_types=rejected_rel_types,
            random_sel=random_sel,
        )

    # --- Joint+ ---
    elif task == "joint+":
        for ent in blocks:
            rels = list(ent["relations"])
            if random_sel:
                _random.shuffle(rels)
            ent_str = (
                f"{tok.ent_start} {ent['text']} {tok.type_} {ent.get('type') or ''}"
            )
            for rel in rels:
                ent_str += (
                    f" {tok.rel} {rel['type']} {tok.tail} {rel['tail']}"
                )
            ent_str += f" {tok.ent_end}"
            parts.append(ent_str)
        _append_null_block(
            parts, tok,
            ent_types=rejected_ent_types,
            rel_types=rejected_rel_types,
            random_sel=random_sel,
        )

    return " ".join(parts)


# ---- SEL PARSING ----


class _State(Enum):
    """Parser FSM states."""
    IDLE        = auto()
    ENT_SPAN    = auto()
    TYPE_LABEL  = auto()
    REL_LABEL   = auto()
    TAIL_SPAN   = auto()
    NULL_LABEL  = auto()


def parse_sel(
    text: str,
    tok: AnyTokens = PIPELINE_TOKENS,
) -> Tuple[List[EntityBlock], List[RejectedItem]]:
    """Parse a generated SEL string into entity blocks and rejected labels.

    Implements a single-pass left-to-right scan over the decoded token
    sequence.  The algorithm is task-agnostic: all five SEL grammars are
    handled uniformly through the state machine described in the module
    docstring.

    The critical design choice: inside the ``<null>`` block (NULL_LABEL
    state), ``<type>`` and ``<rel>`` tokens do **not** change the parser
    state.  They only flush the accumulating label into ``rejected`` and
    update ``last_null_token`` to record whether the next label is an
    entity type (``kind="type"``) or a relation type (``kind="rel"``).
    This avoids the ambiguity that would arise if ``<type>`` naively
    transitioned to TYPE_LABEL while ``current_entity`` is ``None``.

    Args:
        text: Decoded SEL string from
              ``tokenizer.decode(output_ids, skip_special_tokens=False)``.
        tok:  Token registry matching the model that produced *text*.

    Returns:
        ``(entities, rejected)`` where *entities* is a list of
        :data:`EntityBlock` dicts (each has ``"text"``, ``"type"``,
        ``"relations"``) and *rejected* is a list of
        :data:`RejectedItem` dicts (each has ``"kind"`` and ``"label"``).
    """
    special_set: Set[str] = set(tok.all_tokens)

    # Pad special tokens with spaces so they survive whitespace splitting.
    padded = text
    for token in special_set:
        padded = padded.replace(token, f" {token} ")

    words = padded.strip().split()
    tokens: List[str] = _segment_words(words, special_set)

    # ---- Parser state ----
    state = _State.IDLE
    current_entity: Optional[Dict] = None   # {span, type, relations}
    current_label:  Optional[str]  = None   # TYPE_LABEL / REL_LABEL / NULL_LABEL buffer
    current_tail:   Optional[str]  = None   # TAIL_SPAN buffer
    last_null_token: Optional[str] = None   # "type" | "rel", used in NULL_LABEL

    entities:  List[EntityBlock]  = []
    rejected:  List[RejectedItem] = []

    # --- Flush helpers ---

    def flush_tail() -> None:
        """Commit a completed (rel_label, tail_span) pair to current_entity."""
        nonlocal current_label, current_tail
        if (
            state == _State.TAIL_SPAN
            and current_tail is not None
            and current_label is not None
            and current_entity is not None
        ):
            current_entity["relations"].append(
                {
                    "type": current_label.strip(),
                    "tail": current_tail.strip(),
                }
            )
            current_label = None
            current_tail  = None

    def flush_type_or_rel_label() -> None:
        """Commit an in-progress TYPE_LABEL or NULL_LABEL to their targets.

        REL_LABEL is a deliberate no-op: an in-progress relation label is
        only committed via flush_tail (which requires a <tail> to have
        been seen).  An incomplete <rel> LABEL without a <tail> is
        silently dropped, covering malformed decoder output gracefully.
        """
        nonlocal current_label
        if state == _State.TYPE_LABEL and current_label is not None:
            if current_entity is not None:
                current_entity["type"] = current_label.strip()
            current_label = None
        elif (
            state == _State.NULL_LABEL
            and current_label is not None
            and current_label.strip()
        ):
            kind = last_null_token if last_null_token else "rel"
            rejected.append({"kind": kind, "label": current_label.strip()})
            current_label = None

    def flush_entity() -> None:
        """Commit current_entity to the entities list if non-empty."""
        nonlocal current_entity
        if current_entity is not None and current_entity["span"].strip():
            entities.append(
                {
                    "text":      current_entity["span"].strip(),
                    "type":      current_entity["type"],
                    "relations": current_entity["relations"],
                }
            )
        current_entity = None

    # --- Main scan ---

    for token in tokens:

        if token == tok.ent_start:
            flush_tail()
            flush_type_or_rel_label()
            flush_entity()
            current_entity = {"span": "", "type": None, "relations": []}
            state = _State.ENT_SPAN

        elif token == tok.type_:
            if state == _State.NULL_LABEL:
                # Inside null block: <type> is a kind-marker / separator.
                # Flush the accumulating label first, then record the kind.
                flush_type_or_rel_label()
                last_null_token = "type"
                current_label = ""
                # state stays NULL_LABEL
            else:
                flush_tail()
                flush_type_or_rel_label()
                current_label = ""
                state = _State.TYPE_LABEL

        elif token == tok.rel:
            if state == _State.NULL_LABEL:
                # Inside null block: <rel> is a kind-marker / separator.
                flush_type_or_rel_label()
                last_null_token = "rel"
                current_label = ""
                # state stays NULL_LABEL
            else:
                flush_tail()
                flush_type_or_rel_label()
                current_label = ""
                state = _State.REL_LABEL

        elif token == tok.tail:
            # <tail> opens the tail-span buffer; current_label (the
            # relation-type label) is preserved for flush_tail later.
            current_tail = ""
            state = _State.TAIL_SPAN

        elif token == tok.ent_end:
            # </ent> is the authoritative entity-block closer.
            flush_tail()
            flush_type_or_rel_label()
            flush_entity()
            state = _State.IDLE

        elif token == tok.null:
            flush_tail()
            flush_type_or_rel_label()
            flush_entity()
            current_label    = ""
            last_null_token  = None
            state = _State.NULL_LABEL

        else:
            # Content token: append to whichever buffer is active.
            if state == _State.ENT_SPAN and current_entity is not None:
                current_entity["span"] = _str_append(
                    current_entity["span"], token
                )
            elif state in (_State.TYPE_LABEL, _State.REL_LABEL, _State.NULL_LABEL):
                if current_label is not None:
                    current_label = _str_append(current_label, token)
            elif state == _State.TAIL_SPAN and current_tail is not None:
                current_tail = _str_append(current_tail, token)

    # ---- EOS: flush remaining state ----
    flush_tail()
    flush_type_or_rel_label()
    flush_entity()

    # ---- Post-processing ----
    deduped = _deduplicate_entities(entities)
    return deduped, rejected


# ---- TRIPLET EXTRACTION ----


def extract_triplets(entities: List[EntityBlock]) -> List[Triplet]:
    """Flatten entity blocks into ``(head, relation_type, tail)`` triplets.

    Entity type information is not included; use the ``"type"`` field on
    each entity block directly for NER metrics or RE strict evaluation.

    Args:
        entities: Parsed entity blocks from :func:`parse_sel`.

    Returns:
        List of ``(head_text, relation_type, tail_text)`` tuples.
    """
    triplets: List[Triplet] = []
    for ent in entities:
        for rel in ent["relations"]:
            triplets.append((ent["text"], rel["type"], rel["tail"]))
    return triplets


# ---- HELPERS ----


def _append_null_block(
    parts: List[str],
    tok: AnyTokens,
    ent_types: List[str],
    rel_types: List[str],
    random_sel: bool,
) -> None:
    """Append a grouped ``<null>`` rejection block to *parts* if non-empty.

    Joint+ receives both entity types and relation types (entity types
    first).  All other tasks receive at most one non-empty list.  The
    block is omitted entirely when both lists are empty.

    Entity types and relation types are sorted alphabetically unless
    *random_sel* is ``True``.
    """
    e_types = list(ent_types)
    r_types = list(rel_types)

    if random_sel:
        _random.shuffle(e_types)
        _random.shuffle(r_types)
    else:
        e_types.sort()
        r_types.sort()

    null_parts: List[str] = []
    for t in e_types:
        null_parts.append(f"{tok.type_} {t}")
    for r in r_types:
        null_parts.append(f"{tok.rel} {r}")

    if null_parts:
        parts.append(f"{tok.null} {' '.join(null_parts)}")


def _segment_words(words: List[str], special_set: Set[str]) -> List[str]:
    """Merge consecutive non-special words into single span strings.

    Special tokens remain as individual elements.  This lets the parser
    operate on pre-merged spans rather than accumulating single words.

    Example::

        ["<ent>", "Barack", "Obama", "<rel>", "place", "of", "birth"]
        → ["<ent>", "Barack Obama", "<rel>", "place of birth"]
    """
    result: List[str] = []
    buffer: List[str] = []

    for w in words:
        if w in special_set:
            if buffer:
                result.append(" ".join(buffer))
                buffer = []
            result.append(w)
        else:
            buffer.append(w)

    if buffer:
        result.append(" ".join(buffer))

    return result


def _str_append(current: str, token: str) -> str:
    """Append *token* to *current* with a single separating space."""
    return f"{current} {token}" if current else token


def _deduplicate_entities(entities: List[EntityBlock]) -> List[EntityBlock]:
    """Merge duplicate entity blocks (same surface text) into the first.

    Relations from later duplicates are appended in order of appearance.
    The type of the first occurrence is kept.
    """
    seen: Dict[str, int] = {}
    deduped: List[EntityBlock] = []

    for ent in entities:
        key = ent["text"]
        if key in seen:
            deduped[seen[key]]["relations"].extend(ent["relations"])
        else:
            seen[key] = len(deduped)
            deduped.append(ent)

    return deduped