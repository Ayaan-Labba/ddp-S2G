"""
SSI construction and text augmentation for the S2G encoder input.

Three categories of function are provided:

1. **SSI prefix builders** — produce the ``<type>``- or ``<rel>``-prefixed
   type list that precedes the task delimiter in the encoder input.

2. **Text augmentation** — wrap entity spans inline with structural
   markers to produce the boundary-augmented text (NER encoder) and the
   entity+type-augmented text (RE encoder).

3. **Encoder input builders** — assemble SSI prefix, task delimiter, and
   text segment into the complete string passed to the tokeniser.

All functions are string-level only.  Subword tokenisation happens
downstream in the collator or evaluation loop.

Offset convention
-----------------
Entity spans are represented as half-open token-index intervals
``[start, end)`` over the NLTK ``tokens`` list of the source sentence.
End is exclusive (Python convention).  Overlapping spans are resolved
greedily: the leftmost span is kept; any later span that starts before
the previous span's end is discarded.
"""

from __future__ import annotations

import random as _random
from typing import List, Optional, Tuple

from .special_tokens import (
    AnyTokens,
    JointTokens,
    JOINT_TOKENS,
    PipelineTokens,
    PIPELINE_TOKENS,
)


# ---- SSI PREFIX BUILDERS ----


def build_ner_ssi(
    entity_types: List[str],
    random_order: bool = False,
    tok: AnyTokens = PIPELINE_TOKENS,
) -> str:
    """Build the entity-type SSI prefix used by NER and Joint+.

    Each entity type is preceded by ``<type>``.  Types are sorted
    alphabetically by default; set *random_order* to shuffle instead.

    Args:
        entity_types: Entity-type label strings to include.
        random_order: Shuffle order if ``True``, else sort alphabetically.
        tok:          Token registry (Pipeline or Joint).

    Returns:
        SSI prefix string, e.g.
        ``"<type> city <type> country <type> person"``.
    """
    types = list(entity_types)
    _random.shuffle(types) if random_order else types.sort()
    return " ".join(f"{tok.type_} {t}" for t in types)


def build_rel_ssi(
    rel_types: List[str],
    random_order: bool = False,
    tok: AnyTokens = PIPELINE_TOKENS,
) -> str:
    """Build the relation-type SSI prefix used by RE, Joint, and Joint+.

    Each relation type is preceded by ``<rel>``.  Types are sorted
    alphabetically by default.

    Args:
        rel_types:    Relation-type label strings to include.
        random_order: Shuffle order if ``True``, else sort alphabetically.
        tok:          Token registry (Pipeline or Joint).

    Returns:
        SSI prefix string, e.g.
        ``"<rel> located in <rel> place of birth <rel> president of"``.
    """
    types = list(rel_types)
    _random.shuffle(types) if random_order else types.sort()
    return " ".join(f"{tok.rel} {t}" for t in types)


# ---- TEXT AUGMENTATION ----


def augment_ner_text(
    source_tokens: List[str],
    entity_spans: List[Tuple[int, int]],
    tok: AnyTokens = PIPELINE_TOKENS,
) -> str:
    """Build boundary-augmented text for the NER encoder input.

    Wraps each entity span as ``<ent> SPAN </ent>`` inline in the token
    sequence.  Non-entity tokens pass through unchanged.  Overlapping
    spans are resolved greedily (leftmost wins).

    At training time *entity_spans* comes from gold annotations.
    At inference time it comes from Boundary model predictions, converted
    to token positions via :func:`find_token_span`.

    Args:
        source_tokens: NLTK word tokens for the source sentence.
        entity_spans:  ``(start, end)`` half-open token-index intervals,
                       one per entity span to mark.
        tok:           Token registry.

    Returns:
        Augmented text string ready for tokenisation.
    """
    resolved = _resolve_overlaps(sorted(entity_spans, key=lambda s: s[0]))
    parts: List[str] = []
    cursor = 0
    for start, end in resolved:
        parts.extend(source_tokens[cursor:start])
        parts.append(tok.ent_start)
        parts.extend(source_tokens[start:end])
        parts.append(tok.ent_end)
        cursor = end
    parts.extend(source_tokens[cursor:])
    return " ".join(parts)


def augment_re_text(
    source_tokens: List[str],
    entity_data: List[Tuple[int, int, str]],
    tok: AnyTokens = PIPELINE_TOKENS,
) -> str:
    """Build entity+type-augmented text for the RE encoder input.

    Wraps each entity span as ``<ent> SPAN <type> TYPE </ent>`` inline
    in the token sequence.  Overlapping spans resolved greedily.

    At training time *entity_data* comes from gold NER annotations.
    At inference time it comes from NER model predictions.

    Args:
        source_tokens: NLTK word tokens for the source sentence.
        entity_data:   ``(start, end, type_str)`` tuples — half-open
                       token-index interval plus the entity type label.
        tok:           Token registry.

    Returns:
        Augmented text string ready for tokenisation.
    """
    data = sorted(entity_data, key=lambda e: e[0])
    resolved = _resolve_overlaps([(s, e) for s, e, _ in data])
    resolved_set = set(resolved)
    filtered = [(s, e, t) for s, e, t in data if (s, e) in resolved_set]

    parts: List[str] = []
    cursor = 0
    for start, end, type_str in filtered:
        parts.extend(source_tokens[cursor:start])
        parts.append(tok.ent_start)
        parts.extend(source_tokens[start:end])
        parts.append(tok.type_)
        parts.append(type_str)
        parts.append(tok.ent_end)
        cursor = end
    parts.extend(source_tokens[cursor:])
    return " ".join(parts)


# ---- SPAN LOCATING (INFERENCE) ----


def find_token_span(
    source_tokens: List[str],
    span_text: str,
) -> Optional[Tuple[int, int]]:
    """Find the first occurrence of *span_text* in *source_tokens*.

    Splits *span_text* on whitespace and searches for the first
    contiguous matching run in *source_tokens*.  Returns a half-open
    ``[start, end)`` interval.

    At training time the collator uses gold offsets directly and does
    not call this function.  At evaluation time, use
    :func:`find_all_token_spans` instead so that every occurrence of a
    predicted entity text is marked in the augmented encoder input.

    Args:
        source_tokens: NLTK word tokens for the source sentence.
        span_text:     Whitespace-normalised entity surface text.

    Returns:
        ``(start, end)`` on success; ``None`` if no match is found.
    """
    span_words = span_text.split()
    n = len(span_words)
    if n == 0:
        return None
    for i in range(len(source_tokens) - n + 1):
        if source_tokens[i : i + n] == span_words:
            return (i, i + n)
    return None


def find_all_token_spans(
    source_tokens: List[str],
    span_text: str,
) -> List[Tuple[int, int]]:
    """Find ALL non-overlapping occurrences of *span_text* in *source_tokens*.

    Scans left-to-right; after each match the cursor advances past it so
    occurrences cannot overlap.  This is the correct function to use at
    evaluation time: a predicted entity text may appear multiple times in
    the source sentence, and every occurrence should be marked in the
    augmented NER or RE encoder input.

    Args:
        source_tokens: NLTK word tokens for the source sentence.
        span_text:     Whitespace-normalised entity surface text.

    Returns:
        List of ``(start, end)`` half-open intervals in left-to-right
        order.  Returns ``[]`` if *span_text* is empty or not found.
    """
    span_words = span_text.split()
    n = len(span_words)
    if n == 0:
        return []
    results: List[Tuple[int, int]] = []
    i = 0
    while i <= len(source_tokens) - n:
        if source_tokens[i : i + n] == span_words:
            results.append((i, i + n))
            i += n        # advance past match — non-overlapping
        else:
            i += 1
    return results


# ---- ENCODER INPUT BUILDERS ----


def build_boundary_encoder_input(
    text: str,
    tok: PipelineTokens = PIPELINE_TOKENS,
) -> str:
    """Build the encoder input for the Boundary task.

    Boundary has no SSI.  The encoder input is the task delimiter
    immediately followed by the raw text.

    Format: ``<bound> raw text``

    Args:
        text: Raw input sentence.
        tok:  Pipeline token registry.

    Returns:
        Encoder input string.
    """
    return f"{tok.bound} {text}"


def build_ner_encoder_input(
    entity_types: List[str],
    source_tokens: List[str],
    entity_spans: List[Tuple[int, int]],
    random_order: bool = False,
    tok: PipelineTokens = PIPELINE_TOKENS,
) -> str:
    """Build the encoder input for the NER task.

    Format: ``<type> T₁ … <type> Tₙ <ner> <ent> span₁ </ent> … raw``

    The text segment is the boundary-augmented text produced by
    :func:`augment_ner_text`.  At training time *entity_spans* are gold
    spans; at inference time they are Boundary model predictions.

    Args:
        entity_types:  Entity-type strings for the SSI.
        source_tokens: NLTK word tokens for the source sentence.
        entity_spans:  ``(start, end)`` token intervals marking entity
                       boundaries in the text segment.
        random_order:  Shuffle SSI type order if ``True``.
        tok:           Pipeline token registry.

    Returns:
        Full encoder input string.
    """
    ssi = build_ner_ssi(entity_types, random_order=random_order, tok=tok)
    augmented = augment_ner_text(source_tokens, entity_spans, tok=tok)
    return f"{ssi} {tok.ner} {augmented}"


def build_re_encoder_input(
    rel_types: List[str],
    source_tokens: List[str],
    entity_data: List[Tuple[int, int, str]],
    random_order: bool = False,
    tok: PipelineTokens = PIPELINE_TOKENS,
) -> str:
    """Build the encoder input for the RE task.

    Format: ``<rel> R₁ … <rel> Rₙ <re> <ent> span <type> TYPE </ent> …``

    The text segment is the entity+type-augmented text produced by
    :func:`augment_re_text`.  At training time *entity_data* comes from
    gold NER annotations; at inference time from NER model predictions.

    Args:
        rel_types:     Relation-type strings for the SSI.
        source_tokens: NLTK word tokens for the source sentence.
        entity_data:   ``(start, end, type_str)`` tuples for each entity.
        random_order:  Shuffle SSI type order if ``True``.
        tok:           Pipeline token registry.

    Returns:
        Full encoder input string.
    """
    ssi = build_rel_ssi(rel_types, random_order=random_order, tok=tok)
    augmented = augment_re_text(source_tokens, entity_data, tok=tok)
    return f"{ssi} {tok.re} {augmented}"


def build_joint_encoder_input(
    rel_types: List[str],
    text: str,
    random_order: bool = False,
    tok: JointTokens = JOINT_TOKENS,
) -> str:
    """Build the encoder input for the Joint task.

    Format: ``<rel> R₁ … <rel> Rₙ <joint> raw text``

    The text segment is raw and unaugmented.

    Args:
        rel_types:    Relation-type strings for the SSI.
        text:         Raw input sentence.
        random_order: Shuffle SSI type order if ``True``.
        tok:          Joint token registry.

    Returns:
        Full encoder input string.
    """
    ssi = build_rel_ssi(rel_types, random_order=random_order, tok=tok)
    return f"{ssi} {tok.joint} {text}"


def build_joint_plus_encoder_input(
    entity_types: List[str],
    rel_types: List[str],
    text: str,
    random_order: bool = False,
    tok: JointTokens = JOINT_TOKENS,
) -> str:
    """Build the encoder input for the Joint+ task.

    Format: ``<type> T₁ … <rel> R₁ … <joint+> raw text``

    The entity-type SSI always precedes the relation-type SSI (per spec).
    *random_order* shuffles within each group independently; entity types
    always come before relation types regardless of this setting.

    Args:
        entity_types: Entity-type strings for the SSI.
        rel_types:    Relation-type strings for the SSI.
        text:         Raw input sentence.
        random_order: Shuffle order within each type group if ``True``.
        tok:          Joint token registry.

    Returns:
        Full encoder input string.
    """
    ent_ssi = build_ner_ssi(entity_types, random_order=random_order, tok=tok)
    rel_ssi = build_rel_ssi(rel_types, random_order=random_order, tok=tok)
    # Filter out empty halves so the delimiter is not double-spaced.
    prefix = " ".join(p for p in (ent_ssi, rel_ssi) if p)
    return f"{prefix} {tok.joint_plus} {text}"


# ---- HELPERS ----


def _resolve_overlaps(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return non-overlapping spans from a sorted list using greedy-first selection.

    Assumes *spans* is already sorted ascending by start position.
    Any span whose start falls before the end of the last accepted span
    is discarded.

    Args:
        spans: ``(start, end)`` pairs sorted by start.

    Returns:
        Subset of *spans* with no overlaps, in the same order.
    """
    result: List[Tuple[int, int]] = []
    last_end = -1
    for start, end in spans:
        if start >= last_end:
            result.append((start, end))
            last_end = end
    return result