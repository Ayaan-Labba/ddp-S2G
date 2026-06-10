"""
Structured Extraction Language (SEL) — construction and parsing.
"""
from __future__ import annotations

import random
import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from .special_tokens import AnyTokens, PIPELINE_TOKENS

EntityBlock = Dict[str, Any]
RejectedItem = Dict[str, str]
Triplet = Tuple[str, str, str]


def organize_by_entity(entities: List[Dict], relations: List[Dict]) -> List[EntityBlock]:
    entity_blocks: List[EntityBlock] = []
    offset_to_idx: Dict[Tuple[int, int], int] = {}

    for ent in sorted(entities, key=lambda e: e["offset"][0]):
        key = (int(ent["offset"][0]), int(ent["offset"][1]))
        entity_blocks.append({
            "text": ent["text"],
            "type": ent.get("type"),
            "offset": list(ent["offset"]),
            "relations": []
        })
        offset_to_idx[key] = len(entity_blocks) - 1

    for rel in relations:
        head_key = (int(rel["head"]["offset"][0]), int(rel["head"]["offset"][1]))
        if head_key in offset_to_idx:
            entity_blocks[offset_to_idx[head_key]]["relations"].append({
                "type": rel["type"],
                "tail": rel["tail"]["text"],
                "_tail_offset": int(rel["tail"]["offset"][0]),
            })

    for block in entity_blocks:
        block["relations"].sort(key=lambda r: r["_tail_offset"])
        for rel in block["relations"]:
            del rel["_tail_offset"]

    return entity_blocks


def filter_entity_blocks(entity_blocks: List[EntityBlock], allowed_rel_types: Set[str]) -> List[EntityBlock]:
    return [{
        **block,
        "relations": [r for r in block["relations"] if r["type"] in allowed_rel_types]
    } for block in entity_blocks]


def build_sel(
        entity_blocks: List[EntityBlock], 
        task: str, 
        tok: AnyTokens = PIPELINE_TOKENS, 
        rejected_ent_types: Optional[List[str]] = None, 
        rejected_rel_types: Optional[List[str]] = None, 
        random_sel: bool = False
    ) -> str:
    if task not in {"boundary", "ner", "re", "joint", "joint+"}:
        raise ValueError(f"Unknown task {task!r}.")

    # Memory Optimization: Only copy lists if mutating (shuffling)
    blocks = list(entity_blocks) if random_sel else entity_blocks
    if random_sel: 
        random.shuffle(blocks)

    parts = []
    for ent in blocks:
        rels = list(ent["relations"]) if random_sel else ent["relations"]
        if random_sel: 
            random.shuffle(rels)

        if task == "boundary":
            parts.extend([tok.ent_start, ent['text'], tok.ent_end])
            continue

        if task == "re" and not rels:
            continue

        # Accumulate string components in a list to prevent Python string concatenation overhead
        ent_parts = [tok.ent_start, ent['text']]
        if task in {"ner", "joint+"}:
            ent_parts.extend([tok.type_, ent.get('type', '')])

        for rel in rels:
            ent_parts.extend([tok.rel, rel['type'], tok.tail, rel['tail']])

        ent_parts.append(tok.ent_end)
        parts.append(" ".join(ent_parts))

    if task != "boundary":
        _append_null_block(
            parts, tok, 
            ent_types=(rejected_ent_types or []) if task in {"ner", "joint+"} else [],
            rel_types=(rejected_rel_types or []) if task in {"re", "joint", "joint+"} else [],
            random_sel=random_sel
        )

    return " ".join(parts)


class _State(Enum):
    IDLE = auto()
    ENT_SPAN = auto()
    TYPE_LABEL = auto()
    REL_LABEL = auto()
    TAIL_SPAN = auto()
    NULL_LABEL = auto()


def parse_sel(text: str, tok: AnyTokens = PIPELINE_TOKENS) -> Tuple[List[EntityBlock], List[RejectedItem]]:
    # Compile pattern once per parse (sort by length descending to match longest multi-character tokens first)
    special_tokens = sorted(tok.all_tokens, key=len, reverse=True)
    pattern = re.compile(f"({'|'.join(map(re.escape, special_tokens))})")
    
    # C-level split replaces the inefficient O(N * Tokens) .replace() loops
    tokens = [t.strip() for t in pattern.split(text) if t.strip()]

    state = _State.IDLE
    curr_ent = None
    curr_lbl_parts: List[str] = []
    curr_tail_parts: List[str] = []
    curr_span_parts: List[str] = []
    last_null = None
    
    entities: List[EntityBlock] = []
    rejected: List[RejectedItem] = []

    def flush_tail():
        if state == _State.TAIL_SPAN and curr_tail_parts and curr_lbl_parts and curr_ent:
            curr_ent["relations"].append({
                "type": " ".join(curr_lbl_parts), 
                "tail": " ".join(curr_tail_parts)
            })
            curr_lbl_parts.clear()
            curr_tail_parts.clear()

    def flush_lbl():
        if state == _State.TYPE_LABEL and curr_lbl_parts and curr_ent:
            curr_ent["type"] = " ".join(curr_lbl_parts)
            curr_lbl_parts.clear()
        elif state == _State.NULL_LABEL and curr_lbl_parts:
            label_str = " ".join(curr_lbl_parts)
            if label_str:
                rejected.append({"kind": last_null or "rel", "label": label_str})
            curr_lbl_parts.clear()

    def flush_ent():
        nonlocal curr_ent
        if curr_ent and curr_span_parts:
            span_text = " ".join(curr_span_parts)
            if span_text:
                entities.append({
                    "text": span_text, 
                    "type": curr_ent.get("type"), 
                    "relations": curr_ent.get("relations", [])
                })
        curr_ent = None
        curr_span_parts.clear()

    for t in tokens:
        if t == tok.ent_start:
            flush_tail(); flush_lbl(); flush_ent()
            curr_ent, state = {"type": None, "relations": []}, _State.ENT_SPAN
        elif t == tok.type_:
            flush_tail(); flush_lbl(); curr_lbl_parts.clear()
            if state == _State.NULL_LABEL: last_null = "type"
            else: state = _State.TYPE_LABEL
        elif t == tok.rel:
            flush_tail(); flush_lbl(); curr_lbl_parts.clear()
            if state == _State.NULL_LABEL: last_null = "rel"
            else: state = _State.REL_LABEL
        elif t == tok.tail:
            curr_tail_parts.clear()
            state = _State.TAIL_SPAN
        elif t == tok.ent_end:
            flush_tail(); flush_lbl(); flush_ent()
            state = _State.IDLE
        elif t == tok.null:
            flush_tail(); flush_lbl(); flush_ent()
            curr_lbl_parts.clear()
            last_null = None
            state = _State.NULL_LABEL
        else:
            if state == _State.ENT_SPAN and curr_ent is not None: 
                curr_span_parts.append(t)
            elif state in (_State.TYPE_LABEL, _State.REL_LABEL, _State.NULL_LABEL): 
                curr_lbl_parts.append(t)
            elif state == _State.TAIL_SPAN: 
                curr_tail_parts.append(t)

    flush_tail(); flush_lbl(); flush_ent()
    return _deduplicate_entities(entities), rejected


def extract_triplets(entities: List[EntityBlock]) -> List[Triplet]:
    return [(ent["text"], rel["type"], rel["tail"]) for ent in entities for rel in ent["relations"]]


def _append_null_block(
        parts: List[str], 
        tok: AnyTokens, 
        ent_types: List[str], 
        rel_types: List[str], 
        random_sel: bool
    ) -> None:
    e_types = random.sample(ent_types, len(ent_types)) if random_sel else sorted(ent_types)
    r_types = random.sample(rel_types, len(rel_types)) if random_sel else sorted(rel_types)
    
    null_parts = [f"{tok.type_} {t}" for t in e_types] + [f"{tok.rel} {r}" for r in r_types]
    if null_parts: 
        parts.append(f"{tok.null} {' '.join(null_parts)}")


def _deduplicate_entities(entities: List[EntityBlock]) -> List[EntityBlock]:
    seen, deduped = {}, []
    for ent in entities:
        text_key = ent["text"]
        if text_key in seen:
            deduped[seen[text_key]]["relations"].extend(ent["relations"])
        else:
            seen[text_key] = len(deduped)
            deduped.append(ent)
    return deduped