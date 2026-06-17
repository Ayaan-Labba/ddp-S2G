"""
Structured Extraction Language (SEL) — construction and parsing.
"""
from __future__ import annotations

import random
import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from .special_tokens import AnyTokens, S2GTokens

EntityBlock = Dict[str, Any]
RejectedItem = Dict[str, str]
Triplet = Tuple[str, str, str]


def organize_by_entity(entities: List[Dict], relations: List[Dict]) -> List[EntityBlock]:
    entity_blocks: List[EntityBlock] = []
    offset_to_idx: Dict[Tuple[int, int], int] = {}
    offset_to_type: Dict[Tuple[int, int], str] = {}

    for ent in sorted(entities, key=lambda e: e["offset"][0]):
        key = (int(ent["offset"][0]), int(ent["offset"][1]))
        entity_blocks.append({
            "text": ent["text"],
            "type": ent.get("type"),
            "offset": list(ent["offset"]),
            "relations": []
        })
        offset_to_idx[key] = len(entity_blocks) - 1
        offset_to_type[key] = ent.get("type", "")

    for rel in relations:
        head_key = (int(rel["head"]["offset"][0]), int(rel["head"]["offset"][1]))
        tail_key = (int(rel["tail"]["offset"][0]), int(rel["tail"]["offset"][1]))
        if head_key in offset_to_idx:
            tail_type = offset_to_type.get(tail_key) or rel["tail"].get("type") or ""
            entity_blocks[offset_to_idx[head_key]]["relations"].append({
                "type": rel["type"],
                "tail": rel["tail"]["text"],
                "tail_type": tail_type,
                "_tail_offset": int(rel["tail"]["offset"][0]),
            })

    for block in entity_blocks:
        block["relations"].sort(key=lambda r: r["_tail_offset"])
        for rel in block["relations"]:
            del rel["_tail_offset"]

    return entity_blocks


def filter_entity_blocks(entity_blocks: List[EntityBlock], allowed_rel_types: Set[str], allowed_ent_types: Optional[Set[str]] = None) -> List[EntityBlock]:
    return [{
        **block,
        "relations": [
            r for r in block["relations"] 
            if r["type"] in allowed_rel_types and (allowed_ent_types is None or r.get("tail_type") in allowed_ent_types)
        ]
    } for block in entity_blocks]


def build_sel(
        entity_blocks: List[EntityBlock], 
        task: str, 
        tok: AnyTokens, 
        rejected_ent_types: Optional[List[str]] = None, 
        rejected_rel_types: Optional[List[str]] = None, 
        random_sel: bool = False,
        use_rejection: bool = False,
        use_nesting: bool = True,
        rel_map: Optional[Dict[str, str]] = None
    ) -> str:
    if task not in {"re", "boundary_re", "boundary_joint", "joint"}:
        raise ValueError(f"Unknown task {task!r}.")
    blocks = list(entity_blocks) if random_sel else entity_blocks
    if random_sel: 
        random.shuffle(blocks)

    if task in {"re", "boundary_re"}:
        rel_map = rel_map or {}
        analysis_parts = []
        extract_parts = []

        for ent in blocks:
            rels = list(ent["relations"]) if random_sel else ent["relations"]
            if random_sel: 
                random.shuffle(rels)
            if not rels:
                continue

            # --- SUMMARY ---
            head_text = ent['text']
            ent_analysis = []
            if task == "re":
                head_type = ent.get('type') or ''
                ent_analysis.append(f"{head_text} [{head_type}]")
                rel_str_list = []
                for rel in rels:
                    rel_expanded = rel_map.get(rel['type'], rel['type'])
                    tail_text = rel['tail']
                    tail_type = rel.get('tail_type') or ''
                    rel_str_list.append(f"{rel_expanded} {tail_text} [{tail_type}]")
            else:  # boundary_re
                ent_analysis.append(head_text)
                rel_str_list = []
                for rel in rels:
                    rel_expanded = rel_map.get(rel['type'], rel['type'])
                    tail_text = rel['tail']
                    rel_str_list.append(f"{rel_expanded} {tail_text}")
            
            ent_analysis.append(" ; ".join(rel_str_list))
            analysis_parts.append(" ".join(ent_analysis))

            # --- TRIPLETS ---
            for i, rel in enumerate(rels):
                if task == "re":
                    if i == 0 or not use_nesting:
                        extract_parts.extend([tok.head, head_text, tok.type_, ent.get('type') or '', tok.rel, rel['type'], tok.tail, rel['tail'], tok.type_, rel.get('tail_type') or ''])
                    else:
                        extract_parts.extend([tok.nest, rel['type'], tok.tail, rel['tail'], tok.type_, rel.get('tail_type') or ''])
                else:  # boundary_re
                    if i == 0 or not use_nesting:
                        extract_parts.extend([tok.head, head_text, tok.rel, rel['type'], tok.tail, rel['tail']])
                    else:
                        extract_parts.extend([tok.nest, rel['type'], tok.tail, rel['tail']])

        analysis_str = " . ".join(analysis_parts)
        extract_str = " ".join(extract_parts)

        if use_rejection and rejected_rel_types:
            r_types = random.sample(rejected_rel_types, len(rejected_rel_types)) if random_sel else sorted(rejected_rel_types)
            missing_str = " , ".join(f"'{r}'" for r in r_types)
        else:
            missing_str = ""

        parts = []
        parts.append(f"SUMMARY: {analysis_str}" if analysis_str else "SUMMARY:")
        parts.append(f"TRIPLETS: {extract_str}" if extract_str else "TRIPLETS:")
        if use_rejection:
            parts.append(f"MISSING: {missing_str}" if missing_str else "MISSING:")
        return " ".join(parts)

    if task in {"joint", "boundary_joint"}:
        parts = []
        ent_parts = []
        for ent in blocks:
            if task == "joint":
                ent_parts.extend([tok.ent_start, ent['text'], tok.type_, ent.get('type') or ''])
            else:  # boundary_joint
                ent_parts.extend([tok.ent_start, ent['text']])
        if ent_parts:
            parts.append(" ".join(ent_parts))

        triplet_parts = []
        for ent in blocks:
            rels = list(ent["relations"]) if random_sel else ent["relations"]
            if random_sel: 
                random.shuffle(rels)
            if not rels:
                continue

            ent_triplet = []
            for i, rel in enumerate(rels):
                if i == 0 or not use_nesting:
                    ent_triplet.extend([tok.head, ent['text'], tok.rel, rel['type'], tok.tail, rel['tail']])
                else:
                    ent_triplet.extend([tok.nest, rel['type'], tok.tail, rel['tail']])
            triplet_parts.append(" ".join(ent_triplet))

        if triplet_parts:
            parts.append(" ".join(triplet_parts))

        if use_rejection:
            _append_null_block(
                parts, tok, 
                ent_types=(rejected_ent_types or []) if task == "joint" else [],
                rel_types=(rejected_rel_types or []) if task in {"boundary_joint", "joint"} else [],
                random_sel=random_sel
            )
        return " ".join(parts)

    return ""


class _State(Enum):
    IDLE = auto()
    ENT_SPAN = auto()
    TYPE_LABEL = auto()
    REL_LABEL = auto()
    TAIL_SPAN = auto()
    TAIL_TYPE_LABEL = auto()
    NULL_LABEL = auto()


def parse_sel(text: str, tok: AnyTokens) -> Tuple[List[EntityBlock], List[RejectedItem]]:
    special_tokens = sorted(tok.all_tokens, key=len, reverse=True)
    pattern = re.compile(f"({'|'.join(map(re.escape, special_tokens))})")
    tokens = [t.strip() for t in pattern.split(text) if t.strip()]

    if tok.variant in {"joint", "boundary_joint"}:
        state = "IDLE"
        current_ent_text = []
        current_ent_type = []
        current_head_parts = []
        current_rel_parts = []
        current_tail_parts = []
        current_lbl_parts = []
        last_null = None

        entity_list = []
        entity_dict = {}
        rejected = []

        def flush_current_state():
            nonlocal state, last_null
            if state in {"ENT_TEXT", "ENT_TYPE"}:
                if current_ent_text:
                    ent_text = " ".join(current_ent_text)
                    ent_type = " ".join(current_ent_type) if current_ent_type else None
                    if ent_text not in entity_dict:
                        block = {"text": ent_text, "type": ent_type, "relations": []}
                        entity_list.append(block)
                        entity_dict[ent_text] = block
                    current_ent_text.clear()
                    current_ent_type.clear()
            elif state == "TAIL":
                if current_head_parts and current_rel_parts and current_tail_parts:
                    head_text = " ".join(current_head_parts)
                    rel_type = " ".join(current_rel_parts)
                    tail_text = " ".join(current_tail_parts)
                    
                    if head_text not in entity_dict:
                        block = {"text": head_text, "type": None, "relations": []}
                        entity_list.append(block)
                        entity_dict[head_text] = block
                    
                    tail_type = entity_dict[tail_text]["type"] if tail_text in entity_dict else None
                    
                    entity_dict[head_text]["relations"].append({
                        "type": rel_type,
                        "tail": tail_text,
                        "tail_type": tail_type
                    })
                    current_rel_parts.clear()
                    current_tail_parts.clear()
            elif state == "NULL":
                if current_lbl_parts:
                    label_str = " ".join(current_lbl_parts)
                    if label_str:
                        rejected.append({"kind": last_null or "rel", "label": label_str})
                    current_lbl_parts.clear()

        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == tok.ent_start:
                flush_current_state()
                state = "ENT_TEXT"
                current_ent_text.clear()
                current_ent_type.clear()
                i += 1
            elif t == getattr(tok, "head", None):
                flush_current_state()
                state = "HEAD"
                current_head_parts.clear()
                i += 1
            elif t == tok.type_:
                if state == "ENT_TEXT":
                    state = "ENT_TYPE"
                    current_ent_type.clear()
                elif state == "NULL":
                    flush_current_state()
                    last_null = "type"
                else:
                    pass
                i += 1
            elif t == tok.rel:
                if state == "NULL":
                    flush_current_state()
                    last_null = "rel"
                else:
                    flush_current_state()
                    state = "REL"
                    current_rel_parts.clear()
                i += 1
            elif t == tok.tail:
                state = "TAIL"
                current_tail_parts.clear()
                i += 1
            elif t == getattr(tok, "nest", None):
                flush_current_state()
                state = "REL"
                current_rel_parts.clear()
                i += 1
            elif t == tok.null:
                flush_current_state()
                state = "NULL"
                last_null = None
                current_lbl_parts.clear()
                i += 1
            else:
                if state == "ENT_TEXT":
                    current_ent_text.append(t)
                elif state == "ENT_TYPE":
                    current_ent_type.append(t)
                elif state == "HEAD":
                    current_head_parts.append(t)
                elif state == "REL":
                    current_rel_parts.append(t)
                elif state == "TAIL":
                    current_tail_parts.append(t)
                elif state == "NULL":
                    current_lbl_parts.append(t)
                i += 1

        flush_current_state()
        return _deduplicate_entities(entity_list), rejected

    if tok.variant in {"re", "boundary_re"}:
        entities: List[EntityBlock] = []
        entity_dict: Dict[str, EntityBlock] = {}
        rejected: List[RejectedItem] = []
        
        # Extract only the TRIPLETS: portion of the text
        if "TRIPLETS:" in text:
            extract_part = text.split("TRIPLETS:", 1)[1]
            if "MISSING:" in extract_part:
                extract_part = extract_part.split("MISSING:", 1)[0]
        else:
            extract_part = text
            
        # Tokenize the extract part using special tokens
        special_tokens = sorted(tok.all_tokens, key=len, reverse=True)
        pattern = re.compile(f"({'|'.join(map(re.escape, special_tokens))})")
        tokens = [t.strip() for t in pattern.split(extract_part) if t.strip()]
        
        state = "IDLE"
        current_head_text = []
        current_head_type = []
        current_rel_type = []
        current_tail_text = []
        current_tail_type = []
        
        def flush_triplet():
            nonlocal current_head_text, current_head_type, current_rel_type, current_tail_text, current_tail_type
            h_txt = " ".join(current_head_text).strip()
            r_typ = " ".join(current_rel_type).strip()
            t_txt = " ".join(current_tail_text).strip()
            h_typ = " ".join(current_head_type).strip() if current_head_type else None
            t_typ = " ".join(current_tail_type).strip() if current_tail_type else None
            
            if h_txt and r_typ and t_txt:
                if h_txt not in entity_dict:
                    entity_dict[h_txt] = {"text": h_txt, "type": h_typ, "relations": []}
                    entities.append(entity_dict[h_txt])
                elif h_typ and not entity_dict[h_txt].get("type"):
                    entity_dict[h_txt]["type"] = h_typ
                
                entity_dict[h_txt]["relations"].append({
                    "type": r_typ,
                    "tail": t_txt,
                    "tail_type": t_typ
                })
            # Clear relation and tail for the next triplet
            current_rel_type.clear()
            current_tail_text.clear()
            current_tail_type.clear()
            
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == tok.head:
                flush_triplet()
                current_head_text.clear()
                current_head_type.clear()
                state = "HEAD_TEXT"
            elif t == getattr(tok, "nest", None):
                flush_triplet()
                state = "REL"
                current_rel_type.clear()
            elif t == tok.type_:
                if state == "HEAD_TEXT":
                    state = "HEAD_TYPE"
                elif state == "TAIL_TEXT":
                    state = "TAIL_TYPE"
            elif t == tok.rel:
                state = "REL"
            elif t == tok.tail:
                state = "TAIL_TEXT"
            else:
                if state == "HEAD_TEXT":
                    current_head_text.append(t)
                elif state == "HEAD_TYPE":
                    current_head_type.append(t)
                elif state == "REL":
                    current_rel_type.append(t)
                elif state == "TAIL_TEXT":
                    current_tail_text.append(t)
                elif state == "TAIL_TYPE":
                    current_tail_type.append(t)
            i += 1
            
        flush_triplet()
        return _deduplicate_entities(entities), rejected

    return [], []


def extract_triplets(entities: List[EntityBlock], include_types: bool = False) -> List[Triplet]:
    if include_types:
        return [(
            f"{ent['text']} [{ent.get('type') or '?'}]", 
            rel["type"], 
            f"{rel['tail']} [{rel.get('tail_type') or '?'}]"
        ) for ent in entities for rel in ent["relations"]]
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
    
    null_parts = [f"{tok.null} {t}" for t in e_types] + [f"{tok.null} {r}" for r in r_types]
    parts.extend(null_parts)


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