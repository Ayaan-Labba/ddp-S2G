"""
SSI construction and text augmentation for the S2G encoder input.
"""
from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple

from .special_tokens import AnyTokens, S2GTokens


def build_ner_ssi(entity_types: List[str], random_order: bool = False, tok: AnyTokens = S2GTokens("pipeline")) -> str:
    types = random.sample(entity_types, len(entity_types)) if random_order else sorted(entity_types)
    return " ".join(f"{tok.ner} {t}" for t in types)


def build_rel_ssi(rel_types: List[str], random_order: bool = False, tok: AnyTokens = S2GTokens("pipeline")) -> str:
    types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
    return " ".join(f"{tok.re} {t}" for t in types)


def augment_ner_text(source_tokens: List[str], entity_spans: List[Tuple[int, int]], tok: AnyTokens = S2GTokens("pipeline")) -> str:
    # Sort primarily by start index
    spans = sorted(entity_spans, key=lambda s: s[0])
    parts, cursor, last_end = [], 0, -1
    
    for start, end in spans:
        if start >= last_end:  # Greedily ignore overlaps
            parts.extend(source_tokens[cursor:start])
            parts.extend((tok.ent_start, *source_tokens[start:end], tok.ent_end))
            cursor, last_end = end, end
            
    parts.extend(source_tokens[cursor:])
    return " ".join(parts)


def augment_re_text(source_tokens: List[str], entity_data: List[Tuple[int, int, str]], tok: AnyTokens = S2GTokens("pipeline")) -> str:
    data = sorted(entity_data, key=lambda e: e[0])
    accepted: Set[Tuple[int, int]] = set()
    last_end = -1
    for start, end, _ in data:
        if start >= last_end:
            accepted.add((start, end))
            last_end = end

    parts, cursor = [], 0
    for start, end, type_str in data:
        if (start, end) in accepted:
            parts.extend(source_tokens[cursor:start])
            if type_str:
                parts.extend((tok.ent_start, *source_tokens[start:end], tok.type_, type_str, tok.ent_end))
            else:
                parts.extend((tok.ent_start, *source_tokens[start:end], tok.ent_end))
            cursor = end

    parts.extend(source_tokens[cursor:])
    return " ".join(parts)


def find_token_span(source_tokens: List[str], span_text: str) -> Optional[Tuple[int, int]]:
    span_words = span_text.split()
    n = len(span_words)
    if not n: 
        return None
        
    first_word = span_words[0]
    start_idx = 0
    while True:
        try:
            i = source_tokens.index(first_word, start_idx)
            if source_tokens[i : i + n] == span_words:
                return i, i + n
            start_idx = i + 1
        except ValueError:
            return None


def find_all_token_spans(source_tokens: List[str], span_text: str) -> List[Tuple[int, int]]:
    span_words = span_text.split()
    n = len(span_words)
    results = []
    if not n: 
        return results

    first_word = span_words[0]
    start_idx = 0
    
    while start_idx <= len(source_tokens) - n:
        try:
            i = source_tokens.index(first_word, start_idx)
            if source_tokens[i : i + n] == span_words:
                results.append((i, i + n))
                start_idx = i + n
            else:
                start_idx = i + 1
        except ValueError:
            break
            
    return results


def build_boundary_encoder_input(text: str, tok: AnyTokens = S2GTokens("boundary"), ssi_prompt: str = "ssi") -> str:
    if ssi_prompt == "false":
        return text
    elif ssi_prompt == "natural":
        return f"List all entities in the given text:  {text}"
    else:
        return f"{tok.bound} {text}"


def build_ner_encoder_input(
    entity_types: List[str], source_tokens: List[str], entity_spans: List[Tuple[int, int]], 
    random_order: bool = False, tok: AnyTokens = S2GTokens("ner"), ssi_prompt: str = "ssi"
) -> str:
    text = augment_ner_text(source_tokens, entity_spans, tok)
    if ssi_prompt == "natural":
        types = random.sample(entity_types, len(entity_types)) if random_order else sorted(entity_types)
        prefix = f"List all entities of types [{', '.join(types)}] in the given text: "
        return f"{prefix}  {text}"
    elif ssi_prompt == "false":
        return text
    else:
        ssi = build_ner_ssi(entity_types, random_order, tok)
        return f"{ssi} {tok.text} {text}"


def build_re_encoder_input(
    entity_types: List[str], rel_types: List[str], text: str, 
    random_order: bool = False, tok: AnyTokens = S2GTokens("re"), ssi_prompt: str = "ssi"
) -> str:
    if ssi_prompt == "natural":
        r_types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
        e_types = random.sample(entity_types, len(entity_types)) if random_order else sorted(entity_types)
        prefix = f"List all relations of types [{', '.join(r_types)}] among the entities of types [{', '.join(e_types)}] in the given text: "
        return f"{prefix}  {text}"
    elif ssi_prompt == "false":
        return text
    else:
        ent_ssi = build_ner_ssi(entity_types, random_order, tok)
        rel_ssi = build_rel_ssi(rel_types, random_order, tok)
        prefix = " ".join(filter(None, [ent_ssi, rel_ssi]))
        return f"{prefix} {tok.text} {text}"


def build_pipeline_re_encoder_input(
    rel_types: List[str], source_tokens: List[str], entity_data: List[Tuple[int, int, str]], 
    random_order: bool = False, tok: AnyTokens = S2GTokens("re"), ssi_prompt: str = "ssi"
) -> str:
    text = augment_re_text(source_tokens, entity_data, tok)
    if ssi_prompt == "natural":
        types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
        prefix = f"List all relations of types [{', '.join(types)}] among the entities in the given text: "
        return f"{prefix}  {text}"
    elif ssi_prompt == "false":
        return text
    else:
        ssi = build_rel_ssi(rel_types, random_order, tok)
        return f"{ssi} {tok.text} {text}"


def build_boundary_re_encoder_input(
    rel_types: List[str], text: str, 
    random_order: bool = False, tok: AnyTokens = S2GTokens("boundary_re"), ssi_prompt: str = "ssi"
) -> str:
    if ssi_prompt == "natural":
        types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
        prefix = f"List all relations of types [{', '.join(types)}] among the entities in the given text: "
        return f"{prefix}  {text}"
    elif ssi_prompt == "false":
        return text
    else:
        ssi = build_rel_ssi(rel_types, random_order, tok)
        return f"{ssi} {tok.text} {text}"


def build_boundary_joint_encoder_input(rel_types: List[str], text: str, random_order: bool = False, tok: AnyTokens = S2GTokens("boundary_joint"), ssi_prompt: str = "ssi") -> str:
    if ssi_prompt == "natural":
        types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
        prefix = f"List all entities and all relations of types [{', '.join(types)}] in the given text: "
        return f"{prefix}  {text}"
    elif ssi_prompt == "false":
        return text
    else:
        return f"{build_rel_ssi(rel_types, random_order, tok)} {tok.text} {text}"


def build_joint_encoder_input(
    entity_types: List[str], rel_types: List[str], text: str, random_order: bool = False, tok: AnyTokens = S2GTokens("joint"), ssi_prompt: str = "ssi"
) -> str:
    if ssi_prompt == "natural":
        ent_types = random.sample(entity_types, len(entity_types)) if random_order else sorted(entity_types)
        r_types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
        prefix = f"List all entities of types [{', '.join(ent_types)}] and all relations of types [{', '.join(r_types)}] among them in the given text:"
        return f"{prefix}  {text}"
    elif ssi_prompt == "false":
        return text
    else:
        ent_ssi = build_ner_ssi(entity_types, random_order, tok)
        rel_ssi = build_rel_ssi(rel_types, random_order, tok)
        prefix = " ".join(filter(None, [ent_ssi, rel_ssi]))
        return f"{prefix} {tok.text} {text}"