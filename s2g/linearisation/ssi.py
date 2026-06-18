"""
SSI construction and text augmentation for the S2G encoder input.
"""
from __future__ import annotations

import random
from typing import List, Tuple

from .special_tokens import AnyTokens, S2GTokens


def build_ent_ssi(entity_types: List[str], random_order: bool = False, tok: AnyTokens = S2GTokens("joint")) -> str:
    types = random.sample(entity_types, len(entity_types)) if random_order else sorted(entity_types)
    return " ".join(f"{tok.ner} {t}" for t in types)


def build_rel_ssi(rel_types: List[str], random_order: bool = False, tok: AnyTokens = S2GTokens("joint")) -> str:
    types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
    return " ".join(f"{tok.re} {t}" for t in types)


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


def build_re_encoder_input(
    entity_types: List[str], rel_types: List[str], text: str, 
    random_order: bool = False, tok: AnyTokens = S2GTokens("re"), ssi_prompt: str = "ssi"
) -> str:
    if ssi_prompt == "natural":
        r_types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
        e_types = random.sample(entity_types, len(entity_types)) if random_order else sorted(entity_types)
        r_types_str = ", ".join(f"{r}" for r in r_types)
        e_types_str = ", ".join(f"{e}" for e in e_types)
        return f"Extract all relations of type [{r_types_str}] among the entities of type [{e_types_str}] in the given text. Text: {text}"
    elif ssi_prompt in {False, "false", "False"}:
        return text
    else:
        ent_ssi = build_ent_ssi(entity_types, random_order, tok)
        rel_ssi = build_rel_ssi(rel_types, random_order, tok)
        prefix = " ".join(filter(None, [ent_ssi, rel_ssi]))
        return f"{prefix} {tok.text} {text}"


def build_boundary_re_encoder_input(
    rel_types: List[str], text: str, 
    random_order: bool = False, tok: AnyTokens = S2GTokens("boundary_re"), ssi_prompt: str = "ssi"
) -> str:
    if ssi_prompt == "natural":
        r_types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
        r_types_str = ", ".join(f"{r}" for r in r_types)
        return f"Extract all relations of type [{r_types_str}] among the entities in the given text. Text: {text}"
    elif ssi_prompt in {False, "false", "False"}:
        return text
    else:
        ssi = build_rel_ssi(rel_types, random_order, tok)
        return f"{ssi} {tok.text} {text}"


def build_boundary_joint_encoder_input(rel_types: List[str], text: str, random_order: bool = False, tok: AnyTokens = S2GTokens("boundary_joint"), ssi_prompt: str = "ssi") -> str:
    if ssi_prompt == "natural":
        types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
        r_types_str = ", ".join(f"{r}" for r in types)
        return f"Extract all entities and find relations of type [{r_types_str}] among the extracted entities in the given text. Text: {text}"
    elif ssi_prompt in {False, "false", "False"}:
        return text
    else:
        return f"{build_rel_ssi(rel_types, random_order, tok)} {tok.text} {text}"


def build_joint_encoder_input(
    entity_types: List[str], rel_types: List[str], text: str, random_order: bool = False, tok: AnyTokens = S2GTokens("joint"), ssi_prompt: str = "ssi"
) -> str:
    if ssi_prompt == "natural":
        ent_types = random.sample(entity_types, len(entity_types)) if random_order else sorted(entity_types)
        r_types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
        ent_types_str = ", ".join(f"{e}" for e in ent_types)
        r_types_str = ", ".join(f"{r}" for r in r_types)
        return f"Extract all entities of type [{ent_types_str}] and find relations of type [{r_types_str}] among the extracted entities in the given text. Text: {text}"
    elif ssi_prompt in {False, "false", "False"}:
        return text
    else:
        ent_ssi = build_ent_ssi(entity_types, random_order, tok)
        rel_ssi = build_rel_ssi(rel_types, random_order, tok)
        prefix = " ".join(filter(None, [ent_ssi, rel_ssi]))
        return f"{prefix} {tok.text} {text}"