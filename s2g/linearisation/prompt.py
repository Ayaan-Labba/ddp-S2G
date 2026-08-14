"""
SSI construction and text augmentation for the S2G encoder input.
"""
from __future__ import annotations

import random
from typing import List, Tuple, Dict

from .special_tokens import S2GTokens


TOK_CACHE: Dict[str, S2GTokens] = {}


def get_tok(variant: str, prompt: str = 'ssi') -> S2GTokens:
    """Helper to lazily load and cache the S2GTokens instance on demand."""
    cache_key = (variant, prompt)
    if cache_key not in TOK_CACHE:
        TOK_CACHE[cache_key] = S2GTokens(variant, prompt=prompt)
    
    return TOK_CACHE[cache_key]


def build_ent_ssi(ent_types: List[str], variant: str, random_order: bool = False) -> str:
    tok = get_tok(variant, 'ssi')
    types = random.sample(ent_types, len(ent_types)) if random_order else sorted(ent_types)
    ner_str = tok.token_strs['ner']
    return " ".join([f"{ner_str} {t}" for t in types])


def build_rel_ssi(rel_types: List[str], variant: str, random_order: bool = False) -> str:
    tok = get_tok(variant, 'ssi')
    types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
    re_str = tok.token_strs['re']
    return " ".join([f"{re_str} {t}" for t in types])


def build_re_encoder_input(
        ent_types: List[str], 
        rel_types: List[str], 
        text: str, 
        random_order: bool = False, 
        prompt: str = 'natural'
    ) -> str:
    if prompt == 'ssi':
        ent_ssi = build_ent_ssi(ent_types, 're', random_order)
        rel_ssi = build_rel_ssi(rel_types, 're', random_order)
        prefix = " ".join(filter(None, [ent_ssi, rel_ssi]))
        return f"{prefix} {get_tok('re').token_strs['text']} {text}"
    
    if prompt in {False, 'false', 'False'}:
        return text

    r_types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
    e_types = random.sample(ent_types, len(ent_types)) if random_order else sorted(ent_types)
    r_types_str = ", ".join(f"{r}" for r in r_types)
    e_types_str = ", ".join(f"{e}" for e in e_types)
    return f"Extract all relations of type [{r_types_str}] among the entities of type [{e_types_str}] in the given text. Text: {text}"


def build_boundary_re_encoder_input(
        rel_types: List[str], 
        text: str, 
        random_order: bool = False, 
        prompt: str = 'natural'
    ) -> str:
    if prompt == 'ssi':
        return f"{build_rel_ssi(rel_types, 'boundary_re', random_order)} {get_tok('boundary_re').token_strs['text']} {text}"
    
    if prompt in {False, 'false', 'False'}:
        return text
    
    r_types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
    r_types_str = ", ".join(f"{r}" for r in r_types)
    return f"Extract all relations of type [{r_types_str}] among the entities in the given text. Text: {text}"


def build_joint_encoder_input(
        ent_types: List[str], 
        rel_types: List[str], 
        text: str, 
        random_order: bool = False, 
        prompt: str = 'natural'
    ) -> str:
    if prompt == 'ssi':
        ent_ssi = build_ent_ssi(ent_types, 'joint', random_order)
        rel_ssi = build_rel_ssi(rel_types, 'joint', random_order)
        prefix = " ".join(filter(None, [ent_ssi, rel_ssi]))
        return f"{prefix} {get_tok('joint').token_strs['text']} {text}"
        
    if prompt in {False, 'false', 'False'}:
        return text

    ent_types = random.sample(ent_types, len(ent_types)) if random_order else sorted(ent_types)
    r_types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
    ent_types_str = ", ".join(f"{e}" for e in ent_types)
    r_types_str = ", ".join(f"{r}" for r in r_types)
    return f"Extract all entities of type [{ent_types_str}] and find relations of type [{r_types_str}] among the extracted entities. Text: {text}"


def build_boundary_joint_encoder_input(
        rel_types: List[str], 
        text: str, 
        random_order: bool = False,  
        prompt: str = 'natural'
    ) -> str:
    if prompt == 'ssi':
        return f"{build_rel_ssi(rel_types, 'boundary_joint', random_order)} {get_tok('boundary_joint').token_strs['text']} {text}"
    
    if prompt in {False, 'false', 'False'}:
        return text
    
    types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
    r_types_str = ", ".join(f"{r}" for r in types)
    return f"Extract all entities and find relations of type [{r_types_str}] among the extracted entities. Text: {text}"
