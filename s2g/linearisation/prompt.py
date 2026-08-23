"""
Encoder input (prompt) construction for the S2G model.
"""
from __future__ import annotations

import random
from typing import List


def build_re_encoder_input(
        ent_types: List[str], 
        rel_types: List[str], 
        text: str, 
        random_order: bool = False, 
        prompt: str = 'natural'
    ) -> str:
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
    if prompt in {False, 'false', 'False'}:
        return text
    
    types = random.sample(rel_types, len(rel_types)) if random_order else sorted(rel_types)
    r_types_str = ", ".join(f"{r}" for r in types)
    return f"Extract all entities and find relations of type [{r_types_str}] among the extracted entities. Text: {text}"
