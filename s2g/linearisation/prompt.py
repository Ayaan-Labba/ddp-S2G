"""
Encoder input (prompt) construction for the S2G model.

One builder serves every variant; the boundary variants simply drop the entity
clause. The leading verb ("Extract" / "Mark") is an ablation arm edited by hand
here — it is deliberately not a config key.
"""
from __future__ import annotations

import random
from typing import List, Optional

RAW_TEXT_PROMPTS = {False, 'false', 'False'}


def _order(types: Optional[List[str]], random_order: bool) -> List[str]:
    types = list(types or [])
    return random.sample(types, len(types)) if random_order else sorted(types)


def build_instruction(
        rel_types: List[str],
        ent_types: Optional[List[str]] = None,
        use_ent_types: bool = True,
        random_order: bool = False,
    ) -> str:
    """
    The task instruction alone, without the source text.

    Kept separate from ``build_encoder_input`` so that Stage 3's CoT prompt can
    reuse the identical wording around a different frame.
    """
    r_types_str = ", ".join(_order(rel_types, random_order))
    if not use_ent_types:
        return f"Extract all relations from [{r_types_str}] in the given text."

    e_types_str = ", ".join(_order(ent_types, random_order))
    return f"Extract all entities from [{e_types_str}] and relations from [{r_types_str}] in the given text."


def build_encoder_input(
        text: str,
        rel_types: List[str],
        ent_types: Optional[List[str]] = None,
        use_ent_types: bool = True,
        random_order: bool = False,
        prompt: str = 'natural',
    ) -> str:
    if prompt in RAW_TEXT_PROMPTS:
        return text

    instruction = build_instruction(rel_types, ent_types, use_ent_types=use_ent_types, random_order=random_order)
    return f"{instruction} Text: {text}"


# Per-variant wrappers. Signatures are unchanged from the branch's history so that
# ``collator.py`` needs no edits here.

def build_re_encoder_input(
        ent_types: List[str],
        rel_types: List[str],
        text: str,
        random_order: bool = False,
        prompt: str = 'natural'
    ) -> str:
    return build_encoder_input(text, rel_types, ent_types, True, random_order, prompt)


def build_joint_encoder_input(
        ent_types: List[str],
        rel_types: List[str],
        text: str,
        random_order: bool = False,
        prompt: str = 'natural'
    ) -> str:
    return build_encoder_input(text, rel_types, ent_types, True, random_order, prompt)


def build_boundary_re_encoder_input(
        rel_types: List[str],
        text: str,
        random_order: bool = False,
        prompt: str = 'natural'
    ) -> str:
    return build_encoder_input(text, rel_types, None, False, random_order, prompt)


def build_boundary_joint_encoder_input(
        rel_types: List[str],
        text: str,
        random_order: bool = False,
        prompt: str = 'natural'
    ) -> str:
    return build_encoder_input(text, rel_types, None, False, random_order, prompt)
