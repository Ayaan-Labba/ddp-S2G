"""
Encoder input (prompt) construction for the S2G model.

One builder serves both variants; ``boundary_joint`` simply drops the entity
clause, in the instruction and in the CoT frame alike. The leading verb
("Extract" / "Mark") is an ablation arm edited by hand here — it is deliberately
not a config key.
"""
from __future__ import annotations

import random
from typing import List, Optional

RAW_TEXT_PROMPTS = {False, 'false', 'False'}


def order_types(types: Optional[List[str]], random_order: bool) -> List[str]:
    """
    The prompt's ordering rule, sorted unless shuffling is asked for.

    Public because ``build_graph``'s rejection tail restates the prompt's negatives
    and has to list them the same way; a second copy of this would be free to drift.
    """
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
    r_types_str = ", ".join(order_types(rel_types, random_order))
    if not use_ent_types:
        return f"Extract all relations from [{r_types_str}] in the given text."

    e_types_str = ", ".join(order_types(ent_types, random_order))
    return f"Extract all entities from [{e_types_str}] and relations from [{r_types_str}] in the given text."


def build_encoder_input(
        text: str,
        rel_types: List[str],
        ent_types: Optional[List[str]] = None,
        use_ent_types: bool = True,
        random_order: bool = False,
        prompt: str = 'natural',
        style: str = 'direct',
    ) -> str:
    # Raw text wins over the frame: there is no instruction to wrap.
    if prompt in RAW_TEXT_PROMPTS:
        return text

    instruction = build_instruction(rel_types, ent_types, use_ent_types=use_ent_types, random_order=random_order)
    if style != 'cot':
        return f"{instruction} Text: {text}"

    # Axis 3 / C2. The clause naming what to look for follows the same
    # ``use_ent_types`` gate as the instruction, so the two never disagree about
    # whether this variant deals in entity types at all.
    missing = "entities and relations" if use_ent_types else "relations"
    return (
        f"Q: {instruction} Find the missing {missing}. "
        f"Text: {text} A: Let's think step-by-step"
    )


# Per-variant wrappers, kept so ``collator.py`` dispatches by name.

def build_joint_encoder_input(
        ent_types: List[str],
        rel_types: List[str],
        text: str,
        random_order: bool = False,
        prompt: str = 'natural',
        style: str = 'direct',
    ) -> str:
    return build_encoder_input(text, rel_types, ent_types, True, random_order, prompt, style)


def build_boundary_joint_encoder_input(
        rel_types: List[str],
        text: str,
        random_order: bool = False,
        prompt: str = 'natural',
        style: str = 'direct',
    ) -> str:
    return build_encoder_input(text, rel_types, None, False, random_order, prompt, style)
