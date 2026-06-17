"""
Public API for the S2G encoder/decoder format.
"""
from .special_tokens import (
    AnyTokens, S2GTokens, add_special_tokens_to_tokenizer, get_token_ids, VARIANT_TO_TASKS,
)
from .ssi import (
    build_boundary_joint_encoder_input,
    build_joint_encoder_input, build_ent_ssi, build_re_encoder_input,
    build_boundary_re_encoder_input, build_rel_ssi,
    find_all_token_spans,
)
from .sel import (
    EntityBlock, RejectedItem, Triplet, build_sel, extract_triplets,
    filter_entity_blocks, organize_by_entity, parse_sel,
)

__all__ = [
    "AnyTokens", "S2GTokens", "add_special_tokens_to_tokenizer", "get_token_ids", "VARIANT_TO_TASKS",
    "build_boundary_joint_encoder_input", "build_joint_encoder_input",
    "build_ent_ssi", "build_re_encoder_input", "build_boundary_re_encoder_input",
    "build_rel_ssi", "find_all_token_spans",
    "EntityBlock", "RejectedItem", "Triplet", "build_sel", "extract_triplets",
    "filter_entity_blocks", "organize_by_entity", "parse_sel",
]