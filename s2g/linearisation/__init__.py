"""
linearisation package — public API for the S2G encoder/decoder format.

Re-exports the complete public surface of all three submodules so that
callers can write::

    from s2g.linearisation import (
        PIPELINE_TOKENS, build_boundary_encoder_input, parse_sel, ...
    )

rather than importing from the individual modules.
"""

from .special_tokens import (
    AnyTokens,
    JointTokens,
    JOINT_TOKENS,
    PipelineTokens,
    PIPELINE_TOKENS,
    add_special_tokens_to_tokenizer,
    get_token_ids,
)

from .ssi import (
    augment_ner_text,
    augment_re_text,
    build_boundary_encoder_input,
    build_joint_encoder_input,
    build_joint_plus_encoder_input,
    build_ner_encoder_input,
    build_ner_ssi,
    build_re_encoder_input,
    build_rel_ssi,
    find_all_token_spans,
    find_token_span,
)

from .sel import (
    EntityBlock,
    RejectedItem,
    Triplet,
    build_sel,
    extract_triplets,
    filter_entity_blocks,
    organize_by_entity,
    parse_sel,
)

__all__ = [
    # special_tokens
    "AnyTokens",
    "JointTokens",
    "JOINT_TOKENS",
    "PipelineTokens",
    "PIPELINE_TOKENS",
    "add_special_tokens_to_tokenizer",
    "get_token_ids",
    # ssi
    "augment_ner_text",
    "augment_re_text",
    "build_boundary_encoder_input",
    "build_joint_encoder_input",
    "build_joint_plus_encoder_input",
    "build_ner_encoder_input",
    "build_ner_ssi",
    "build_re_encoder_input",
    "build_rel_ssi",
    "find_all_token_spans",
    "find_token_span",
    # sel
    "EntityBlock",
    "RejectedItem",
    "Triplet",
    "build_sel",
    "extract_triplets",
    "filter_entity_blocks",
    "organize_by_entity",
    "parse_sel",
]