"""
Special token registry for the S2G model.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Union

from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class PipelineTokens:
    """Immutable token registry for the Pipeline model."""
    bound: str = "<bound>"
    ner:   str = "<ner>"
    re:    str = "<re>"
    type_: str = "<type>"
    rel:   str = "<rel>"
    ent_start: str = "<ent>"
    ent_end:   str = "</ent>"
    tail: str = "<tail>"
    null: str = "<null>"

    @cached_property
    def task_delimiters(self) -> List[str]:
        return [self.bound, self.ner, self.re]

    @cached_property
    def all_tokens(self) -> List[str]:
        return list(self.as_dict().values())

    def as_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class JointTokens:
    """Immutable token registry for the Joint model."""
    joint:      str = "<joint>"
    joint_plus: str = "<joint+>"
    type_: str = "<type>"
    rel:   str = "<rel>"
    ent_start: str = "<ent>"
    ent_end:   str = "</ent>"
    tail: str = "<tail>"
    null: str = "<null>"

    @cached_property
    def task_delimiters(self) -> List[str]:
        return [self.joint, self.joint_plus]

    @cached_property
    def all_tokens(self) -> List[str]:
        return list(self.as_dict().values())

    def as_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


PIPELINE_TOKENS = PipelineTokens()
JOINT_TOKENS = JointTokens()

AnyTokens = Union[PipelineTokens, JointTokens]


def add_special_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    tokens: AnyTokens,
    model: Optional[PreTrainedModel] = None,
) -> int:
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens.all_tokens}
    )
    if model is not None and num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return num_added


def get_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    tokens: AnyTokens,
) -> Dict[str, int]:
    return {
        name: tokenizer.convert_tokens_to_ids(token_str)
        for name, token_str in tokens.as_dict().items()
    }