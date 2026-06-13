"""
Special token registry for the S2G model.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase


class S2GTokens:
    def __init__(self, variant: str, use_rejection: bool = False) -> None:
        self.variant = variant
        self.use_rejection = use_rejection
        
        self.bound = "<bound>"
        self.ner = "<ner>"
        self.re = "<re>"
        self.type_ = "<type>"
        self.rel = "<rel>"
        self.ent_start = "<ent>"
        self.ent_end = "</ent>"
        self.tail = "<tail>"
        self.null = "<null>"
        self.head = "<head>"
        self.nest = "<nest>"
        self.text = "<text>"

        active_map = {
            "boundary": {"bound", "ent_start", "ent_end"},
            "ner": {"ner", "text", "ent_start", "ent_end", "type_"},
            "re": {"re", "text", "ent_start", "ent_end", "type_", "head", "rel", "tail", "nest"},
            "boundary_re": {"re", "text", "ent_start", "ent_end", "head", "rel", "tail", "nest"},
            "pipeline": {"bound", "ner", "re", "text", "ent_start", "ent_end", "type_", "head", "rel", "tail", "nest"},
            "boundary_pipeline": {"bound", "re", "text", "ent_start", "ent_end", "head", "rel", "tail", "nest"},
            "boundary_joint": {"re", "text", "head", "rel", "tail", "nest"},
            "joint": {"ner", "re", "text", "head", "type_", "rel", "tail", "nest"},
        }
        self._active = set(active_map.get(variant, active_map["pipeline"]))
        if use_rejection or variant != "boundary":
            self._active.add("null")

    @property
    def task_delimiters(self) -> List[str]:
        delims = []
        if "bound" in self._active:
            delims.append(self.bound)
        if "text" in self._active:
            delims.append(self.text)
        return delims

    @property
    def all_tokens(self) -> List[str]:
        all_attrs = ["bound", "ner", "re", "type_", "rel", "ent_start", "ent_end", "tail", "null", "head", "nest", "text"]
        return [getattr(self, attr) for attr in all_attrs if attr in self._active]

    def as_dict(self) -> Dict[str, str]:
        all_attrs = ["bound", "ner", "re", "type_", "rel", "ent_start", "ent_end", "tail", "null", "head", "nest", "text"]
        return {attr: getattr(self, attr) for attr in all_attrs}


AnyTokens = S2GTokens

VARIANT_TO_TASKS: Dict[str, List[str]] = {
    "boundary": ["boundary"],
    "ner": ["ner"],
    "re": ["re"],
    "boundary_re": ["boundary_re"],
    "pipeline": ["ner", "re"],
    "boundary_pipeline": ["boundary", "boundary_re"],
    "boundary_joint": ["boundary_joint"],
    "joint": ["joint"],
}


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
    res = {}
    unk_id = tokenizer.unk_token_id
    all_attrs = ["bound", "ner", "re", "type_", "rel", "ent_start", "ent_end", "tail", "null", "head", "nest", "text"]
    for idx, name in enumerate(all_attrs):
        token_str = getattr(tokens, name)
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if name in tokens._active and token_id is not None and token_id != unk_id:
            res[name] = token_id
        else:
            res[name] = -(idx + 200)
    return res