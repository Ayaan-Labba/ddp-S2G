"""
Special token registry for the S2G model.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch

# All attribute names that carry a token string, in a stable order.
_ALL_ATTR_NAMES: List[str] = [
    "bound", "ner", "re", "type_", "rel",
    "ent_start", "ent_end", "tail", "null",
    "head", "nest", "text", "trip", "sep",
]


class S2GTokens:
    def __init__(self, variant: str, use_rejection: bool = False) -> None:
        self.variant = variant
        self.use_rejection = use_rejection

        self.bound     = "<bound>"
        self.ner       = "<ner>"
        self.re        = "<re>"
        self.type_     = "<type>"
        self.rel       = "<rel>"
        self.ent_start = "<ent>"
        self.ent_end   = "</ent>"
        self.tail      = "<tail>"
        self.null      = "<null>"
        self.head      = "<head>"
        self.nest      = "<nest>"
        self.text      = "<text>"
        self.trip      = "<trip>"
        self.sep       = "<sep>"

        active_map = {
            "boundary":          {"bound", "ent_start", "ent_end"},
            "ner":               {"ner", "text", "ent_start", "ent_end", "type_"},
            "re":                {"re", "text", "trip", "sep", "type_"},
            "boundary_re":       {"re", "text", "trip", "sep"},
            "pipeline":          {"bound", "ner", "re", "text", "ent_start", "ent_end", "type_", "head", "rel", "tail", "nest"},
            "boundary_pipeline": {"bound", "re", "text", "ent_start", "ent_end", "head", "rel", "tail", "nest"},
            "boundary_joint":    {"re", "text", "head", "rel", "tail", "nest", "ent_start"},
            "joint":             {"ner", "re", "text", "head", "type_", "rel", "tail", "nest", "ent_start"},
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
        return [getattr(self, attr) for attr in _ALL_ATTR_NAMES if attr in self._active]

    def as_dict(self) -> Dict[str, str]:
        return {attr: getattr(self, attr) for attr in _ALL_ATTR_NAMES}


AnyTokens = S2GTokens

VARIANT_TO_TASKS: Dict[str, List[str]] = {
    "boundary":          ["boundary"],
    "ner":               ["ner"],
    "re":                ["re"],
    "boundary_re":       ["boundary_re"],
    "pipeline":          ["ner", "re"],
    "boundary_pipeline": ["boundary", "boundary_re"],
    "boundary_joint":    ["boundary_joint"],
    "joint":             ["joint"],
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

        # Warm-start new special-token embeddings with the mean embedding of a
        # semantically related natural-language phrase.  This gives the model a
        # better initialisation than a random vector and typically speeds up
        # convergence.
        #
        # BUG FIX: the original code compared token *strings* (e.g. "<trip>")
        # against tokens._active, which stores *attribute names* (e.g. "trip").
        # These never matched, so every warm-start was silently skipped.
        # Fix: build a reverse mapping {token_string → attr_name} and check the
        # attr_name against tokens._active instead.
        token_map = {
            tokens.trip:      ".",
            tokens.sep:       ":",
            tokens.head:      "the subject",
            tokens.tail:      "to the object",
            tokens.rel:       "has relation",
            tokens.type_:     "of type",
            tokens.ner:       "entity",
            tokens.re:        "relation",
            tokens.bound:     "boundary",
            tokens.text:      "text",
            tokens.nest:      "the same subject",
            tokens.ent_start: "entity span start",
            tokens.ent_end:   "entity span end",
            tokens.null:      "did not find",
        }

        # Reverse map: token string → attribute name
        _str_to_attr: Dict[str, str] = {
            getattr(tokens, a): a for a in _ALL_ATTR_NAMES
        }

        with torch.no_grad():
            in_emb  = model.get_input_embeddings().weight
            out_mod = model.get_output_embeddings()
            out_emb = out_mod.weight if out_mod is not None else None

            for special_tok, init_text in token_map.items():
                # Resolve the token string back to its attribute name and check
                # whether that attribute is active for this variant.
                attr = _str_to_attr.get(special_tok)
                if attr is None or attr not in tokens._active:
                    continue

                new_id   = tokenizer.convert_tokens_to_ids(special_tok)
                init_ids = tokenizer.encode(init_text, add_special_tokens=False)
                if init_ids and new_id != tokenizer.unk_token_id:
                    in_emb[new_id].copy_(in_emb[init_ids].mean(dim=0))
                    if out_emb is not None and out_emb.data_ptr() != in_emb.data_ptr():
                        out_emb[new_id].copy_(out_emb[init_ids].mean(dim=0))

    return num_added


def get_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    tokens: AnyTokens,
) -> Dict[str, int]:
    res = {}
    unk_id = tokenizer.unk_token_id
    for idx, name in enumerate(_ALL_ATTR_NAMES):
        token_str = getattr(tokens, name)
        token_id  = tokenizer.convert_tokens_to_ids(token_str)
        if name in tokens._active and token_id is not None and token_id != unk_id:
            res[name] = token_id
        else:
            res[name] = -(idx + 200)
    return res