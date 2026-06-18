"""
Special token registry for the S2G model.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase
import torch

# All attribute names that carry a token string, in a stable order.
_ALL_ATTR_NAMES: List[str] = [
    "ner", "re", "type_", "rel",
    "ent_start", "tail", "null",
    "head", "nest", "text",
]


class S2GTokens:
    def __init__(self, variant: str, use_rejection: bool = False) -> None:
        self.variant = variant
        self.use_rejection = use_rejection

        self.ner       = "<extra_id_0>"
        self.re        = "<extra_id_1>"
        self.type_     = "<extra_id_2>"
        self.rel       = "<extra_id_3>"
        self.ent_start = "<extra_id_4>"
        self.tail      = "<extra_id_5>"
        self.null      = "<extra_id_6>"
        self.head      = "<extra_id_7>"
        self.nest      = "<extra_id_8>"
        self.text      = "<extra_id_9>"

        active_map = {
            "re":                {"re", "text", "type_", "head", "rel", "tail", "nest"},
            "boundary_re":       {"re", "text", "head", "rel", "tail", "nest"},
            "boundary_joint":    {"re", "text", "head", "rel", "tail", "nest", "ent_start"},
            "joint":             {"ner", "re", "text", "head", "type_", "rel", "tail", "nest", "ent_start"},
        }
        self._active = set(active_map.get(variant, active_map["joint"]))
        self._active.add("null")

    @property
    def all_tokens(self) -> List[str]:
        return [getattr(self, attr) for attr in _ALL_ATTR_NAMES if attr in self._active]


AnyTokens = S2GTokens

VARIANT_TO_TASKS: Dict[str, List[str]] = {
    "re":                ["re"],
    "boundary_re":       ["boundary_re"],
    "boundary_joint":    ["boundary_joint"],
    "joint":             ["joint"],
}


def add_special_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    tokens: AnyTokens,
    model: Optional[PreTrainedModel] = None,
    warm: bool = True,
) -> int:
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens.all_tokens}
    )
    if model is not None:
        if num_added > 0:
            model.resize_token_embeddings(len(tokenizer))

        if warm:
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
                tokens.head:      "subject: ",
                tokens.tail:      "object: ",
                tokens.rel:       "relation: ",
                tokens.type_:     "type: ",
                tokens.ner:       "find type: ",
                tokens.re:        "find relation: ",
                tokens.text:      "in the text: ",
                tokens.nest:      "the same subject",
                tokens.ent_start: "entity: ",
                tokens.null:      "not found: ",
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