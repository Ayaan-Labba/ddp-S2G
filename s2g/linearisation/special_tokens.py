"""
Special token registry for the S2G model (ablation branch).

**Every** linearisation token is a reserved T5 sentinel, so nothing is ever added
to the vocabulary and no embedding resize is needed. The roles sit at the top of
the range and block markers count up from ``<extra_id_0>`` into what is left.
"""
from __future__ import annotations

from typing import Dict, List, Set
from transformers import AutoTokenizer

ALL_TOKEN_NAMES: List[str] = ['e_type', 'r_type', 'nr_type', 'tail', 'null']
VALID_VARIANTS: Set = {'re', 'boundary_re', 'boundary_joint', 'joint'}

# T5 / Flan-T5 ship exactly <extra_id_0> .. <extra_id_99>. The roles take the top
# of that range, leaving the rest to block markers; deriving the split keeps the
# two from ever disagreeing.
NUM_SENTINELS = 100
MAX_MARKER_SENTINELS = NUM_SENTINELS - len(ALL_TOKEN_NAMES)


class S2GTokens:
    # <extra_id_95> .. <extra_id_99>, in ALL_TOKEN_NAMES order.
    token_strs: Dict[str, str] = {
        name: f"<extra_id_{MAX_MARKER_SENTINELS + i}>"
        for i, name in enumerate(ALL_TOKEN_NAMES)
    }

    base_tok_map = {
        're':             {'e_type', 'r_type', 'nr_type', 'tail'},
        'boundary_re':    {'r_type', 'nr_type', 'tail'},
        'boundary_joint': {'r_type', 'nr_type', 'tail'},
        'joint':          {'e_type', 'r_type', 'nr_type', 'tail'},
    }

    def __init__(self, variant: str, use_rejection: bool = False) -> None:
        self.variant = variant
        self.use_rejection = use_rejection
        self.active_tokens = self.base_tok_map.get(variant, self.base_tok_map['joint']).copy()

        if use_rejection:
            self.active_tokens.add('null')

    @property
    def role_token_strs(self) -> Set[str]:
        """
        Active role tokens.

        ``parse_graph`` tests these by exact identity *before* falling back to the
        sentinel pattern, so a role sentinel can never be mistaken for a marker.
        """
        return {self.token_strs[name] for name in self.active_tokens}

    @staticmethod
    def sentinel_token(idx: int) -> str:
        return f"<extra_id_{idx}>"


def verify_token_integrity(tokenizer: AutoTokenizer) -> None:
    """
    Assert that every sentinel survives tokenisation intact.

    Markers and roles are all sentinels, so checking the range covers everything
    the format can emit. It needs exactly two properties from each, and asserts
    those directly rather than checking membership of any registry:

    1. it encodes to a **single id** — a marker split across pieces would break the
       rolling numbering, and a role token split across pieces would never be
       matched by the parser;
    2. it decodes back **verbatim** under ``skip_special_tokens=False`` — the
       evaluator parses the decoded generation, not the emitted string.

    Membership lists are the wrong probe here: ``additional_special_tokens`` was
    removed in transformers 5, and ``all_special_tokens`` can drop the sentinels
    while ``added_tokens_decoder`` keeps them flagged and both properties above
    continue to hold. Checking behaviour survives both quirks.
    """
    unk_id = tokenizer.unk_token_id
    unknown, multi, mangled = [], [], []

    for token in (S2GTokens.sentinel_token(i) for i in range(NUM_SENTINELS)):
        if tokenizer.convert_tokens_to_ids(token) == unk_id:
            unknown.append(token)
            continue

        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            multi.append(token)
        elif tokenizer.decode(ids, skip_special_tokens=False).strip() != token:
            mangled.append(token)

    if unknown or multi or mangled:
        trim = lambda xs: f"{xs[:5]}{'...' if len(xs) > 5 else ''}"
        raise RuntimeError(
            "Token integrity check failed — the linearisation format cannot survive "
            f"a tokenizer round trip. Unknown: {trim(unknown)}; split into several "
            f"ids: {trim(multi)}; did not decode back verbatim: {trim(mangled)}."
        )


def get_token_ids(tokenizer, tokens: S2GTokens) -> Dict[str, int]:
    res = {}
    unk_id = tokenizer.unk_token_id
    for idx, name in enumerate(ALL_TOKEN_NAMES):
        if name in tokens.active_tokens:
            token_str = tokens.token_strs[name]
            token_id = tokenizer.convert_tokens_to_ids(token_str)

            if token_id is not None and token_id != unk_id:
                res[name] = token_id
                continue

        res[name] = -(idx + 200)

    return res
