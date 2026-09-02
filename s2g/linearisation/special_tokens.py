"""
Special token registry for the S2G model (ablation branch).

Each linearisation role has its own dedicated token, added to the tokenizer at
load time.  Rolling block markers are the exception: they stay on T5's reserved
sentinels, so the whole ``<extra_id_0>`` .. ``<extra_id_99>`` range is available
to them.

Only the *active* tokens are added, so a rolling run never pays for ``<ent>`` and
a run without rejection never pays for ``<null>``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set
from transformers import AutoModel, AutoTokenizer
import torch

ALL_TOKEN_NAMES: List[str] = ['ent', 'e_type', 'r_type', 'nr_type', 'tail', 'null']
VALID_VARIANTS: Set = {'re', 'boundary_re', 'boundary_joint', 'joint'}
VALID_MARKERS: Set = {'fixed', 'rolling'}

# T5 / Flan-T5 ship exactly <extra_id_0> .. <extra_id_99>. The role tokens no
# longer occupy any of them, so the entire range is free for rolling markers.
NUM_SENTINELS = 100
MAX_MARKER_SENTINELS = 100


class S2GTokens:
    token_strs = {
        'ent':      '<ent>',
        'e_type':   '<e_type>',
        'r_type':   '<r_type>',
        'nr_type':  '<nr_type>',
        'tail':     '<tail>',
        'null':     '<null>',
    }

    # ``ent`` is deliberately absent: it is a *marker*, not a role, and joins the
    # active set only under fixed markers.
    base_tok_map = {
        're':             {'e_type', 'r_type', 'nr_type', 'tail'},
        'boundary_re':    {'r_type', 'nr_type', 'tail'},
        'boundary_joint': {'r_type', 'nr_type', 'tail'},
        'joint':          {'e_type', 'r_type', 'nr_type', 'tail'},
    }

    def __init__(self, variant: str, use_rejection: bool = False, markers: str = 'fixed') -> None:
        if markers not in VALID_MARKERS:
            raise ValueError(f"Unknown marker style {markers!r}; expected one of {VALID_MARKERS}.")

        self.variant = variant
        self.markers = markers
        self.use_rejection = use_rejection
        self.active_tokens = self.base_tok_map.get(variant, self.base_tok_map['joint']).copy()

        if markers == 'fixed':
            self.active_tokens.add('ent')
            # Under rolling markers rejection is opened by the next sentinel in the
            # sequence rather than by a dedicated token, so ``null`` stays inactive.
            if use_rejection:
                self.active_tokens.add('null')

        self._all_tokens = [self.token_strs[tok] for tok in ALL_TOKEN_NAMES if tok in self.active_tokens]

    @property
    def all_tokens(self) -> List[str]:
        return self._all_tokens

    @property
    def role_token_strs(self) -> Set[str]:
        """
        Active role tokens — everything except the block marker.

        ``parse_graph`` tests these by exact identity *before* falling back to the
        sentinel pattern, so a role token can never be mistaken for a separator.
        """
        return {self.token_strs[name] for name in self.active_tokens if name != 'ent'}

    @staticmethod
    def sentinel_token(idx: int) -> str:
        return f"<extra_id_{idx}>"


def verify_token_integrity(tokenizer: AutoTokenizer, tokens: Optional[S2GTokens] = None) -> None:
    """
    Assert that every token the format emits survives tokenisation intact.

    The format needs exactly two properties from each of them, and asserts those
    directly rather than checking membership of any registry:

    1. it encodes to a **single id** — a marker split across pieces would break the
       rolling numbering, and a role token split across pieces would never be
       matched by the parser;
    2. it decodes back **verbatim** under ``skip_special_tokens=False`` — the
       evaluator parses the decoded generation, not the emitted string.

    Membership lists are the wrong probe here. ``additional_special_tokens`` was
    removed in transformers 5, and ``all_special_tokens`` drops the sentinels the
    moment new tokens are added — while ``added_tokens_decoder`` keeps them flagged
    and both properties above continue to hold. Checking behaviour survives both
    quirks.
    """
    unk_id = tokenizer.unk_token_id
    checked = [S2GTokens.sentinel_token(i) for i in range(NUM_SENTINELS)]
    if tokens is not None:
        checked += list(tokens.all_tokens)

    unknown, multi, mangled = [], [], []
    for token in checked:
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


def add_special_tokens_to_tokenizer(
        tokenizer: AutoTokenizer,
        tokens: S2GTokens,
        model: Optional[AutoModel] = None,
        warm_start: bool = True,
    ) -> int:
    # Only the *active* tokens are registered, so a rolling run never adds ``<ent>``
    # and a run without rejection never adds ``<null>``. Re-running on a saved
    # checkpoint is a no-op: nothing is missing, so nothing is added and the
    # embedding matrix is left alone.
    missing = [t for t in tokens.all_tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': tokens.all_tokens}) if missing else 0

    if model is not None:
        if num_added > 0:
            model.config.tie_word_embeddings = False # weights are untied for flan-t5 models
            model.resize_token_embeddings(len(tokenizer))

        if warm_start:
            # The role tokens are new vocabulary, so their rows start random and this
            # is the only thing that gives them a sensible prior. Held off anyway
            # (``train.warm_start: False``) to keep the ablation's held-constant list
            # intact — the arms then differ only in the format under test.
            token_init_phrases = {
                'ent':      'entity: ',
                'e_type':   'entity type: ',
                'r_type':   'relation: ',
                'nr_type':  'next relation: ',
                'tail':     'object: ',
                'null':     'not found: ',
            }

            with torch.no_grad():
                in_emb = model.get_input_embeddings().weight
                out_mod = model.get_output_embeddings()
                out_emb = out_mod.weight if out_mod is not None else None

                for tok_name, init_text in token_init_phrases.items():
                    if tok_name not in tokens.active_tokens:
                        continue

                    special_tok = tokens.token_strs[tok_name]
                    new_id = tokenizer.convert_tokens_to_ids(special_tok)
                    init_ids = tokenizer.encode(init_text, add_special_tokens=False)

                    if init_ids and new_id != tokenizer.unk_token_id:
                        # Warm start input embeddings by taking the mean of the initialization phrase
                        mean_in_emb = in_emb[init_ids].mean(dim=0)
                        in_emb[new_id].copy_(mean_in_emb)
                        if out_emb is not None:
                            try:
                                mean_out_emb = out_emb[init_ids].mean(dim=0)
                                out_emb[new_id].copy_(mean_out_emb)
                            except (IndexError, RuntimeError):
                                out_emb[new_id].copy_(mean_in_emb)

    verify_token_integrity(tokenizer, tokens)

    return num_added


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
