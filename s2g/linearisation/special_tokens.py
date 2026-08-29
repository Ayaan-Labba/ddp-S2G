"""
Special token registry for the S2G model (ablation branch).

Every linearisation role sits on a reserved T5 sentinel at the top of the range,
so nothing is ever added to the vocabulary.  The bottom of the range
(``<extra_id_0>`` .. ``<extra_id_93>``) is left free for rolling block markers.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set
from transformers import AutoModel, AutoTokenizer
import torch

ALL_TOKEN_NAMES: List[str] = ['ent', 'e_type', 'r_type', 'nr_type', 'tail', 'null']
VALID_VARIANTS: Set = {'re', 'boundary_re', 'boundary_joint', 'joint'}
VALID_MARKERS: Set = {'fixed', 'rolling'}

# T5 / Flan-T5 ship exactly <extra_id_0> .. <extra_id_99>. The top six carry the
# role tokens, leaving <extra_id_0> .. <extra_id_93> for rolling markers.
NUM_SENTINELS = 100
MAX_MARKER_SENTINELS = 94


class S2GTokens:
    token_strs = {
        'ent':      '<extra_id_94>',
        'e_type':   '<extra_id_95>',
        'r_type':   '<extra_id_96>',
        'nr_type':  '<extra_id_97>',
        'tail':     '<extra_id_98>',
        'null':     '<extra_id_99>',
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


def verify_sentinel_integrity(tokenizer: AutoTokenizer) -> None:
    """
    Assert that every ``<extra_id_i>`` survives tokenizer construction.

    ``add_special_tokens({'additional_special_tokens': [...]})`` can *replace* the
    tokenizer's existing list, which would deregister the rolling-marker range and
    break only the rolling arm — silently, and while the fixed arm keeps working.

    Probed via ``all_special_tokens``, which exists across transformers 4 and 5;
    ``additional_special_tokens`` was removed in 5.x.
    """
    registered = set(getattr(tokenizer, 'all_special_tokens', []) or [])
    unk_id = tokenizer.unk_token_id
    missing, multi = [], []

    for idx in range(NUM_SENTINELS):
        token = S2GTokens.sentinel_token(idx)
        if token not in registered or tokenizer.convert_tokens_to_ids(token) == unk_id:
            missing.append(token)
            continue
        if len(tokenizer.encode(token, add_special_tokens=False)) != 1:
            multi.append(token)

    if missing or multi:
        raise RuntimeError(
            "Sentinel integrity check failed — the linearisation format cannot be "
            f"tokenised. Deregistered or unknown: {missing[:5]}{'...' if len(missing) > 5 else ''}; "
            f"split into several ids: {multi[:5]}{'...' if len(multi) > 5 else ''}."
        )


def add_special_tokens_to_tokenizer(
        tokenizer: AutoTokenizer,
        tokens: S2GTokens,
        model: Optional[AutoModel] = None,
        warm_start: bool = True,
    ) -> int:
    # Every linearisation token is a reserved sentinel, so it is already in the
    # vocabulary and no registration is needed. Calling ``add_special_tokens``
    # anyway would overwrite ``additional_special_tokens`` and drop the rolling
    # marker range, so it runs only if something is genuinely absent.
    missing = [t for t in tokens.all_tokens if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id]
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': tokens.all_tokens}) if missing else 0

    if model is not None:
        if num_added > 0:
            model.config.tie_word_embeddings = False # weights are untied for flan-t5 models
            model.resize_token_embeddings(len(tokenizer))

        if warm_start:
            # Inert while ``train.warm_start: False``, which every ablation run sets:
            # the sentinels already carry pretrained embeddings, and overwriting them
            # would do so asymmetrically between the fixed and rolling arms.
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

    verify_sentinel_integrity(tokenizer)

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
