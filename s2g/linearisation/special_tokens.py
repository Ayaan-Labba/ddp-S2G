"""
Special token registry for the S2G model.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from transformers import AutoModel, AutoTokenizer
import torch

# All linearisation token names
ALL_TOKEN_NAMES: List[str] = [
    'ner', 're', 'text', 
    'ent', 'e_type',
    'head', 'r_type', 'nr_type', 'tail',
    'null'
]

VALID_VARIANTS: Set = {'re', 'boundary_re', 'boundary_joint', 'joint'}

class S2GTokens:
    # Token string mappings for each linearisation token
    token_strs = {
        'ner':      '<extra_id_0>', 
        're':       '<extra_id_1>', 
        'text':     '<extra_id_2>',
        'ent':      '<extra_id_3>', 
        'e_type':   '<extra_id_4>', 
        'head':     '<extra_id_5>',
        'r_type':   '<extra_id_6>', 
        'nr_type':  '<extra_id_7>', 
        'tail':     '<extra_id_8>',
        'null':     '<extra_id_9>'
    }

    # Token maps for each variant to get active tokens
    base_tok_map = {
        're':             {'head', 'e_type', 'r_type', 'nr_type', 'tail'},
        'boundary_re':    {'head', 'r_type', 'nr_type', 'tail'},
        'boundary_joint': {'ent', 'head', 'r_type', 'nr_type', 'tail'},
        'joint':          {'ent', 'e_type', 'head', 'r_type', 'nr_type', 'tail'},
    }

    ssi_tok_map = {
        're':             {'head', 'e_type', 'r_type', 'nr_type', 'tail', 're', 'text', 'ner'},
        'boundary_re':    {'head', 'r_type', 'nr_type', 'tail', 're', 'text'},
        'boundary_joint': {'ent', 'head', 'r_type', 'nr_type', 'tail', 're', 'text'},
        'joint':          {'ent', 'e_type', 'head', 'r_type', 'nr_type', 'tail', 're', 'text', 'ner'},
    }

    def __init__(self, variant: str, use_rejection: bool = False, prompt: str = 'natural') -> None:
        self.variant = variant
        active_map = self.ssi_tok_map if prompt == 'ssi' else self.base_tok_map
        self.active_tokens = active_map.get(variant, active_map['joint']).copy()
        if use_rejection: 
            self.active_tokens.add('null')

        self._all_tokens = [self.token_strs[tok] for tok in ALL_TOKEN_NAMES if tok in self.active_tokens]

    @property
    def all_tokens(self) -> List[str]:
        return self._all_tokens


def add_special_tokens_to_tokenizer(
        tokenizer: AutoTokenizer, 
        tokens: S2GTokens, 
        model: Optional[AutoModel] = None, 
        warm_start: bool = True,
    ) -> int:
    # Add special tokens to tokenizer and optionally warm start model embeddings
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': tokens.all_tokens})
    if model is not None:
        if num_added > 0:
            model.config.tie_word_embeddings = False # weights are untied for flan-t5 models
            model.resize_token_embeddings(len(tokenizer))

        if warm_start:
            token_init_phrases = {
                'ner':      'find entities: ',
                're':       'find relations: ',
                'text':     'text: ',
                'ent':      'entity: ',
                'e_type':   'entity type: ',
                'head':     'subject: ',
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