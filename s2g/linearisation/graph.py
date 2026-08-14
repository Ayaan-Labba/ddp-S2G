"""
Linearised graph construction and parsing for Sentinel Branch (with static <tail> token).
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from .special_tokens import S2GTokens, VALID_VARIANTS

EntityBlock = Dict[str, Any]
Triplet = Tuple[int, str, int]
RejectedItem = str


def organise_filter_and_block(
        entities: List, 
        relations: List, 
        allowed_ent_types: Set[str], 
        allowed_rel_types: Set[str]
    ) -> List[EntityBlock]:

    # Filter entities and relations
    filtered_ents = [e for e in entities if e['type'] in allowed_ent_types]
    valid_offsets = {tuple(e['offset']) for e in filtered_ents}
    filtered_rels = [
        r for r in relations
        if r['type'] in allowed_rel_types
        and tuple(r['head']['offset']) in valid_offsets
        and tuple(r['tail']['offset']) in valid_offsets
    ]
    
    # Sort filtered data by offset
    filtered_ents.sort(key=lambda e: e['offset'])
    filtered_rels.sort(key=lambda r: (r['head']['offset'], r['tail']['offset']))
    
    # Map tail entity offset to its position index in filtered_ents
    offset_to_idx = {tuple(e['offset']): idx for idx, e in enumerate(filtered_ents)}

    # Group sorted relations by head entity offset
    rel_groups = defaultdict(list)
    for rel in filtered_rels:
        tail_idx = offset_to_idx[tuple(rel['tail']['offset'])]
        rel_ = {'type': rel['type'], 'tail_id': tail_idx}
        rel_groups[tuple(rel['head']['offset'])].append(rel_)
        
    # Pack into entity blocks
    ent_blocks = []
    for ent in filtered_ents:
        ent_block = dict(ent)
        ent_block['relations'] = rel_groups[tuple(ent['offset'])]
        ent_blocks.append(ent_block)
        
    return ent_blocks


def build_graph(
        ent_blocks: List[EntityBlock], 
        variant: str, 
        tokens: S2GTokens, 
        use_nesting: bool = True,
        random_graph: bool = False,
        use_rejection: bool = False,
        rejected_ent_types: List[str] = None, 
        rejected_rel_types: List[str] = None
    ) -> str:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}.")

    if random_graph: 
        for old_idx, ent in enumerate(ent_blocks):
            ent['_old_idx'] = old_idx
        shuffled_blocks = random.sample(ent_blocks, len(ent_blocks))
        old_to_new = {ent['_old_idx']: new_idx for new_idx, ent in enumerate(shuffled_blocks)}
        for ent in shuffled_blocks:
            new_rels = []
            for r in ent.get('relations', []):
                new_r = dict(r)
                new_r['tail_id'] = old_to_new[r['tail_id']]
                new_rels.append(new_r)
            ent['relations'] = new_rels
            ent.pop('_old_idx', None)
        ent_blocks = shuffled_blocks

    parts = []

    if variant in {'joint', 'boundary_joint'}:
        for idx, ent in enumerate(ent_blocks):
            ent_sentinel = S2GTokens.sentinel_token(idx)
            ent_tokens = [ent_sentinel, ent['text']]
            
            if variant == 'joint' and ent.get('type'):
                ent_tokens.extend([tokens.token_strs['e_type'], ent['type']])
            
            rels = ent.get('relations', [])
            if random_graph:
                random.shuffle(rels)

            for i, rel in enumerate(rels):
                tail_idx = rel['tail_id']
                tail_sentinel = S2GTokens.sentinel_token(tail_idx)
                tail_text = ent_blocks[tail_idx]['text']
                
                rel_token = tokens.token_strs['r_type'] if (i == 0 or not use_nesting) else tokens.token_strs['nr_type']
                tail_token = tokens.token_strs['tail']
                ent_tokens.extend([rel_token, rel['type'], tail_token, tail_sentinel, tail_text])
                
            parts.append(" ".join(ent_tokens))

    elif variant in {'re', 'boundary_re'}:
        for idx, ent in enumerate(ent_blocks):
            rels = ent.get('relations', [])
            if not rels:
                continue

            if random_graph:
                random.shuffle(rels)

            ent_sentinel = S2GTokens.sentinel_token(idx)
            ent_tokens = [ent_sentinel, ent['text']]
            if variant == 're' and ent.get('type'):
                ent_tokens.extend([tokens.token_strs['e_type'], ent['type']])

            for i, rel in enumerate(rels):
                tail_idx = rel['tail_id']
                tail_sentinel = S2GTokens.sentinel_token(tail_idx)
                tail_text = ent_blocks[tail_idx]['text']
                tail_type = ent_blocks[tail_idx].get('type', '')

                rel_token = tokens.token_strs['r_type'] if (i == 0 or not use_nesting) else tokens.token_strs['nr_type']
                tail_token = tokens.token_strs['tail']
                
                if variant == 're':
                    ent_tokens.extend([
                        rel_token, rel['type'], 
                        tail_token, tail_sentinel, tail_text, 
                        tokens.token_strs['e_type'], tail_type
                    ])
                else:
                    ent_tokens.extend([rel_token, rel['type'], tail_token, tail_sentinel, tail_text])
            parts.append(" ".join(ent_tokens))

    if use_rejection:
        append_null_block(
            parts, 
            tokens, 
            ent_types=(rejected_ent_types or []) if variant in {'joint', 're'} else [],
            rel_types=rejected_rel_types or [],
            random_graph=random_graph
        )

    return " ".join(parts)


def parse_graph(text: str, tok: S2GTokens, use_nesting: bool = True) -> Tuple[List[EntityBlock], List[RejectedItem]]:
    """
    Complete state-machine parser for linearised target graphs in Sentinel Branch (with static <tail> token).
    """
    sentinel_pattern = re.compile(r'(<extra_id_\d+>|<e_type>|<r_type>|<nr_type>|<tail>|<null>)')
    raw_tokens = [t.strip() for t in sentinel_pattern.split(text) if t.strip()]

    entities: List[EntityBlock] = []
    rejected: List[RejectedItem] = []

    def get_or_create_entity(idx: int) -> EntityBlock:
        while len(entities) <= idx:
            entities.append({'text': '', 'type': None, 'relations': []})
        return entities[idx]

    current_head_idx: Optional[int] = None
    current_rel: Optional[Dict[str, Any]] = None
    current_tail_idx: Optional[int] = None
    state: str = 'IDLE'

    i = 0
    while i < len(raw_tokens):
        token = raw_tokens[i]

        if token == '<null>':
            state = 'NULL'
            i += 1
            continue

        if state == 'NULL':
            rejected.append(token)
            state = 'IDLE'
            i += 1
            continue

        match = re.match(r'<extra_id_(\d+)>', token)
        if match:
            sent_idx = int(match.group(1))

            if state in ('EXPECT_TAIL_SENTINEL', 'EXPECT_TAIL_TEXT'):
                current_tail_idx = sent_idx
                get_or_create_entity(sent_idx)
                state = 'READ_TAIL_TEXT'
            else:
                current_head_idx = sent_idx
                ent = get_or_create_entity(sent_idx)
                ent['text'] = ''
                state = 'READ_ENT_TEXT'
            i += 1
            continue

        if token == '<e_type>':
            if state in ('READ_TAIL_TEXT', 'READ_TAIL_TYPE', 'EXPECT_TAIL_TYPE'):
                state = 'READ_TAIL_TYPE'
            elif state in ('READ_ENT_TEXT', 'IDLE'):
                state = 'READ_ENT_TYPE'
            i += 1
            continue

        if token in ('<r_type>', '<nr_type>'):
            state = 'READ_REL_TYPE'
            current_rel = {'type': '', 'tail_id': None}
            i += 1
            continue

        if token == '<tail>':
            state = 'EXPECT_TAIL_SENTINEL'
            i += 1
            continue

        # Content processing
        if state == 'READ_ENT_TEXT' and current_head_idx is not None:
            ent = entities[current_head_idx]
            ent['text'] = f"{ent['text']} {token}".strip() if ent['text'] else token
        elif state == 'READ_ENT_TYPE' and current_head_idx is not None:
            entities[current_head_idx]['type'] = token
            state = 'IDLE'
        elif state == 'READ_REL_TYPE' and current_rel is not None:
            current_rel['type'] = token
            state = 'EXPECT_TAIL_SENTINEL'
        elif state == 'READ_TAIL_TEXT' and current_rel is not None:
            if current_tail_idx is not None:
                current_rel['tail_id'] = current_tail_idx
                if current_head_idx is not None:
                    entities[current_head_idx]['relations'].append(current_rel)
                if not entities[current_tail_idx]['text']:
                    entities[current_tail_idx]['text'] = token
            state = 'EXPECT_TAIL_TYPE'
        elif state == 'READ_TAIL_TYPE' and current_tail_idx is not None:
            if not entities[current_tail_idx]['type']:
                entities[current_tail_idx]['type'] = token
            state = 'IDLE'

        i += 1

    return entities, rejected


def extract_triplets(entities: List[EntityBlock], include_types: bool = False) -> List[Tuple[str, str, str]]:
    ent_map = {idx: ent for idx, ent in enumerate(entities) if ent.get('text')}
    res = []
    for h_idx, ent in enumerate(entities):
        if not ent.get('text'):
            continue
        h_text = ent.get('text', '?')
        h_type = ent.get('type', '')
        for rel in ent.get('relations', []):
            t_idx = rel.get('tail_id')
            t_ent = ent_map.get(t_idx, {})
            t_text = t_ent.get('text', '?')
            t_type = t_ent.get('type', '')

            if include_types:
                h_str = f"({h_idx}, {h_text}) [{h_type}]" if h_type else f"({h_idx}, {h_text})"
                t_str = f"({t_idx}, {t_text}) [{t_type}]" if t_type else f"({t_idx}, {t_text})"
            else:
                h_str = f"({h_idx}, {h_text})"
                t_str = f"({t_idx}, {t_text})"

            res.append((h_str, rel['type'], t_str))
    return res


def append_null_block(
        parts: List[str], 
        tok: S2GTokens, 
        ent_types: List[str], 
        rel_types: List[str], 
        random_graph: bool
    ) -> None:
    e_types = random.sample(ent_types, len(ent_types)) if random_graph else sorted(ent_types)
    r_types = random.sample(rel_types, len(rel_types)) if random_graph else sorted(rel_types)
    null_tok = tok.token_strs.get('null', '<null>')
    null_parts = [f"{null_tok} {t}" for t in e_types] + [f"{null_tok} {r}" for r in r_types]
    parts.extend(null_parts)
