"""
Linearised graph construction and parsing.
"""
from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict

from .special_tokens import S2GTokens


EntityBlock = Tuple[str, Any]
Triplet = Tuple[str, str, str]
RejectedItem = Dict[str, Any]


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
    
    # Sort filtered data
    filtered_ents.sort(key=lambda e: e['offset'])
    filtered_rels.sort(key=lambda r: (r['head']['offset'], r['tail']['offset']))
    
    # Group sorted relations by head entity
    rel_groups = defaultdict(list)
    for rel in filtered_rels:
        rel_ = {'type': rel['type'], 'tail': rel['tail']['text'], 'tail_type': rel['tail']['type']}
        rel_groups[tuple(rel['head']['offset'])].append(rel_)
        
    # Pack into entity blocks
    ent_blocks = []
    for ent in filtered_ents:
        ent['relations'] = rel_groups[tuple(ent['offset'])]
        ent_blocks.append(ent)
        
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
    if variant not in {'re', 'boundary_re', 'boundary_joint', 'joint'}:
        raise ValueError(f"Unknown variant {variant!r}.")
    
    if random_graph: 
        random.shuffle(ent_blocks)

    if rejected_ent_types is None:
        rejected_ent_types = []
    
    if rejected_rel_types is None:
        rejected_rel_types = []
    
    if variant in {'re', 'boundary_re'}:
        parts = []
        extract_parts = []

        for ent in ent_blocks:
            rels = ent['relations']
            if not rels:
                continue

            if random_graph: 
                random.shuffle(rels)

            for i, rel in enumerate(rels):
                if variant == 're':
                    if i == 0 or not use_nesting:
                        extract_parts.extend([
                            tokens.token_strs['head'], ent['text'], 
                            tokens.token_strs['e_type'], ent.get('type'), 
                            tokens.token_strs['r_type'], rel['type'], 
                            tokens.token_strs['tail'], rel['tail'], 
                            tokens.token_strs['e_type'], rel.get('tail_type')
                        ])
                    else:
                        extract_parts.extend([
                            tokens.token_strs['nr_type'], rel['type'], 
                            tokens.token_strs['tail'], rel['tail'], 
                            tokens.token_strs['e_type'], rel.get('tail_type')
                        ])
                else:  # boundary_re
                    if i == 0 or not use_nesting:
                        extract_parts.extend([
                            tokens.token_strs['head'], ent['text'], 
                            tokens.token_strs['r_type'], rel['type'], 
                            tokens.token_strs['tail'], rel['tail']
                        ])
                    else:
                        extract_parts.extend([
                            tokens.token_strs['nr_type'], rel['type'], 
                            tokens.token_strs['tail'], rel['tail']
                        ])

        if extract_parts:
            parts.append(" ".join(extract_parts))

        if use_rejection:
            append_null_block(
                parts, 
                tokens, 
                ent_types=rejected_ent_types if variant=='re' else [],
                rel_types=rejected_rel_types,
                random_graph=random_graph
            )
        
        return " ".join(parts)

    if variant in {'joint', 'boundary_joint'}:
        parts = []
        ent_parts = []
        for ent in ent_blocks:
            if variant == 'joint':
                ent_parts.extend([
                    tokens.token_strs['ent'], ent['text'], 
                    tokens.token_strs['e_type'], ent.get('type')
                ])
            else:  # boundary_joint
                ent_parts.extend([tokens.token_strs['ent'], ent['text']])
        
        if ent_parts:
            parts.append(" ".join(ent_parts))
            triplet_parts = []
            for ent in ent_blocks:
                rels = ent['relations']
                if not rels:
                    continue

                if random_graph: 
                    random.shuffle(rels)

                ent_triplet = []
                for i, rel in enumerate(rels):
                    if i == 0 or not use_nesting:
                        ent_triplet.extend([
                            tokens.token_strs['head'], ent['text'], 
                            tokens.token_strs['r_type'], rel['type'], 
                            tokens.token_strs['tail'], rel['tail']
                        ])
                    else:
                        ent_triplet.extend([
                            tokens.token_strs['nr_type'], rel['type'], 
                            tokens.token_strs['tail'], rel['tail']
                        ])
                
                triplet_parts.append(" ".join(ent_triplet))

            if triplet_parts:
                parts.append(" ".join(triplet_parts))

            if use_rejection:
                append_null_block(
                    parts, 
                    tokens, 
                    ent_types=rejected_ent_types if variant == 'joint' else [],
                    rel_types=rejected_rel_types,
                    random_graph=random_graph
                )
            
            return " ".join(parts)

    return ""


@lru_cache(maxsize=16)
def _get_compiled_special_token_pattern(tokens_tuple: Tuple[str, ...]) -> re.Pattern:
    special_tokens = sorted(tokens_tuple, key=len, reverse=True)
    return re.compile(f"({'|'.join(map(re.escape, special_tokens))})")


def parse_graph(text: str, tok: S2GTokens, use_nesting: bool = True) -> Tuple[List, List[RejectedItem]]:
    pattern = _get_compiled_special_token_pattern(tuple(tok.all_tokens))
    tokens = [t.strip() for t in pattern.split(text) if t.strip()]
    
    entities: List[EntityBlock] = []
    entity_dict: Dict[str, EntityBlock] = {}
    rejected: List[RejectedItem] = []
    current_head_text = []
    current_rel = []
    current_tail_text = []
    current_reject = []

    if tok.variant in {'joint', 'boundary_joint'}:
        state = 'IDLE'
        current_ent_text = []
        current_ent_type = []

        def flush_current_state():
            nonlocal state, current_ent_text, current_ent_type, current_head_text, current_rel, current_tail_text, \
                current_reject, entities, entity_dict, rejected
            
            if state == 'ENT_TEXT' or state == 'ENT_TYPE':
                ent_text = " ".join(current_ent_text).strip()
                current_ent_text.clear()                
                if ent_text not in entity_dict:
                    block = {'text': ent_text, 'relations': []}
                    if tok.variant == 'joint': 
                        ent_type = " ".join(current_ent_type).strip()
                        current_ent_type.clear()
                        block['type'] = ent_type
                    
                    entities.append(block)
                    entity_dict[ent_text] = block 
            
            elif state == 'TAIL':
                head_text = " ".join(current_head_text).strip()
                rel_type = " ".join(current_rel).strip()
                current_rel.clear()
                tail_text = " ".join(current_tail_text).strip()
                current_tail_text.clear()                 
                rel = {'type': rel_type, 'tail': tail_text}
                if tok.variant == 'joint': 
                    tail_ent = entity_dict.get(tail_text)
                    if tail_ent:
                        rel['tail_type'] = tail_ent.get('type', '?')

                if head_text in entity_dict:
                    entity_dict[head_text]['relations'].append(rel)

            elif state == 'NULL':
                label_str = " ".join(current_reject).strip()
                if label_str:
                    rejected.append(label_str)
                
                current_reject.clear()

        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == tok.token_strs['ent']:
                flush_current_state()
                state = 'ENT_TEXT'
            
            elif t == tok.token_strs['e_type']:
                state = 'ENT_TYPE'
            
            elif t == tok.token_strs['head']:
                flush_current_state()
                state = 'HEAD'
                current_head_text.clear()

            elif t == tok.token_strs['r_type']:
                state = 'REL'

            elif t == tok.token_strs['tail']:
                state = 'TAIL'

            elif t == tok.token_strs['nr_type']:
                flush_current_state()
                state = 'REL'
            
            elif t == tok.token_strs['null']:
                flush_current_state()
                state = 'NULL'
            
            else:
                if state == 'ENT_TEXT':
                    current_ent_text.append(t)
                elif state == 'ENT_TYPE':
                    current_ent_type.append(t)
                elif state == 'HEAD':
                    current_head_text.append(t)
                elif state == 'REL':
                    current_rel.append(t)
                elif state == 'TAIL':
                    current_tail_text.append(t)
                elif state == 'NULL':
                    current_reject.append(t)
                
            i += 1
        
        flush_current_state()
        
        return deduplicate_entities(entities), rejected

    if tok.variant in {'re', 'boundary_re'}:        
        state = 'IDLE'
        current_head_type = []
        current_tail_type = []
        current_entity_block = None 
        
        def flush_triplet():
            nonlocal current_head_text, current_head_type, current_rel, current_tail_text, current_tail_type, \
                current_entity_block, entities, rejected
            
            h_txt = " ".join(current_head_text).strip()
            r_typ = " ".join(current_rel).strip()
            t_txt = " ".join(current_tail_text).strip()
            if tok.variant == 're':
                h_typ = " ".join(current_head_type).strip()
                t_typ = " ".join(current_tail_type).strip()
            
            if use_nesting:
                if current_entity_block is None:
                    current_entity_block = {'text': h_txt, 'relations': []}
                    if tok.variant == 're': current_entity_block['type'] = h_typ
                    entities.append(current_entity_block)
                
                rel = {'type': r_typ, 'tail': t_txt}
                if tok.variant == 're': rel['tail_type'] = t_typ
                current_entity_block['relations'].append(rel)
            
            else:
                ent = {'text': h_txt, 'type': h_typ, 'relations': []}
                rel = {'type': r_typ, 'tail': t_txt}
                if tok.variant == 're':
                    ent['type'] = h_typ
                    rel['tail_type'] = t_typ
                
                entities.append(ent)
            
            current_rel.clear()
            current_tail_text.clear()
            current_tail_type.clear()
            
        def flush_null():
            nonlocal current_reject, rejected
            if current_reject:
                lbl = " ".join(current_reject).strip()
                if lbl:
                    rejected.append(lbl)
                
                current_reject.clear()

        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == tok.token_strs['head']:
                if not state == 'IDLE': flush_triplet()
                current_head_text.clear()
                current_head_type.clear()
                current_entity_block = None
                state = 'HEAD_TEXT'
            
            elif t == tok.token_strs['e_type']:
                if state == 'HEAD_TEXT':
                    state = 'HEAD_TYPE'
                
                elif state == 'TAIL_TEXT':
                    state = 'TAIL_TYPE'
            
            elif t == tok.token_strs['r_type']:
                state = 'REL'
            
            elif t == tok.token_strs['tail']:
                state = 'TAIL_TEXT'

            elif t == tok.token_strs['nr_type']:
                flush_triplet()
                state = 'REL'
            
            elif t == tok.token_strs['null']:
                if state != 'NULL': 
                    flush_triplet()
                    state = 'NULL'
                
                else:
                    flush_null()
            
            else:
                if state == 'HEAD_TEXT':
                    current_head_text.append(t)
                elif state == 'HEAD_TYPE':
                    current_head_type.append(t)
                elif state == 'REL':
                    current_rel.append(t)
                elif state == 'TAIL_TEXT':
                    current_tail_text.append(t)
                elif state == 'TAIL_TYPE':
                    current_tail_type.append(t)
                elif state == 'NULL':
                    current_reject.append(t)
            
            i += 1
            
        flush_triplet()
        flush_null()
        
        return entities, rejected

    return [], []


def extract_triplets(entities: List[EntityBlock], include_types: bool = False) -> List[Triplet]:
    if include_types:
        return [(
            f"{ent['text']} [{ent.get('type')}]", 
            rel["type"], 
            f"{rel['tail']} [{rel.get('tail_type')}]"
        ) for ent in entities for rel in ent["relations"]]
    
    return [(ent["text"], rel["type"], rel["tail"]) for ent in entities for rel in ent["relations"]]


def append_null_block(
        parts: List[str], 
        tok: S2GTokens, 
        ent_types: List[str], 
        rel_types: List[str], 
        random_graph: bool
    ) -> None:
    e_types = random.sample(ent_types, len(ent_types)) if random_graph else sorted(ent_types)
    r_types = random.sample(rel_types, len(rel_types)) if random_graph else sorted(rel_types)
    null_parts = [f"{tok.token_strs['null']} {t}" for t in e_types] + [f"{tok.token_strs['null']} {r}" for r in r_types]
    parts.extend(null_parts)


def deduplicate_entities(entities: List[EntityBlock]) -> List[EntityBlock]:
    seen, deduped = {}, []
    for ent in entities:
        text_key = ent["text"]
        if text_key in seen:
            deduped[seen[text_key]]["relations"].extend(ent["relations"])
        else:
            seen[text_key] = len(deduped)
            deduped.append(ent)
    return deduped

