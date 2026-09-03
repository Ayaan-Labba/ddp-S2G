"""
Linearised graph construction and parsing (ablation branch).

Marker style (fixed / rolling), nesting mode and inline tail types are all
emission-time settings; parsing is a single code path shared by every arm.
"""
from __future__ import annotations

import logging
import random
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from .special_tokens import MAX_MARKER_SENTINELS, S2GTokens, VALID_MARKERS, VALID_VARIANTS

logger = logging.getLogger(__name__)

EntityBlock = Dict[str, Any]
Triplet = Tuple[str, str, str]
RejectedItem = str

VALID_NESTING: Set[str] = {'nr_type', 'r_type', 'none'}
JOINT_VARIANTS: Set[str] = {'joint', 'boundary_joint'}
TYPED_VARIANTS: Set[str] = {'joint', 're'}

# Rolling block markers are sentinels; every other linearisation token is
# dedicated vocabulary. This matches markers only — role tokens are recognised by
# identity, never by pattern.
MARKER_PATTERN = re.compile(r'<extra_id_\d+>')


@lru_cache(maxsize=16)
def split_pattern(token_strs: Tuple[str, ...]) -> re.Pattern:
    """
    Pattern isolating every linearisation token during parsing.

    Built from the token strings rather than hard-coded, because they are no longer
    all of one shape: role tokens are dedicated vocabulary and markers are
    sentinels. Longest-first alternation so that no token can be shadowed by
    another that happens to be its prefix.

    Deliberately given *all* the role tokens, not just the active ones. An inactive
    token appearing in a generation is malformed either way, but isolating it keeps
    it out of the entity text, where it would silently corrupt a mention.
    """
    alternation = "|".join(re.escape(t) for t in sorted(token_strs, key=len, reverse=True))
    return re.compile(f"({alternation}|{MARKER_PATTERN.pattern})")


def organise_filter_and_block(
        entities: List,
        relations: List,
        allowed_ent_types: Set[str],
        allowed_rel_types: Set[str],
        variant: str = 'joint',
        use_types: bool = True,
        dedup: bool = True
    ) -> List[EntityBlock]:

    # 1. Filter entities and relations
    filtered_ents = [e for e in entities if e['type'] in allowed_ent_types] if use_types else list(entities)
    valid_offsets = {tuple(e['offset']) for e in filtered_ents}
    filtered_rels = [
        r for r in relations
        if r['type'] in allowed_rel_types
        and tuple(r['head']['offset']) in valid_offsets
        and tuple(r['tail']['offset']) in valid_offsets
    ]

    # 2. Sort filtered data by offset
    filtered_ents.sort(key=lambda e: e['offset'])
    filtered_rels.sort(key=lambda r: (r['head']['offset'], r['tail']['offset']))

    # 3. Select the entities that are entitled to a block: the joint variants emit
    # every entity, the RE variants only those heading at least one relation.
    if variant in JOINT_VARIANTS:
        block_ents = filtered_ents
    else:
        head_offsets = {tuple(r['head']['offset']) for r in filtered_rels}
        block_ents = [e for e in filtered_ents if tuple(e['offset']) in head_offsets]

    # 4. Build blocks. Without ``dedup`` every mention keeps its own block; otherwise
    # mentions collapse on (text, type), so genuine homographs stay separate.
    offset_to_ent: Dict[Tuple[int, int], EntityBlock] = {}
    blocks: List[EntityBlock] = []
    key_to_ent: Dict[Tuple[str, Optional[str]], EntityBlock] = {}

    for ent in block_ents:
        ent_type = ent.get('type') if use_types else None
        block_key = (ent['text'], ent_type)
        block = key_to_ent.get(block_key) if dedup else None

        if block is None:
            block = {
                'text': ent['text'],
                'type': ent_type,
                'offset': ent['offset'],
                'relations': []
            }
            blocks.append(block)
            if dedup:
                key_to_ent[block_key] = block

        offset_to_ent[tuple(ent['offset'])] = block

    # 5. Attach relations to their head block.
    seen_rels: Set[Tuple] = set()
    for rel in filtered_rels:
        head_block = offset_to_ent[tuple(rel['head']['offset'])]
        tail_text = rel['tail']['text']
        tail_type = rel['tail'].get('type') if use_types else None
        rel_type = rel['type']

        if dedup:
            rel_key = (head_block['text'], head_block['type'], rel_type, tail_text, tail_type)
            if rel_key in seen_rels:
                continue
            seen_rels.add(rel_key)

        head_block['relations'].append({
            'type': rel_type,
            'tail_text': tail_text,
            'tail_type': tail_type
        })

    return blocks


def max_emitted_blocks(markers: str, use_rejection: bool = False) -> Optional[int]:
    """
    Ceiling on emitted blocks, or ``None`` when uncapped.

    Fixed markers reuse one token, so there is no ceiling. Rolling markers spend
    one sentinel per block, the first included — an n-block graph uses
    ``<extra_id_0>`` .. ``<extra_id_{n-1}>`` — so the full range of 100 sentinels
    allows 100 blocks. Rejection claims one further index for its own marker
    (Stage 3), leaving 99.
    """
    if markers != 'rolling':
        return None
    return MAX_MARKER_SENTINELS - 1 if use_rejection else MAX_MARKER_SENTINELS


def marker_token(block_idx: int, tokens: S2GTokens, markers: str) -> str:
    """
    Marker opening block ``block_idx``.

    Every block carries one, the first included: a single-block graph opens with a
    marker and an n-block graph carries exactly n. Fixed markers reuse ``<ent>``
    throughout; rolling markers count up from ``<extra_id_0>``, so block *i* is
    opened by ``<extra_id_i>``.
    """
    return tokens.token_strs['ent'] if markers == 'fixed' else tokens.sentinel_token(block_idx)


def build_graph(
        ent_blocks: List[EntityBlock],
        variant: str,
        tokens: S2GTokens,
        nesting: str = 'nr_type',
        markers: str = 'fixed',
        joint_tail_type: bool = False,
        random_graph: bool = False,
        use_rejection: bool = False,
        rejected_ent_types: List[str] = None,
        rejected_rel_types: List[str] = None
    ) -> str:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}.")
    if nesting not in VALID_NESTING:
        raise ValueError(f"Unknown nesting mode {nesting!r}; expected one of {VALID_NESTING}.")
    if markers not in VALID_MARKERS:
        raise ValueError(f"Unknown marker style {markers!r}; expected one of {VALID_MARKERS}.")

    if random_graph and ent_blocks:
        ent_blocks = random.sample(ent_blocks, len(ent_blocks))

    # Pair each block with its relation order up front, so the ``none`` expansion
    # below splits an already-ordered list rather than re-shuffling per block.
    ordered: List[Tuple[EntityBlock, List[Dict[str, Any]]]] = []
    for ent in ent_blocks:
        rels = list(ent.get('relations') or [])
        if random_graph and rels:
            rels = random.sample(rels, len(rels))
        ordered.append((ent, rels))

    # Only emitted blocks consume a marker: the joint variants emit every entity,
    # the RE variants only those heading at least one relation. Capping the
    # candidate list instead would under-fill the RE targets, dropping heads that
    # would have fitted once the relation-less entities were skipped.
    emit = ordered if variant in JOINT_VARIANTS else [(e, r) for e, r in ordered if r]

    if nesting == 'none':
        # One relation per block: a k-relation head becomes k blocks with its
        # mention and type repeated. Block *grouping* is untouched — this is an
        # emission-time split, not a ``dedup`` change.
        expanded: List[Tuple[EntityBlock, List[Dict[str, Any]]]] = []
        for ent, rels in emit:
            if rels:
                expanded.extend((ent, [rel]) for rel in rels)
            else:
                expanded.append((ent, []))
        emit = expanded

    cap = max_emitted_blocks(markers, use_rejection)
    if cap is not None and len(emit) > cap:
        logger.warning(
            "Truncating %d entity blocks to %d: no rolling marker exists beyond <extra_id_%d>.",
            len(emit), cap, cap - 1,
        )
        emit = emit[:cap]

    emits_ent_type = variant in TYPED_VARIANTS
    emits_tail_type = variant == 're' or (variant == 'joint' and joint_tail_type)

    parts = []
    for block_idx, (ent, rels) in enumerate(emit):
        ent_toks = [marker_token(block_idx, tokens, markers), ent['text']]
        if emits_ent_type and ent.get('type'):
            ent_toks.extend([tokens.token_strs['e_type'], ent['type']])

        for i, rel in enumerate(rels):
            rel_token = (
                tokens.token_strs['nr_type'] if (nesting == 'nr_type' and i > 0)
                else tokens.token_strs['r_type']
            )
            ent_toks.extend([rel_token, rel['type'], tokens.token_strs['tail'], rel['tail_text']])
            if emits_tail_type and (tail_type := rel.get('tail_type')):
                ent_toks.extend([tokens.token_strs['e_type'], tail_type])

        parts.append(" ".join(ent_toks))

    if use_rejection:
        append_null_block(
            parts,
            tokens,
            ent_types=(rejected_ent_types or []) if variant in TYPED_VARIANTS else [],
            rel_types=rejected_rel_types or [],
            random_graph=random_graph
        )

    return " ".join(parts).strip()


def parse_graph(text: str, tok: S2GTokens) -> Tuple[List[EntityBlock], List[RejectedItem]]:
    """
    State-machine parser for nested linearised target graphs.

    Blocks are allocated by **append, in emission order**, in every arm: a rolling
    marker's index is read as a separator and then discarded, which keeps the fixed
    and rolling arms on one code path and makes a repeated index harmless.

    Parsing never deduplicates: every emitted block is retained, so repeated
    mentions and repeated relations survive into scoring exactly as generated.
    """
    pattern = split_pattern(tuple(tok.token_strs.values()))
    raw_tokens = [t.strip() for t in pattern.split(text) if t.strip()]

    role_tokens = tok.role_token_strs
    ent_token = tok.token_strs['ent']
    e_type_token = tok.token_strs['e_type']
    r_type_token = tok.token_strs['r_type']
    nr_type_token = tok.token_strs['nr_type']
    tail_token = tok.token_strs['tail']
    null_token = tok.token_strs['null']

    # Seeded so that content appearing before any marker still lands somewhere —
    # a malformed generation that omits the leading marker, or an earlier format in
    # which the first block carried none. Dropped again at the end if it never
    # received any text, which is the normal case now that every block is marked.
    entities: List[EntityBlock] = [{'text': '', 'type': None, 'relations': []}]
    rejected: List[RejectedItem] = []

    current_head_idx: Optional[int] = 0
    current_rel: Optional[Dict[str, Any]] = None
    state: str = 'READ_ENT_TEXT'

    def flush_rel():
        nonlocal current_rel
        if current_rel is not None and current_head_idx is not None and current_head_idx < len(entities):
            if current_rel.get('type') and current_rel.get('tail_text'):
                entities[current_head_idx]['relations'].append(current_rel)
            current_rel = None

    i = 0
    while i < len(raw_tokens):
        token = raw_tokens[i]

        # Role tokens are matched by exact identity before anything is treated as a
        # separator, so a role sentinel can never be read as a marker.
        if token in role_tokens:
            if token == null_token:
                flush_rel()
                state = 'NULL'
                i += 1
                continue

            if token == e_type_token:
                if state in ('READ_TAIL_TEXT', 'READ_TAIL_TYPE'):
                    state = 'READ_TAIL_TYPE'
                else:
                    state = 'READ_ENT_TYPE'
                i += 1
                continue

            if token in (r_type_token, nr_type_token):
                flush_rel()
                state = 'READ_REL_TYPE'
                current_rel = {'type': '', 'tail_text': '', 'tail_type': None}
                i += 1
                continue

            if token == tail_token:
                state = 'READ_TAIL_TEXT'
                i += 1
                continue

        if state == 'NULL':
            rejected.append(token)
            state = 'IDLE'
            i += 1
            continue

        if token == ent_token or MARKER_PATTERN.fullmatch(token):
            # The fixed marker, or any rolling sentinel. A rolling index is read and
            # then discarded: blocks are appended, so a repeated index cannot
            # overwrite a block.
            flush_rel()
            entities.append({'text': '', 'type': None, 'relations': []})
            current_head_idx = len(entities) - 1
            state = 'READ_ENT_TEXT'
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
            if token.strip().lower() == 'none' and not current_rel['type']:
                current_rel = None
                state = 'IDLE'
            else:
                current_rel['type'] = f"{current_rel['type']} {token}".strip() if current_rel['type'] else token
        elif state == 'READ_TAIL_TEXT' and current_rel is not None:
            current_rel['tail_text'] = f"{current_rel['tail_text']} {token}".strip() if current_rel['tail_text'] else token
        elif state == 'READ_TAIL_TYPE' and current_rel is not None:
            current_rel['tail_type'] = token
            state = 'IDLE'

        i += 1

    flush_rel()

    return resolve_tail_entities([e for e in entities if e.get('text')]), rejected


def resolve_tail_entities(entities: List[EntityBlock]) -> List[EntityBlock]:
    """
    Reconcile relation tails against the entity blocks, in place.

    A tail mention resolves to the *first* block carrying that text: without inline
    tail types the type has to be recovered from the entity's own block, and
    duplicated mentions must resolve deterministically. Tails with no block of
    their own are appended so they still count as entities.

    Shared by ``parse_graph`` and gold-block construction so that both sides of a
    comparison are reconciled identically.
    """
    ent_by_text: Dict[str, EntityBlock] = {}
    for ent in entities:
        ent_by_text.setdefault(ent['text'].strip(), ent)

    for ent in list(entities):
        for rel in ent.get('relations', []):
            t_text = (rel.get('tail_text') or '').strip()
            if not t_text:
                continue

            match = ent_by_text.get(t_text)
            if match is None:
                # Tail that never appeared as a block of its own: keep it so it
                # still counts towards entity recall.
                new_ent: EntityBlock = {'text': t_text, 'type': rel.get('tail_type'), 'relations': []}
                entities.append(new_ent)
                ent_by_text[t_text] = new_ent
            elif not rel.get('tail_type') and match.get('type'):
                rel['tail_type'] = match['type']
            elif not match.get('type') and rel.get('tail_type'):
                match['type'] = rel.get('tail_type')

    return entities


def extract_triplets(entities: List[EntityBlock], include_types: bool = False) -> List[Tuple[str, str, str]]:
    ent_map = {ent['text'].strip(): ent for ent in entities if ent.get('text')}
    res = []
    for ent in entities:
        if not ent.get('text'):
            continue
        h_text = ent.get('text', '?')
        h_type = ent.get('type', '')
        for rel in ent.get('relations', []):
            t_text = rel.get('tail_text', '?')
            t_ent = ent_map.get(t_text, {})
            t_type = rel.get('tail_type') or t_ent.get('type', '')

            if include_types:
                h_str = f"{h_text} [{h_type}]" if h_type else h_text
                t_str = f"{t_text} [{t_type}]" if t_type else t_text
            else:
                h_str = h_text
                t_str = t_text

            res.append((h_str, rel['type'], t_str))
    return res


def append_null_block(
        parts: List[str],
        tok: S2GTokens,
        ent_types: List[str],
        rel_types: List[str],
        random_graph: bool
    ) -> None:
    """
    Per-type rejection markers, ``<null> type`` for each sampled negative.

    Stage 3 replaces this with the single-marker CoT rejection tail; until then
    rejection is only meaningful under fixed markers, where ``null`` is an active
    role token.
    """
    e_types = random.sample(ent_types, len(ent_types)) if random_graph else sorted(ent_types)
    r_types = random.sample(rel_types, len(rel_types)) if random_graph else sorted(rel_types)
    null_tok = tok.token_strs['null']
    null_parts = [f"{null_tok} {t}" for t in e_types] + [f"{null_tok} {r}" for r in r_types]
    parts.extend(null_parts)
