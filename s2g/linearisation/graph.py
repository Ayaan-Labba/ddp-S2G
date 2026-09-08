"""
Linearised graph construction and parsing (ablation branch).

Nesting mode and inline tail types are emission-time settings; parsing is a
single code path shared by every arm.
"""
from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .prompt import order_types
from .special_tokens import MAX_MARKER_SENTINELS, S2GTokens, VALID_VARIANTS

logger = logging.getLogger(__name__)

EntityBlock = Dict[str, Any]
Triplet = Tuple[str, str, str]
RejectedItem = str

VALID_NESTING: Set[str] = {'nr_type', 'r_type', 'none'}

# Every linearisation token is a sentinel, so one pattern isolates them all during
# parsing. Which of them are *roles* is decided by identity, never by pattern.
SENTINEL_PATTERN = re.compile(r'(<extra_id_\d+>)')

# The rejection tail, as emitted by ``append_rejection_tail``. The entity clause is
# optional because ``boundary_joint`` deals in no entity types and says so by
# omitting it, exactly as its prompt does.
REJECTION_TAIL = re.compile(
    r'missing\s+(?:entities\s*\[(?P<ents>[^\]]*)\]\s*and\s+)?relations\s*\[(?P<rels>[^\]]*)\]',
    re.IGNORECASE,
)


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

    # 3. Build blocks. Every entity is entitled to one, relation-less included —
    # the RE variants, which gave a block only to relation heads, are retired.
    # Without ``dedup`` every mention keeps its own block; otherwise mentions
    # collapse on (text, type), so genuine homographs stay separate.
    offset_to_ent: Dict[Tuple[int, int], EntityBlock] = {}
    blocks: List[EntityBlock] = []
    key_to_ent: Dict[Tuple[str, Optional[str]], EntityBlock] = {}

    for ent in filtered_ents:
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

    # 4. Attach relations to their head block.
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


def max_emitted_blocks() -> int:
    """
    Ceiling on emitted blocks.

    Markers spend one sentinel per block, the first included — an n-block graph
    uses ``<extra_id_0>`` .. ``<extra_id_{n-1}>`` — so the 94 sentinels below the
    roles allow 94 blocks. Rejection costs nothing here: ``<null>`` is a dedicated
    role, not a rolling index, so it claims no marker.
    """
    return MAX_MARKER_SENTINELS


def build_graph(
        ent_blocks: List[EntityBlock],
        variant: str,
        tokens: S2GTokens,
        nesting: str = 'nr_type',
        random_graph: bool = False,
        random_prompt: bool = False,
        use_rejection: bool = False,
        rejected_ent_types: List[str] = None,
        rejected_rel_types: List[str] = None
    ) -> str:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}.")
    if nesting not in VALID_NESTING:
        raise ValueError(f"Unknown nesting mode {nesting!r}; expected one of {VALID_NESTING}.")

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
    emit = ordered

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

    cap = max_emitted_blocks()
    if len(emit) > cap:
        logger.warning(
            "Truncating %d entity blocks to %d: no block marker exists beyond <extra_id_%d>.",
            len(emit), cap, cap - 1,
        )
        emit = emit[:cap]

    # Axis 2 settled both: ``joint`` carries entity types and inline tail types,
    # and the boundary variant carries neither. Neither is configurable any more.
    emits_types = variant == 'joint'

    parts = []
    for block_idx, (ent, rels) in enumerate(emit):
        ent_toks = [tokens.sentinel_token(block_idx), ent['text']]
        if emits_types and ent.get('type'):
            ent_toks.extend([tokens.token_strs['e_type'], ent['type']])

        for i, rel in enumerate(rels):
            rel_token = (
                tokens.token_strs['nr_type'] if (nesting == 'nr_type' and i > 0)
                else tokens.token_strs['r_type']
            )
            ent_toks.extend([rel_token, rel['type'], tokens.token_strs['tail'], rel['tail_text']])
            if emits_types and (tail_type := rel.get('tail_type')):
                ent_toks.extend([tokens.token_strs['e_type'], tail_type])

        # Closes a block that heads nothing, stating the absence rather than leaving
        # it to be inferred from the next marker arriving early.
        if not rels:
            ent_toks.append(tokens.token_strs['no_rel'])

        parts.append(" ".join(ent_toks))

    if use_rejection:
        append_rejection_tail(
            parts,
            tokens,
            ent_types=(rejected_ent_types or []) if emits_types else None,
            rel_types=rejected_rel_types or [],
            random_prompt=random_prompt,
        )

    return " ".join(parts).strip()


def parse_graph(text: str, tok: S2GTokens) -> Tuple[List[EntityBlock], List[RejectedItem]]:
    """
    State-machine parser for nested linearised target graphs.

    Blocks are allocated by **append, in emission order**: a marker's index is read
    as a separator and then discarded, which makes a repeated or out-of-order index
    harmless rather than destructive.

    Parsing never deduplicates: every emitted block is retained, so repeated
    mentions and repeated relations survive into scoring exactly as generated.
    """
    raw_tokens = [t.strip() for t in SENTINEL_PATTERN.split(text) if t.strip()]

    role_tokens = tok.role_token_strs
    e_type_token = tok.token_strs['e_type']
    r_type_token = tok.token_strs['r_type']
    nr_type_token = tok.token_strs['nr_type']
    tail_token = tok.token_strs['tail']
    null_token = tok.token_strs['null']
    no_rel_token = tok.token_strs['no_rel']

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

            if token == no_rel_token:
                # Carries no content: the block simply has no relations, which an
                # empty list already says. The branch still earns its place. Without
                # it the token falls through to the marker test below and — being a
                # sentinel — opens a block; harmless on a well-formed target, where
                # that block stays empty and is filtered out, but on a malformed
                # generation any text following the marker would be credited as a
                # hallucinated entity. IDLE drops it instead.
                flush_rel()
                state = 'IDLE'
                i += 1
                continue

        if state == 'NULL':
            # Everything from ``<null>`` on is the rejection tail. Consume it whole:
            # nothing after the marker may reach an entity or relation block, and a
            # tail that does not match is dropped rather than leaking into one.
            match = REJECTION_TAIL.search(" ".join(raw_tokens[i:]))
            if match:
                for group in ('ents', 'rels'):
                    listed = match.group(group)
                    rejected.extend(t.strip() for t in (listed or '').split(',') if t.strip())
            break

        if SENTINEL_PATTERN.fullmatch(token):
            # Any sentinel that is not an active role is a block marker. Its index is
            # read and then discarded: blocks are appended, so a repeated index
            # cannot overwrite a block.
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


def append_rejection_tail(
        parts: List[str],
        tok: S2GTokens,
        ent_types: Optional[List[str]],
        rel_types: List[str],
        random_prompt: bool = False,
    ) -> None:
    """
    The Axis-3 rejection tail: one ``<null>``, then the sampled negatives in prose.

    ``ent_types=None`` drops the entity clause entirely — ``boundary_joint`` deals
    in no entity types, and its prompt asks only for missing relations, so its tail
    answers only about relations.

    Ordering follows ``random_prompt``, not ``random_graph``: the tail restates the
    prompt's negatives, so it lists them the way the prompt did. It shares
    ``order_types`` with the prompt builder rather than reimplementing the rule.

    Empty lists still emit their brackets — the absence of negatives is itself the
    answer, and a target that simply stopped would be indistinguishable from one
    that was truncated.
    """
    r_types = ", ".join(order_types(rel_types, random_prompt))
    tail = f"relations [{r_types}]"

    if ent_types is not None:
        e_types = ", ".join(order_types(ent_types, random_prompt))
        tail = f"entities [{e_types}] and {tail}"

    parts.append(f"{tok.token_strs['null']} Therefore, missing {tail}")
