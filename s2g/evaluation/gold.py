"""
Gold construction for evaluation.

Gold is taken straight from the preprocessed instance rather than by parsing the
collated labels.  Parsing the labels made gold a round trip through
``build_graph`` -> ``parse_graph``, which silently inherited every lossy step of
that path: targets truncated at ``max_target_length`` lost their tail, and joint
tail types were recovered by surface matching instead of being read off the
annotation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from s2g.linearisation import EntityBlock, organise_filter_and_block, resolve_tail_entities

# (triplets, quintuples, entities, mentions) for one instance
OffsetBundle = Tuple[List[Tuple], List[Tuple], List[Tuple], List[Tuple]]

TYPED_VARIANTS = {'joint', 're'}
HEAD_ONLY_VARIANTS = {'re', 'boundary_re'}


def build_gold_blocks(instance: Dict[str, Any], variant: str, dedup: bool = True) -> List[EntityBlock]:
    """
    Text-side gold blocks, structurally identical to what the collator would
    linearise for this instance.

    The instance's own types are passed as the allowed schema, which reproduces
    ``budget`` sampling exactly: budget keeps every positive and only pads the
    prompt with negatives, so nothing in the gold graph is ever filtered out.
    Evaluation always runs in budget mode (``S2GCollator.to_eval_mode``).
    """
    use_types = variant in TYPED_VARIANTS
    blocks = organise_filter_and_block(
        instance.get('entities', []),
        instance.get('relations', []),
        set(instance.get('entity_types', [])),
        set(instance.get('rel_types', [])),
        variant=variant,
        use_types=use_types,
        dedup=dedup,
    )
    # Predicted blocks are reconciled by ``parse_graph``; gold must be reconciled
    # the same way or the RE variants would score tail mentions as pure precision
    # errors, since only heads get a block of their own there.
    return resolve_tail_entities(blocks)


def build_gold_offsets(instance: Dict[str, Any], variant: str) -> OffsetBundle:
    """
    Offset-side gold, read directly off the annotation.

    Independent of ``dedup``: a deduplicated block keeps only its first offset,
    so offset gold cannot be derived from blocks without losing repeated
    mentions — which is the whole point of scoring on offsets.
    """
    use_types = variant in TYPED_VARIANTS
    entities = instance.get('entities', [])
    relations = instance.get('relations', [])

    if variant in HEAD_ONLY_VARIANTS:
        # These variants only ask for relation participants; scoring against every
        # annotated entity would cap recall at something never trained for.
        participants = {tuple(r['head']['offset']) for r in relations}
        participants |= {tuple(r['tail']['offset']) for r in relations}
        entities = [e for e in entities if tuple(e['offset']) in participants]

    ent_offsets = [tuple(e['offset']) for e in entities]
    mentions = [(tuple(e['offset']), e['type']) for e in entities] if use_types else []

    triplets = [
        (tuple(r['head']['offset']), r['type'], tuple(r['tail']['offset']))
        for r in relations
    ]
    quintuples = [
        (
            tuple(r['head']['offset']), r['head']['type'],
            r['type'],
            tuple(r['tail']['offset']), r['tail']['type'],
        )
        for r in relations
    ] if use_types else []

    return triplets, quintuples, ent_offsets, mentions
