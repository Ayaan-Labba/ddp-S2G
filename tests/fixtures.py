"""
Shared fixtures for the S2G test suite.

Instances follow the on-disk JSONL contract produced by the preprocessors, so the
tests exercise exactly what the collator and evaluator consume at runtime.
"""
from __future__ import annotations

from typing import Any, Dict, List


def entity(text: str, start: int, end: int, ent_type: str) -> Dict[str, Any]:
    return {'text': text, 'offset': [start, end], 'type': ent_type}


def relation(head: Dict, rel_type: str, tail: Dict) -> Dict[str, Any]:
    return {'head': head, 'tail': tail, 'type': rel_type}


def instance(tokens: List[str], entities: List[Dict], relations: List[Dict]) -> Dict[str, Any]:
    return {
        'text': " ".join(tokens),
        'tokens': tokens,
        'entities': entities,
        'relations': relations,
        'entity_types': sorted({e['type'] for e in entities}),
        'rel_types': sorted({r['type'] for r in relations}),
    }


def homograph_instance() -> Dict[str, Any]:
    """
    'Washington' twice with *different* types, plus a relation on each.

    Exercises failure mode 2: text-keyed merging would fuse a person and a
    location into one block and mis-assign a type.
    """
    tokens = ['Washington', 'met', 'Boeing', 'in', 'Washington', 'last', 'May']
    w_person = entity('Washington', 0, 1, 'person')
    boeing = entity('Boeing', 2, 3, 'organization')
    w_place = entity('Washington', 4, 5, 'location')
    return instance(
        tokens,
        [w_person, boeing, w_place],
        [
            relation(w_person, 'works for', boeing),
            relation(boeing, 'is based in', w_place),
        ],
    )


def duplicate_instance() -> Dict[str, Any]:
    """
    'Moscow' twice with the *same* type, each the tail of its own relation.

    Exercises failure mode 1: text-keyed scoring collapses two gold relations
    into one and caps recall below the true count.
    """
    tokens = ['Moscow', 'hosts', 'Bolshoi', 'Ballet', 'in', 'Moscow']
    moscow_a = entity('Moscow', 0, 1, 'location')
    ballet = entity('Bolshoi Ballet', 2, 4, 'organization')
    moscow_b = entity('Moscow', 5, 6, 'location')
    return instance(
        tokens,
        [moscow_a, ballet, moscow_b],
        [
            relation(ballet, 'is based in', moscow_a),
            relation(ballet, 'is based in', moscow_b),
        ],
    )


def simple_instance() -> Dict[str, Any]:
    """One relation, no repeated surface forms — the unambiguous baseline."""
    tokens = ['Boeing', 'is', 'based', 'in', 'Seattle']
    boeing = entity('Boeing', 0, 1, 'organization')
    seattle = entity('Seattle', 4, 5, 'location')
    return instance(tokens, [boeing, seattle], [relation(boeing, 'is based in', seattle)])


VARIANTS = ('joint', 'boundary_joint', 're', 'boundary_re')
TYPED_VARIANTS = ('joint', 're')

ENT_SCHEMA = ['location', 'organization', 'person']
REL_SCHEMA = ['is based in', 'works for']
