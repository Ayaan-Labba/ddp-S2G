"""
Offset projection for predicted graphs.

The decoder emits surface text, so a prediction has to be located in the source
token sequence before it can be scored against offset annotations.  Every match
counts as a distinct predicted mention, and a relation is duplicated across the
cross product of its head and tail matches.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

Offset = Tuple[int, int]
# (triplets, quintuples, entities, mentions) for one instance
OffsetBundle = Tuple[List[Tuple], List[Tuple], List[Tuple], List[Tuple]]


class OffsetResolver:
    """
    Maps a mention's surface text onto every offset where it occurs in ``tokens``.

    Results are memoised per instance so that a mention resolves to the same
    offsets whether it is met as an entity block, a relation head or a relation
    tail — otherwise a triplet's endpoints could disagree with the entity
    predictions drawn from the same text.

    Text that occurs nowhere in the source is a hallucination.  It is given a
    unique negative sentinel offset rather than being dropped: dropping it would
    quietly inflate precision, whereas a sentinel can never match gold and still
    counts towards the prediction total.  Sentinels are visible in the per
    instance records, so hallucinated mentions can be inspected directly.
    """

    def __init__(self, tokens: List[str]) -> None:
        self.tokens = list(tokens)
        self._cache: Dict[str, List[Offset]] = {}
        self._next_sentinel = 1

    def resolve(self, text: str) -> List[Offset]:
        key = (text or '').strip()
        if key in self._cache:
            return self._cache[key]

        parts = key.split()
        matches: List[Offset] = []
        if parts:
            k, n = len(parts), len(self.tokens)
            for i in range(n - k + 1):
                if self.tokens[i:i + k] == parts:
                    matches.append((i, i + k))

        if not matches:
            matches = [(-1, -self._next_sentinel)]
            self._next_sentinel += 1

        self._cache[key] = matches
        return matches

    @property
    def resolution_map(self) -> Dict[str, List[List[int]]]:
        return {text: [list(o) for o in offs] for text, offs in self._cache.items()}


def project_blocks(blocks: List[Dict[str, Any]], tokens: List[str]) -> Tuple[OffsetBundle, Dict[str, List[List[int]]]]:
    """
    Projects parsed prediction blocks onto source offsets.

    Returns the offset tuple bundle plus the text -> offsets map that produced it,
    for inclusion in the per instance evaluation records.
    """
    resolver = OffsetResolver(tokens)
    triplets: List[Tuple] = []
    quintuples: List[Tuple] = []
    entities: List[Offset] = []
    mentions: List[Tuple] = []

    for block in blocks:
        h_text = (block.get('text') or '').strip()
        if not h_text:
            continue

        h_type = block.get('type') or ''
        h_offsets = resolver.resolve(h_text)

        for h_off in h_offsets:
            entities.append(h_off)
            if h_type:
                mentions.append((h_off, h_type))

        for rel in block.get('relations', []):
            r_type = (rel.get('type') or '').strip()
            t_text = (rel.get('tail_text') or '').strip()
            if not (r_type and t_text):
                continue

            t_type = rel.get('tail_type') or ''
            for h_off in h_offsets:
                for t_off in resolver.resolve(t_text):
                    triplets.append((h_off, r_type, t_off))
                    if h_type and t_type:
                        quintuples.append((h_off, h_type, r_type, t_off, t_type))

    return (triplets, quintuples, entities, mentions), resolver.resolution_map
