"""
Block building, linearisation and parsing.
"""
from __future__ import annotations

import unittest

from s2g.linearisation import (
    S2GTokens,
    build_graph,
    organise_filter_and_block,
    parse_graph,
    resolve_tail_entities,
)

from tests.fixtures import (
    ENT_SCHEMA,
    REL_SCHEMA,
    TYPED_VARIANTS,
    VARIANTS,
    duplicate_instance,
    homograph_instance,
)


def blocks_for(inst, variant, dedup):
    use_types = variant in TYPED_VARIANTS
    return organise_filter_and_block(
        inst['entities'], inst['relations'],
        set(ENT_SCHEMA) if use_types else set(), set(REL_SCHEMA),
        variant=variant, use_types=use_types, dedup=dedup,
    )


class TestDedup(unittest.TestCase):
    def test_homographs_never_merge(self):
        """(text, type) keying keeps Washington/person apart from Washington/location."""
        inst = homograph_instance()
        for variant in TYPED_VARIANTS:
            with self.subTest(variant=variant):
                blocks = blocks_for(inst, variant, dedup=True)
                washingtons = [b for b in blocks if b['text'] == 'Washington']
                self.assertEqual(len(washingtons), 2 if variant == 'joint' else 1)
                if variant == 'joint':
                    self.assertEqual(
                        {b['type'] for b in washingtons}, {'person', 'location'}
                    )

    def test_same_text_and_type_merges_when_dedup(self):
        inst = duplicate_instance()
        blocks = blocks_for(inst, 'joint', dedup=True)
        self.assertEqual(len([b for b in blocks if b['text'] == 'Moscow']), 1)

    def test_every_mention_kept_without_dedup(self):
        inst = duplicate_instance()
        blocks = blocks_for(inst, 'joint', dedup=False)
        self.assertEqual(len([b for b in blocks if b['text'] == 'Moscow']), 2)

    def test_relations_collapse_only_when_dedup(self):
        """Both 'Bolshoi Ballet is based in Moscow' relations share a quintuple."""
        inst = duplicate_instance()
        merged = blocks_for(inst, 'joint', dedup=True)
        kept = blocks_for(inst, 'joint', dedup=False)
        self.assertEqual(sum(len(b['relations']) for b in merged), 1)
        self.assertEqual(sum(len(b['relations']) for b in kept), 2)

    def test_boundary_variants_key_on_text_alone(self):
        """With no types the key degenerates to text, so the homographs merge."""
        inst = homograph_instance()
        blocks = blocks_for(inst, 'boundary_joint', dedup=True)
        self.assertEqual(len([b for b in blocks if b['text'] == 'Washington']), 1)


class TestBlockSelection(unittest.TestCase):
    def test_joint_emits_every_entity(self):
        inst = homograph_instance()
        blocks = blocks_for(inst, 'joint', dedup=False)
        self.assertEqual(len(blocks), len(inst['entities']))

    def test_re_emits_heads_only(self):
        """'Washington'/location is a tail only, so it gets no block of its own."""
        inst = homograph_instance()
        blocks = blocks_for(inst, 're', dedup=False)
        self.assertEqual(
            sorted(b['text'] for b in blocks), ['Boeing', 'Washington']
        )
        self.assertTrue(all(b['relations'] for b in blocks))


class TestRoundTrip(unittest.TestCase):
    def test_build_then_parse_preserves_relations(self):
        for inst in (homograph_instance(), duplicate_instance()):
            for variant in VARIANTS:
                for dedup in (True, False):
                    with self.subTest(variant=variant, dedup=dedup):
                        tok = S2GTokens(variant)
                        blocks = blocks_for(inst, variant, dedup)
                        parsed, _ = parse_graph(build_graph(blocks, variant, tok), tok)

                        built = sorted(
                            (b['text'], r['type'], r['tail_text'])
                            for b in blocks for r in b['relations']
                        )
                        got = sorted(
                            (b['text'], r['type'], r['tail_text'])
                            for b in parsed for r in b['relations']
                        )
                        self.assertEqual(got, built)

    def test_parsing_never_deduplicates(self):
        """A graph naming the same mention twice parses back to two blocks."""
        tok = S2GTokens('joint')
        graph = (
            '<extra_id_0> Moscow <e_type> location '
            '<extra_id_1> Moscow <e_type> location'
        )
        parsed, _ = parse_graph(graph, tok)
        self.assertEqual(len(parsed), 2)

    def test_duplicate_relations_survive_parsing(self):
        tok = S2GTokens('joint')
        graph = (
            '<extra_id_0> Bolshoi Ballet <e_type> organization '
            '<r_type> is based in <tail> Moscow '
            '<nr_type> is based in <tail> Moscow'
        )
        parsed, _ = parse_graph(graph, tok)
        self.assertEqual(len(parsed[0]['relations']), 2)

    def test_reused_sentinel_starts_new_block(self):
        tok = S2GTokens('joint')
        parsed, _ = parse_graph('<extra_id_0> Alpha <extra_id_0> Beta', tok)
        self.assertEqual([b['text'] for b in parsed], ['Alpha', 'Beta'])

    def test_rejection_types_parse_into_rejected(self):
        tok = S2GTokens('joint', use_rejection=True)
        blocks = blocks_for(homograph_instance(), 'joint', dedup=True)
        graph = build_graph(
            blocks, 'joint', tok, use_rejection=True,
            rejected_ent_types=['artifact'], rejected_rel_types=['killed'],
        )
        _, rejected = parse_graph(graph, tok)
        self.assertEqual(sorted(rejected), ['artifact', 'killed'])


class TestResolveTailEntities(unittest.TestCase):
    def test_tail_type_taken_from_first_occurrence(self):
        entities = [
            {'text': 'Washington', 'type': 'person', 'relations': []},
            {'text': 'Boeing', 'type': 'organization',
             'relations': [{'type': 'is based in', 'tail_text': 'Washington', 'tail_type': None}]},
            {'text': 'Washington', 'type': 'location', 'relations': []},
        ]
        resolve_tail_entities(entities)
        self.assertEqual(entities[1]['relations'][0]['tail_type'], 'person')

    def test_unseen_tail_appended_as_entity(self):
        entities = [
            {'text': 'Boeing', 'type': 'organization',
             'relations': [{'type': 'is based in', 'tail_text': 'Seattle', 'tail_type': 'location'}]},
        ]
        resolve_tail_entities(entities)
        self.assertEqual([e['text'] for e in entities], ['Boeing', 'Seattle'])
        self.assertEqual(entities[1]['type'], 'location')

    def test_untyped_block_inherits_inline_tail_type(self):
        entities = [
            {'text': 'Seattle', 'type': None, 'relations': []},
            {'text': 'Boeing', 'type': 'organization',
             'relations': [{'type': 'is based in', 'tail_text': 'Seattle', 'tail_type': 'location'}]},
        ]
        resolve_tail_entities(entities)
        self.assertEqual(entities[0]['type'], 'location')


if __name__ == '__main__':
    unittest.main()
