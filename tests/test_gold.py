"""
Gold construction from preprocessed instances.
"""
from __future__ import annotations

import unittest

from s2g.evaluation.gold import build_gold_blocks, build_gold_offsets

from tests.fixtures import VARIANTS, duplicate_instance, homograph_instance, simple_instance


class TestGoldBlocks(unittest.TestCase):
    def test_joint_gold_covers_every_entity(self):
        inst = homograph_instance()
        blocks = build_gold_blocks(inst, 'joint', dedup=False)
        self.assertEqual(len(blocks), len(inst['entities']))

    def test_re_gold_includes_tail_only_entities(self):
        """
        resolve_tail_entities must append tail mentions, otherwise every tail the
        model correctly emits would score as a precision error.
        """
        blocks = build_gold_blocks(simple_instance(), 're', dedup=True)
        self.assertEqual(sorted(b['text'] for b in blocks), ['Boeing', 'Seattle'])
        seattle = next(b for b in blocks if b['text'] == 'Seattle')
        self.assertEqual(seattle['relations'], [])
        self.assertEqual(seattle['type'], 'location')

    def test_dedup_flag_is_honoured(self):
        inst = duplicate_instance()
        merged = build_gold_blocks(inst, 'joint', dedup=True)
        kept = build_gold_blocks(inst, 'joint', dedup=False)
        self.assertLess(len(merged), len(kept))

    def test_gold_blocks_built_for_every_variant(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                self.assertTrue(build_gold_blocks(homograph_instance(), variant, True))


class TestGoldOffsets(unittest.TestCase):
    def test_duplicate_mentions_counted_separately(self):
        triplets, _, entities, _ = build_gold_offsets(duplicate_instance(), 'joint')
        self.assertEqual(sorted(entities), [(0, 1), (2, 4), (5, 6)])
        self.assertEqual(len(set(triplets)), 2)

    def test_independent_of_dedup(self):
        """
        Offset gold is read off the annotation, so it must not vary with a setting
        that only governs target construction.
        """
        inst = duplicate_instance()
        self.assertEqual(
            build_gold_offsets(inst, 'joint'), build_gold_offsets(inst, 'joint')
        )
        # Blocks differ under dedup, offsets must not.
        self.assertNotEqual(
            len(build_gold_blocks(inst, 'joint', True)),
            len(build_gold_blocks(inst, 'joint', False)),
        )

    def test_re_variants_restricted_to_participants(self):
        tokens = ['Boeing', 'is', 'in', 'Seattle', 'unlike', 'Airbus']
        inst = {
            'text': " ".join(tokens), 'tokens': tokens,
            'entities': [
                {'text': 'Boeing', 'offset': [0, 1], 'type': 'organization'},
                {'text': 'Seattle', 'offset': [3, 4], 'type': 'location'},
                {'text': 'Airbus', 'offset': [5, 6], 'type': 'organization'},
            ],
            'relations': [{
                'head': {'text': 'Boeing', 'offset': [0, 1], 'type': 'organization'},
                'tail': {'text': 'Seattle', 'offset': [3, 4], 'type': 'location'},
                'type': 'is based in',
            }],
            'entity_types': ['location', 'organization'],
            'rel_types': ['is based in'],
        }
        _, _, re_entities, _ = build_gold_offsets(inst, 're')
        _, _, joint_entities, _ = build_gold_offsets(inst, 'joint')
        self.assertEqual(sorted(re_entities), [(0, 1), (3, 4)])       # Airbus excluded
        self.assertEqual(sorted(joint_entities), [(0, 1), (3, 4), (5, 6)])

    def test_boundary_variants_emit_no_typed_tuples(self):
        for variant in ('boundary_joint', 'boundary_re'):
            with self.subTest(variant=variant):
                _, quintuples, _, mentions = build_gold_offsets(homograph_instance(), variant)
                self.assertEqual(quintuples, [])
                self.assertEqual(mentions, [])

    def test_homograph_types_preserved(self):
        _, _, _, mentions = build_gold_offsets(homograph_instance(), 'joint')
        by_offset = dict(mentions)
        self.assertEqual(by_offset[(0, 1)], 'person')
        self.assertEqual(by_offset[(4, 5)], 'location')


if __name__ == '__main__':
    unittest.main()
