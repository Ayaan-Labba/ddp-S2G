"""
Projection of predicted mentions onto source offsets.
"""
from __future__ import annotations

import unittest

from s2g.evaluation.offsets import OffsetResolver, project_blocks

from tests.fixtures import duplicate_instance


class TestOffsetResolver(unittest.TestCase):
    def setUp(self):
        self.tokens = duplicate_instance()['tokens']  # Moscow hosts Bolshoi Ballet in Moscow

    def test_finds_every_occurrence(self):
        self.assertEqual(OffsetResolver(self.tokens).resolve('Moscow'), [(0, 1), (5, 6)])

    def test_multi_token_mention(self):
        self.assertEqual(OffsetResolver(self.tokens).resolve('Bolshoi Ballet'), [(2, 4)])

    def test_hallucination_gets_negative_sentinel(self):
        offsets = OffsetResolver(self.tokens).resolve('Atlantis')
        self.assertEqual(len(offsets), 1)
        self.assertLess(offsets[0][0], 0)

    def test_sentinels_are_unique_per_mention(self):
        resolver = OffsetResolver(self.tokens)
        self.assertNotEqual(resolver.resolve('Atlantis'), resolver.resolve('Narnia'))

    def test_resolution_is_memoised(self):
        """Same text must resolve identically wherever it is met in the graph."""
        resolver = OffsetResolver(self.tokens)
        self.assertEqual(resolver.resolve('Atlantis'), resolver.resolve('Atlantis'))

    def test_partial_token_does_not_match(self):
        self.assertLess(OffsetResolver(self.tokens).resolve('Mosc')[0][0], 0)


class TestProjectBlocks(unittest.TestCase):
    def setUp(self):
        self.tokens = duplicate_instance()['tokens']

    def test_one_entity_per_match(self):
        blocks = [{'text': 'Moscow', 'type': 'location', 'relations': []}]
        (_, _, entities, mentions), _ = project_blocks(blocks, self.tokens)
        self.assertEqual(sorted(entities), [(0, 1), (5, 6)])
        self.assertEqual(sorted(mentions), [((0, 1), 'location'), ((5, 6), 'location')])

    def test_relations_expand_over_cross_product(self):
        """One emitted relation with a twice-occurring tail yields two predictions."""
        blocks = [{
            'text': 'Bolshoi Ballet', 'type': 'organization',
            'relations': [{'type': 'is based in', 'tail_text': 'Moscow', 'tail_type': 'location'}],
        }]
        (triplets, quintuples, _, _), _ = project_blocks(blocks, self.tokens)
        self.assertEqual(sorted(triplets), [
            ((2, 4), 'is based in', (0, 1)),
            ((2, 4), 'is based in', (5, 6)),
        ])
        self.assertEqual(len(quintuples), 2)

    def test_duplicate_predictions_collapse_on_offsets(self):
        """Emitting the same mention twice cannot be rewarded twice."""
        once = [{'text': 'Moscow', 'type': 'location', 'relations': []}]
        twice = once + [{'text': 'Moscow', 'type': 'location', 'relations': []}]
        (_, _, e_once, _), _ = project_blocks(once, self.tokens)
        (_, _, e_twice, _), _ = project_blocks(twice, self.tokens)
        self.assertEqual(set(e_once), set(e_twice))

    def test_resolution_map_exposes_hallucinations(self):
        blocks = [{'text': 'Atlantis', 'type': 'location', 'relations': []}]
        _, offset_map = project_blocks(blocks, self.tokens)
        self.assertIn('Atlantis', offset_map)
        self.assertLess(offset_map['Atlantis'][0][0], 0)

    def test_untyped_blocks_emit_no_mentions_or_quintuples(self):
        blocks = [{
            'text': 'Bolshoi Ballet', 'type': None,
            'relations': [{'type': 'is based in', 'tail_text': 'Moscow', 'tail_type': None}],
        }]
        (triplets, quintuples, entities, mentions), _ = project_blocks(blocks, self.tokens)
        self.assertTrue(entities)
        self.assertTrue(triplets)
        self.assertEqual(mentions, [])
        self.assertEqual(quintuples, [])


if __name__ == '__main__':
    unittest.main()
