"""
Scoring: text track, offset track, and the interaction between them.
"""
from __future__ import annotations

import unittest

from s2g.evaluation.gold import build_gold_blocks, build_gold_offsets
from s2g.evaluation.metrics import compute_metrics_for_variant
from s2g.evaluation.offsets import project_blocks
from s2g.linearisation import S2GTokens, build_graph, parse_graph

from tests.fixtures import (
    ENT_SCHEMA,
    REL_SCHEMA,
    VARIANTS,
    duplicate_instance,
    homograph_instance,
    simple_instance,
)


def score(inst, variant, pred_blocks, dedup=True):
    gold_blocks = build_gold_blocks(inst, variant, dedup)
    gold_offsets = build_gold_offsets(inst, variant)
    pred_offsets, _ = project_blocks(pred_blocks, inst['tokens'])
    return compute_metrics_for_variant(
        variant, [pred_blocks], [gold_blocks],
        rel_schema=REL_SCHEMA, ent_schema=ENT_SCHEMA,
        all_pred_offsets=[pred_offsets], all_gold_offsets=[gold_offsets],
    )


def perfect_prediction(inst, variant, dedup):
    """The gold graph, linearised and parsed back — what a flawless model emits."""
    tok = S2GTokens(variant)
    gold_blocks = build_gold_blocks(inst, variant, dedup)
    parsed, _ = parse_graph(build_graph(gold_blocks, variant, tok), tok)
    return parsed


class TestPerfectPrediction(unittest.TestCase):
    def test_scores_one_across_all_variants_and_dedup_settings(self):
        """
        With no ambiguous surface form, a flawless prediction saturates both
        tracks. Repeated mentions of the *same* type are unambiguous, so the
        duplicate fixture belongs here; the homograph fixture does not, and is
        covered by TestHomographLimits below.
        """
        for inst in (simple_instance(), duplicate_instance()):
            for variant in VARIANTS:
                for dedup in (True, False):
                    with self.subTest(variant=variant, dedup=dedup):
                        pred = perfect_prediction(inst, variant, dedup)
                        m = score(inst, variant, pred, dedup)
                        for key, value in m.items():
                            if key.endswith('_f1') and not key.startswith('macro_'):
                                self.assertAlmostEqual(
                                    value, 1.0, places=6, msg=f'{key} = {value}'
                                )


class TestHomographLimits(unittest.TestCase):
    """
    Two ceilings that no amount of correct generation can lift, both arising
    because the decoder emits text while the corpora annotate spans. These tests
    pin the current behaviour so a future change to it is deliberate.
    """

    def setUp(self):
        self.inst = homograph_instance()

    def test_joint_strict_penalised_for_unresolvable_tail_type(self):
        """
        'Boeing is based in Washington' has tail type 'location' in gold. The joint
        format never emits tail types inline, so parsing resolves the tail to the
        first 'Washington' block, typed 'person'. Gold is read from the annotation
        and is therefore right, and the prediction cannot be.
        """
        pred = perfect_prediction(self.inst, 'joint', dedup=True)
        m = score(self.inst, 'joint', pred)

        self.assertEqual(m['boundary_f1'], 1.0)     # untyped relations are fine
        self.assertEqual(m['strict_recall'], 0.5)   # 1 of 2 quintuples matches

    def test_offset_precision_capped_by_ambiguous_projection(self):
        """
        A mention emitted once projects onto every occurrence of its text, so an
        ambiguous form necessarily produces one wrong offset per extra occurrence.
        Recall is unaffected; precision pays.
        """
        for variant in ('boundary_joint', 'boundary_re'):
            with self.subTest(variant=variant):
                pred = perfect_prediction(self.inst, variant, dedup=True)
                m = score(self.inst, variant, pred)

                self.assertEqual(m['offset_boundary_recall'], 1.0)
                self.assertEqual(m['offset_boundary_precision'], 0.5)

    def test_offset_entity_types_replicated_across_occurrences(self):
        """Both 'Washington' spans inherit both predicted types, so one is wrong."""
        pred = perfect_prediction(self.inst, 'joint', dedup=True)
        m = score(self.inst, 'joint', pred)

        self.assertEqual(m['offset_ner_boundary_f1'], 1.0)  # spans are all found
        self.assertLess(m['offset_ner_f1'], 1.0)            # types cannot be placed


class TestDuplicateMentionRecall(unittest.TestCase):
    """Failure mode 1: text scoring cannot count two gold relations that share a
    surface form; offset scoring can."""

    def setUp(self):
        self.inst = duplicate_instance()

    def test_offset_gold_counts_more_relations_than_text_gold(self):
        m = score(self.inst, 'joint', perfect_prediction(self.inst, 'joint', True))
        self.assertEqual(m['boundary_f1'], 1.0)
        self.assertEqual(m['offset_boundary_f1'], 1.0)

    def test_single_emission_credited_with_both_gold_relations(self):
        pred = [{
            'text': 'Bolshoi Ballet', 'type': 'organization',
            'relations': [{'type': 'is based in', 'tail_text': 'Moscow', 'tail_type': 'location'}],
        }, {'text': 'Moscow', 'type': 'location', 'relations': []}]
        m = score(self.inst, 'joint', pred)
        # Text: one gold triplet, one predicted, both recalls saturate.
        self.assertEqual(m['boundary_recall'], 1.0)
        # Offsets: two distinct gold triplets, both matched by the one emission.
        self.assertEqual(m['offset_boundary_recall'], 1.0)
        self.assertEqual(m['offset_ner_boundary_recall'], 1.0)


class TestHallucinations(unittest.TestCase):
    def test_hallucination_counts_against_precision(self):
        inst = duplicate_instance()
        pred = perfect_prediction(inst, 'joint', True)
        with_noise = pred + [{'text': 'Atlantis', 'type': 'location', 'relations': []}]

        clean = score(inst, 'joint', pred)
        noisy = score(inst, 'joint', with_noise)

        self.assertEqual(clean['offset_ner_boundary_precision'], 1.0)
        self.assertLess(noisy['offset_ner_boundary_precision'], 1.0)
        self.assertEqual(noisy['offset_ner_boundary_recall'], 1.0)

    def test_hallucinated_relation_is_not_silently_dropped(self):
        inst = duplicate_instance()
        pred = perfect_prediction(inst, 'joint', True) + [{
            'text': 'Atlantis', 'type': 'location',
            'relations': [{'type': 'is based in', 'tail_text': 'Narnia', 'tail_type': 'location'}],
        }]
        m = score(inst, 'joint', pred)
        self.assertLess(m['offset_boundary_precision'], 1.0)


class TestSchemaFiltering(unittest.TestCase):
    def test_out_of_schema_predictions_discarded(self):
        inst = duplicate_instance()
        pred = perfect_prediction(inst, 'joint', True) + [{
            'text': 'Moscow', 'type': 'location',
            'relations': [{'type': 'not a real relation', 'tail_text': 'Moscow', 'tail_type': 'location'}],
        }]
        m = score(inst, 'joint', pred)
        # The bogus relation type is dropped before scoring, so precision is intact.
        self.assertEqual(m['boundary_precision'], 1.0)
        self.assertEqual(m['offset_boundary_precision'], 1.0)


class TestMetricSurface(unittest.TestCase):
    def test_offset_keys_mirror_text_keys(self):
        inst = homograph_instance()
        m = score(inst, 'joint', perfect_prediction(inst, 'joint', True))
        for key in ['ner_boundary_f1', 'ner_f1', 'boundary_f1', 'strict_f1',
                    'macro_ner_f1', 'macro_boundary_f1', 'macro_strict_f1']:
            self.assertIn(key, m)
            self.assertIn(f'offset_{key}' if not key.startswith('macro_')
                          else f'macro_offset_{key[len("macro_"):]}', m)

    def test_boundary_variants_emit_no_strict_metrics(self):
        inst = homograph_instance()
        for variant in ('boundary_joint', 'boundary_re'):
            with self.subTest(variant=variant):
                m = score(inst, variant, perfect_prediction(inst, variant, True))
                self.assertNotIn('strict_f1', m)
                self.assertNotIn('offset_strict_f1', m)
                self.assertIn('boundary_f1', m)
                self.assertIn('offset_boundary_f1', m)

    def test_text_metrics_unchanged_when_offsets_absent(self):
        inst = homograph_instance()
        pred = perfect_prediction(inst, 'joint', True)
        gold = build_gold_blocks(inst, 'joint', True)
        without = compute_metrics_for_variant(
            'joint', [pred], [gold], rel_schema=REL_SCHEMA, ent_schema=ENT_SCHEMA
        )
        with_offsets = score(inst, 'joint', pred)
        self.assertFalse(any(k.startswith('offset_') for k in without))
        for key, value in without.items():
            self.assertAlmostEqual(with_offsets[key], value, places=9)

    def test_macro_averages_over_absent_types_as_zero(self):
        """A schema type neither predicted nor gold contributes 0.0, REBEL-style."""
        inst = duplicate_instance()  # only 'is based in' occurs
        m = score(inst, 'joint', perfect_prediction(inst, 'joint', True))
        self.assertEqual(m['boundary_f1'], 1.0)
        self.assertAlmostEqual(m['macro_boundary_f1'], 1.0 / len(REL_SCHEMA), places=6)


if __name__ == '__main__':
    unittest.main()
