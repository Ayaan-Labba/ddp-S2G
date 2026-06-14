"""
Evaluation metrics for the S2G model.

Macro F1 is computed REBEL-style: per-type micro PRF is computed globally
across all sentences for each type in the schema, then averaged across types.
This matches re_score() in REBEL's score.py.

Hallucinated types (predicted types not in the provided schema) are silently
discarded before metric computation, again matching REBEL's re_score(), which
only loops over known relation_types and ignores everything else.

The empty/empty case returns 0/0/0 (not 1/1/1), so datasets with many
entity-free or relation-free sentences do not artificially inflate macro scores.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple

Triplet = Tuple[str, str, str]           # (head_text, rel_type, tail_text)
Quintuple = Tuple[str, str, str, str, str]  # (head, head_type, rel, tail, tail_type)
EntityMention = Tuple[str, str]          # (span_text, entity_type)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _prf(predicted: Set, gold: Set) -> Dict[str, float]:
    """
    Micro PRF for a single instance.
    Both-empty → 0/0/0 (not 1/1/1) so that instances with no annotations
    do not inflate per-instance macro averages.
    """
    if not predicted and not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r > 0 else 0.0}


def _corpus_prf(all_predicted: List[List[Any]], all_gold: List[List[Any]], prefix: str) -> Dict[str, float]:
    """Corpus-level (micro) PRF — accumulates TP/FP/FN across all sentences."""
    tp, p_len, g_len = 0, 0, 0
    for p, g in zip(all_predicted, all_gold):
        p_set, g_set = set(p), set(g)
        tp    += len(p_set & g_set)
        p_len += len(p_set)
        g_len += len(g_set)
    p   = tp / p_len if p_len > 0 else 0.0
    r   = tp / g_len if g_len > 0 else 0.0
    f1  = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {f"{prefix}_{k}": v for k, v in zip(["precision", "recall", "f1"], [p, r, f1])}


def _per_type_macro(
    all_predicted: List[List[Any]],
    all_gold: List[List[Any]],
    type_fn,          # callable: element → type string
    schema: List[str],
    prefix: str,
) -> Dict[str, float]:
    """
    REBEL-style macro: for each type in schema, compute a global micro PRF
    (summing TP/FP/FN across all sentences), then average those per-type F1s.

    This mirrors re_score() in REBEL's score.py exactly:
      - types not in schema are ignored for both predictions and gold
      - types with zero gold AND zero predictions contribute 0/0/0 to the average
    """
    type_ps, type_rs, type_f1s = [], [], []
    for t in schema:
        tp, p_cnt, g_cnt = 0, 0, 0
        for pred, gold in zip(all_predicted, all_gold):
            p_t = {x for x in set(pred) if type_fn(x) == t}
            g_t = {x for x in set(gold) if type_fn(x) == t}
            tp    += len(p_t & g_t)
            p_cnt += len(p_t)
            g_cnt += len(g_t)
        p  = tp / p_cnt if p_cnt > 0 else 0.0
        r  = tp / g_cnt if g_cnt > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        type_ps.append(p)
        type_rs.append(r)
        type_f1s.append(f1)

    n = len(schema) or 1
    return {
        f"macro_{prefix}_precision": sum(type_ps)  / n,
        f"macro_{prefix}_recall":    sum(type_rs)  / n,
        f"macro_{prefix}_f1":        sum(type_f1s) / n,
    }


# ---------------------------------------------------------------------------
# Public corpus-level (micro) functions
# ---------------------------------------------------------------------------

def corpus_rel_boundary_f1(
    all_predicted: List[List[Triplet]], all_gold: List[List[Triplet]]
) -> Dict[str, float]:
    return _corpus_prf(all_predicted, all_gold, "boundary")


def corpus_rel_strict_f1(
    all_predicted: List[List[Quintuple]], all_gold: List[List[Quintuple]]
) -> Dict[str, float]:
    return _corpus_prf(all_predicted, all_gold, "strict")


def corpus_ner_boundary_f1(
    all_predicted: List[List[str]], all_gold: List[List[str]]
) -> Dict[str, float]:
    return _corpus_prf(all_predicted, all_gold, "ner_boundary")


def corpus_ner_strict_f1(
    all_predicted: List[List[EntityMention]], all_gold: List[List[EntityMention]]
) -> Dict[str, float]:
    return _corpus_prf(all_predicted, all_gold, "ner")


# ---------------------------------------------------------------------------
# Public macro (per-type) functions — REBEL-style
# ---------------------------------------------------------------------------

def macro_rel_boundary_f1(
    all_predicted: List[List[Triplet]],
    all_gold: List[List[Triplet]],
    rel_schema: List[str],
) -> Dict[str, float]:
    """
    Macro boundary-RE F1, averaged per relation type (REBEL-style).
    Triplet layout: (head_text, rel_type, tail_text) — type is at index 1.
    """
    return _per_type_macro(all_predicted, all_gold, lambda t: t[1], rel_schema, "boundary")


def macro_rel_strict_f1(
    all_predicted: List[List[Quintuple]],
    all_gold: List[List[Quintuple]],
    rel_schema: List[str],
) -> Dict[str, float]:
    """
    Macro strict-RE F1, averaged per relation type (REBEL-style).
    Quintuple layout: (head, head_type, rel_type, tail, tail_type) — rel at index 2.
    """
    return _per_type_macro(all_predicted, all_gold, lambda q: q[2], rel_schema, "strict")


def macro_ner_boundary_f1(
    all_predicted: List[List[str]], all_gold: List[List[str]]
) -> Dict[str, float]:
    """
    Per-instance macro for NER boundary (span text only — no type available).
    Kept as per-instance average; uses 0/0/0 for empty/empty.
    """
    n = len(all_predicted) or 1
    agg = {"ner_boundary_precision": 0.0, "ner_boundary_recall": 0.0, "ner_boundary_f1": 0.0}
    for p, g in zip(all_predicted, all_gold):
        m = _prf(set(p), set(g))
        agg["ner_boundary_precision"] += m["precision"]
        agg["ner_boundary_recall"]    += m["recall"]
        agg["ner_boundary_f1"]        += m["f1"]
    return {f"macro_{k}": v / n for k, v in agg.items()}


def macro_ner_strict_f1(
    all_predicted: List[List[EntityMention]],
    all_gold: List[List[EntityMention]],
    entity_schema: List[str],
) -> Dict[str, float]:
    """
    Macro NER strict F1, averaged per entity type (REBEL-style).
    EntityMention layout: (span_text, entity_type) — type is at index 1.
    """
    return _per_type_macro(all_predicted, all_gold, lambda m: m[1], entity_schema, "ner")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compute_metrics_for_task(
    task: str,
    rel_schema: Optional[List[str]] = None,
    entity_schema: Optional[List[str]] = None,
    all_pred_triplets:        Optional[List[List[Triplet]]]       = None,
    all_gold_triplets:        Optional[List[List[Triplet]]]       = None,
    all_pred_quintuples:      Optional[List[List[Quintuple]]]     = None,
    all_gold_quintuples:      Optional[List[List[Quintuple]]]     = None,
    all_pred_entities:        Optional[List[List[str]]]           = None,
    all_gold_entities:        Optional[List[List[str]]]           = None,
    all_pred_entity_mentions: Optional[List[List[EntityMention]]] = None,
    all_gold_entity_mentions: Optional[List[List[EntityMention]]] = None,
) -> Dict[str, float]:
    """
    Compute all metrics appropriate for *task*.

    Parameters
    ----------
    rel_schema :
        Closed set of valid relation types.  Predicted triplets / quintuples
        whose relation type is NOT in this set are silently discarded before
        computing any metric (mirrors REBEL's re_score loop over relation_types).
        Pass None to skip filtering.
    entity_schema :
        Closed set of valid entity types.  Predicted entity mentions whose
        type is NOT in this set are discarded similarly.
        Pass None to skip filtering.
    """
    if task not in {"boundary", "ner", "re", "boundary_re", "boundary_joint", "joint"}:
        raise ValueError(f"Unknown task {task!r}.")

    # ------------------------------------------------------------------
    # Hallucination filtering — discard out-of-schema predictions only.
    # Gold is never filtered so recall is computed over all gold items.
    # ------------------------------------------------------------------
    rel_set = set(rel_schema) if rel_schema else None
    ent_set = set(entity_schema) if entity_schema else None

    if rel_set is not None and all_pred_triplets is not None:
        all_pred_triplets = [
            [t for t in sent if t[1] in rel_set] for sent in all_pred_triplets
        ]
    if rel_set is not None and all_pred_quintuples is not None:
        all_pred_quintuples = [
            [q for q in sent if q[2] in rel_set] for sent in all_pred_quintuples
        ]
    if ent_set is not None and all_pred_entity_mentions is not None:
        all_pred_entity_mentions = [
            [m for m in sent if m[1] in ent_set] for sent in all_pred_entity_mentions
        ]

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    m: Dict[str, float] = {}

    # Entity boundary (span text only)
    if task in {"boundary", "ner", "joint"}:
        if all_pred_entities is None or all_gold_entities is None:
            raise ValueError(f"'all_pred_entities' and 'all_gold_entities' must be provided for task '{task}'.")
        m.update(corpus_ner_boundary_f1(all_pred_entities, all_gold_entities))
        m.update(macro_ner_boundary_f1(all_pred_entities, all_gold_entities))

    # Entity strict (span + type)
    if task in {"ner", "joint"}:
        if all_pred_entity_mentions is None or all_gold_entity_mentions is None:
            raise ValueError(f"'all_pred_entity_mentions' and 'all_gold_entity_mentions' must be provided for task '{task}'.")
        if entity_schema is None:
            raise ValueError(f"'entity_schema' must be provided for per-type macro in task '{task}'.")
        m.update(corpus_ner_strict_f1(all_pred_entity_mentions, all_gold_entity_mentions))
        m.update(macro_ner_strict_f1(all_pred_entity_mentions, all_gold_entity_mentions, entity_schema))

    # Relation boundary (triplet: head, rel, tail)
    if task in {"re", "boundary_re", "boundary_joint", "joint"}:
        if all_pred_triplets is None or all_gold_triplets is None:
            raise ValueError(f"'all_pred_triplets' and 'all_gold_triplets' must be provided for task '{task}'.")
        if rel_schema is None:
            raise ValueError(f"'rel_schema' must be provided for per-type macro in task '{task}'.")
        m.update(corpus_rel_boundary_f1(all_pred_triplets, all_gold_triplets))
        m.update(macro_rel_boundary_f1(all_pred_triplets, all_gold_triplets, rel_schema))

    # Relation strict (quintuple: head, head_type, rel, tail, tail_type)
    if task in {"re", "joint"}:
        if all_pred_quintuples is None or all_gold_quintuples is None:
            raise ValueError(f"'all_pred_quintuples' and 'all_gold_quintuples' must be provided for task '{task}'.")
        if rel_schema is None:
            raise ValueError(f"'rel_schema' must be provided for per-type macro in task '{task}'.")
        m.update(corpus_rel_strict_f1(all_pred_quintuples, all_gold_quintuples))
        m.update(macro_rel_strict_f1(all_pred_quintuples, all_gold_quintuples, rel_schema))

    return m