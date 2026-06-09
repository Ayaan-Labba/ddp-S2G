"""
Evaluation metrics for the S2G model.

Provides micro (corpus-level) and macro (instance-average) Precision,
Recall, and F1 at four granularity levels:

1. **Relation Boundary F1** — matches ``(head, rel_type, tail)`` triplets.
   Used for Boundary (NER boundary), RE, Joint, and Joint+ tasks.

2. **Relation Strict F1** — matches
   ``(head, head_type, rel_type, tail, tail_type)`` quintuples requiring
   exact entity-type match.  For the RE task, head/tail types are sourced
   from the NER model's predictions, not gold (assembly handled by the
   evaluation loop).

3. **NER Boundary F1** — matches entity text spans only (no type).

4. **NER Strict F1** — matches ``(entity_text, entity_type)`` pairs.

All corpus-level functions index predictions by instance position so that
the same entity span appearing in multiple instances is not deduplicated
across instances.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Triplet       = Tuple[str, str, str]           # (head, rel_type, tail)
Quintuple     = Tuple[str, str, str, str, str]  # (head, head_type, rel_type, tail, tail_type)
EntityMention = Tuple[str, str]                # (text, type)


# ---- CORE P / R / F1 ----


def _prf(predicted: Set, gold: Set) -> Dict[str, float]:
    """Micro Precision, Recall, and F1 over two pre-built sets."""
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp        = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall    = tp / len(gold)      if gold      else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ---- INSTANCE-LEVEL HELPERS (private) ----


def _rel_boundary_instance(
    pred: List[Triplet],
    gold: List[Triplet],
) -> Dict[str, float]:
    raw = _prf(set(pred), set(gold))
    return {
        "rel_boundary_precision": raw["precision"],
        "rel_boundary_recall":    raw["recall"],
        "rel_boundary_f1":        raw["f1"],
    }


def _rel_strict_instance(
    pred: List[Quintuple],
    gold: List[Quintuple],
) -> Dict[str, float]:
    raw = _prf(set(pred), set(gold))
    return {
        "rel_strict_precision": raw["precision"],
        "rel_strict_recall":    raw["recall"],
        "rel_strict_f1":        raw["f1"],
    }


def _ner_boundary_instance(
    pred: List[str],
    gold: List[str],
) -> Dict[str, float]:
    raw = _prf(set(pred), set(gold))
    return {
        "ner_boundary_precision": raw["precision"],
        "ner_boundary_recall":    raw["recall"],
        "ner_boundary_f1":        raw["f1"],
    }


def _ner_strict_instance(
    pred: List[EntityMention],
    gold: List[EntityMention],
) -> Dict[str, float]:
    raw = _prf(set(pred), set(gold))
    return {
        "ner_strict_precision": raw["precision"],
        "ner_strict_recall":    raw["recall"],
        "ner_strict_f1":        raw["f1"],
    }


# ---- CORPUS-LEVEL MICRO METRICS (public) ----
#
# Each corpus-level function indexes items by ``(instance_idx, *item)``
# so that identical items in different instances are counted separately.
# ===================================================================== #


def corpus_rel_boundary_f1(
    all_predicted: List[List[Triplet]],
    all_gold:      List[List[Triplet]],
) -> Dict[str, float]:
    """Micro relation boundary P/R/F1 across all instances.

    Matches ``(head, rel_type, tail)`` triplets; entity types are ignored.

    Args:
        all_predicted: Per-instance lists of predicted triplets.
        all_gold:      Per-instance lists of gold triplets.

    Returns:
        ``{"rel_boundary_precision": ..., "rel_boundary_recall": ...,
           "rel_boundary_f1": ...}``
    """
    pred_set: Set[Tuple] = set()
    gold_set: Set[Tuple] = set()
    for idx, (preds, golds) in enumerate(zip(all_predicted, all_gold)):
        for t in preds:
            pred_set.add((idx, t[0], t[1], t[2]))
        for t in golds:
            gold_set.add((idx, t[0], t[1], t[2]))
    raw = _prf(pred_set, gold_set)
    return {
        "rel_boundary_precision": raw["precision"],
        "rel_boundary_recall":    raw["recall"],
        "rel_boundary_f1":        raw["f1"],
    }


def corpus_rel_strict_f1(
    all_predicted: List[List[Quintuple]],
    all_gold:      List[List[Quintuple]],
) -> Dict[str, float]:
    """Micro relation strict P/R/F1 across all instances.

    Matches ``(head, head_type, rel_type, tail, tail_type)`` quintuples.
    For the RE task, head/tail types must come from the NER model's
    predictions (assembled by the evaluation loop before calling here).

    Args:
        all_predicted: Per-instance predicted quintuple lists.
        all_gold:      Per-instance gold quintuple lists.

    Returns:
        ``{"rel_strict_precision": ..., "rel_strict_recall": ...,
           "rel_strict_f1": ...}``
    """
    pred_set: Set[Tuple] = set()
    gold_set: Set[Tuple] = set()
    for idx, (preds, golds) in enumerate(zip(all_predicted, all_gold)):
        for q in preds:
            pred_set.add((idx,) + q)
        for q in golds:
            gold_set.add((idx,) + q)
    raw = _prf(pred_set, gold_set)
    return {
        "rel_strict_precision": raw["precision"],
        "rel_strict_recall":    raw["recall"],
        "rel_strict_f1":        raw["f1"],
    }


def corpus_ner_boundary_f1(
    all_predicted: List[List[str]],
    all_gold:      List[List[str]],
) -> Dict[str, float]:
    """Micro NER boundary P/R/F1 (entity text span only).

    Args:
        all_predicted: Per-instance lists of predicted entity text spans.
        all_gold:      Per-instance lists of gold entity text spans.

    Returns:
        ``{"ner_boundary_precision": ..., "ner_boundary_recall": ...,
           "ner_boundary_f1": ...}``
    """
    pred_set: Set[Tuple] = set()
    gold_set: Set[Tuple] = set()
    for idx, (preds, golds) in enumerate(zip(all_predicted, all_gold)):
        for e in preds:
            pred_set.add((idx, e))
        for e in golds:
            gold_set.add((idx, e))
    raw = _prf(pred_set, gold_set)
    return {
        "ner_boundary_precision": raw["precision"],
        "ner_boundary_recall":    raw["recall"],
        "ner_boundary_f1":        raw["f1"],
    }


def corpus_ner_strict_f1(
    all_predicted: List[List[EntityMention]],
    all_gold:      List[List[EntityMention]],
) -> Dict[str, float]:
    """Micro NER strict P/R/F1 (entity text + type must match).

    Args:
        all_predicted: Per-instance lists of ``(text, type)`` tuples.
        all_gold:      Per-instance lists of ``(text, type)`` tuples.

    Returns:
        ``{"ner_strict_precision": ..., "ner_strict_recall": ...,
           "ner_strict_f1": ...}``
    """
    pred_set: Set[Tuple] = set()
    gold_set: Set[Tuple] = set()
    for idx, (preds, golds) in enumerate(zip(all_predicted, all_gold)):
        for e in preds:
            pred_set.add((idx, e[0], e[1]))
        for e in golds:
            gold_set.add((idx, e[0], e[1]))
    raw = _prf(pred_set, gold_set)
    return {
        "ner_strict_precision": raw["precision"],
        "ner_strict_recall":    raw["recall"],
        "ner_strict_f1":        raw["f1"],
    }


# ---- CORPUS-LEVEL MACRO METRICS (public) ----
#
# Macro: compute per-instance P/R/F1, then average over all instances.
# ===================================================================== #


def _macro_average(
    per_instance_metrics: List[Dict[str, float]],
    prefix: str,
) -> Dict[str, float]:
    """Average instance-level metrics into macro values.

    Args:
        per_instance_metrics: List of metric dicts, one per instance.
        prefix:               Metric name prefix, e.g. ``"rel_boundary"``.

    Returns:
        Dict with keys ``"macro_{prefix}_precision"``,
        ``"macro_{prefix}_recall"``, ``"macro_{prefix}_f1"``.
    """
    if not per_instance_metrics:
        return {
            f"macro_{prefix}_precision": 0.0,
            f"macro_{prefix}_recall":    0.0,
            f"macro_{prefix}_f1":        0.0,
        }
    n = len(per_instance_metrics)
    p_key = f"{prefix}_precision"
    r_key = f"{prefix}_recall"
    f_key = f"{prefix}_f1"
    return {
        f"macro_{prefix}_precision": sum(m[p_key] for m in per_instance_metrics) / n,
        f"macro_{prefix}_recall":    sum(m[r_key] for m in per_instance_metrics) / n,
        f"macro_{prefix}_f1":        sum(m[f_key] for m in per_instance_metrics) / n,
    }


def macro_rel_boundary_f1(
    all_predicted: List[List[Triplet]],
    all_gold:      List[List[Triplet]],
) -> Dict[str, float]:
    """Macro relation boundary P/R/F1 (instance average).

    Returns:
        ``{"macro_rel_boundary_precision": ..., "macro_rel_boundary_recall": ...,
           "macro_rel_boundary_f1": ...}``
    """
    per = [
        _rel_boundary_instance(p, g)
        for p, g in zip(all_predicted, all_gold)
    ]
    return _macro_average(per, "rel_boundary")


def macro_rel_strict_f1(
    all_predicted: List[List[Quintuple]],
    all_gold:      List[List[Quintuple]],
) -> Dict[str, float]:
    """Macro relation strict P/R/F1 (instance average).

    Returns:
        ``{"macro_rel_strict_precision": ..., "macro_rel_strict_recall": ...,
           "macro_rel_strict_f1": ...}``
    """
    per = [
        _rel_strict_instance(p, g)
        for p, g in zip(all_predicted, all_gold)
    ]
    return _macro_average(per, "rel_strict")


def macro_ner_boundary_f1(
    all_predicted: List[List[str]],
    all_gold:      List[List[str]],
) -> Dict[str, float]:
    """Macro NER boundary P/R/F1 (instance average).

    Returns:
        ``{"macro_ner_boundary_precision": ..., "macro_ner_boundary_recall": ...,
           "macro_ner_boundary_f1": ...}``
    """
    per = [
        _ner_boundary_instance(p, g)
        for p, g in zip(all_predicted, all_gold)
    ]
    return _macro_average(per, "ner_boundary")


def macro_ner_strict_f1(
    all_predicted: List[List[EntityMention]],
    all_gold:      List[List[EntityMention]],
) -> Dict[str, float]:
    """Macro NER strict P/R/F1 (instance average).

    Returns:
        ``{"macro_ner_strict_precision": ..., "macro_ner_strict_recall": ...,
           "macro_ner_strict_f1": ...}``
    """
    per = [
        _ner_strict_instance(p, g)
        for p, g in zip(all_predicted, all_gold)
    ]
    return _macro_average(per, "ner_strict")


# ---- TASK-SPECIFIC DISPATCH (Experiment 1) ----


def compute_metrics_for_task(
    task: str,
    all_pred_triplets:        Optional[List[List[Triplet]]]       = None,
    all_gold_triplets:        Optional[List[List[Triplet]]]       = None,
    all_pred_quintuples:      Optional[List[List[Quintuple]]]     = None,
    all_gold_quintuples:      Optional[List[List[Quintuple]]]     = None,
    all_pred_entities:        Optional[List[List[str]]]           = None,
    all_gold_entities:        Optional[List[List[str]]]           = None,
    all_pred_entity_mentions: Optional[List[List[EntityMention]]] = None,
    all_gold_entity_mentions: Optional[List[List[EntityMention]]] = None,
) -> Dict[str, float]:
    """Compute all applicable metrics for *task*.

    Dispatches to the correct combination of micro + macro metric
    functions based on the task.  Callers supply only the arguments
    relevant to their task; unused arguments may be ``None``.

    Task → metrics computed
    -----------------------
    ``"boundary"``
        NER Boundary F1 (micro + macro).

    ``"ner"``
        NER Boundary F1 (micro + macro) +
        NER Strict F1 (micro + macro).

    ``"re"``
        Relation Boundary F1 (micro + macro) +
        Relation Strict F1 (micro + macro).
        *Note:* quintuples must have types from the NER model's
        predictions, not gold (assembled by the evaluation loop).

    ``"joint"``
        Relation Boundary F1 (micro + macro).

    ``"joint+"``
        Relation Boundary F1 (micro + macro) +
        Relation Strict F1 (micro + macro) +
        NER Boundary F1 (micro + macro) +
        NER Strict F1 (micro + macro).

    Args:
        task:                     One of ``"boundary"``, ``"ner"``,
                                  ``"re"``, ``"joint"``, ``"joint+"``.
        all_pred_triplets:        Per-instance predicted triplet lists.
        all_gold_triplets:        Per-instance gold triplet lists.
        all_pred_quintuples:      Per-instance predicted quintuple lists.
        all_gold_quintuples:      Per-instance gold quintuple lists.
        all_pred_entities:        Per-instance predicted entity span lists.
        all_gold_entities:        Per-instance gold entity span lists.
        all_pred_entity_mentions: Per-instance predicted ``(text, type)`` lists.
        all_gold_entity_mentions: Per-instance gold ``(text, type)`` lists.

    Returns:
        Combined metric dict with all applicable keys.
    """
    if task not in ("boundary", "ner", "re", "joint", "joint+"):
        raise ValueError(
            f"Unknown task {task!r}.  "
            "Expected one of: 'boundary', 'ner', 're', 'joint', 'joint+'."
        )

    m: Dict[str, float] = {}

    # ---- NER boundary ----
    if task in ("boundary", "ner", "joint+"):
        m.update(corpus_ner_boundary_f1(all_pred_entities, all_gold_entities))
        m.update(macro_ner_boundary_f1( all_pred_entities, all_gold_entities))

    # ---- NER strict ----
    if task in ("ner", "joint+"):
        m.update(corpus_ner_strict_f1(all_pred_entity_mentions, all_gold_entity_mentions))
        m.update(macro_ner_strict_f1( all_pred_entity_mentions, all_gold_entity_mentions))

    # ---- Relation boundary ----
    if task in ("re", "joint", "joint+"):
        m.update(corpus_rel_boundary_f1(all_pred_triplets, all_gold_triplets))
        m.update(macro_rel_boundary_f1( all_pred_triplets, all_gold_triplets))

    # ---- Relation strict ----
    if task in ("re", "joint+"):
        m.update(corpus_rel_strict_f1(all_pred_quintuples, all_gold_quintuples))
        m.update(macro_rel_strict_f1( all_pred_quintuples, all_gold_quintuples))

    return m