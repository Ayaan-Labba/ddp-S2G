"""
Evaluation metrics for the S2G model.
"""
from __future__ import annotations
import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple

Triplet = Tuple[str, str, str]
Quintuple = Tuple[str, str, str, str, str]
EntityMention = Tuple[str, str]


_PUNCT = string.punctuation + "‘’“”—–…"
_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([^\w\s])")


def _norm_span(s: str) -> str:
    """Canonicalise a surface span so detokenisation conventions don't cause
    spurious mismatches. Symmetric on gold and predictions. Neutralises:
    case, internal whitespace, spaces before punctuation (NLTK detok), and
    surrounding punctuation tokens (e.g. gold 'Calif .' vs pred 'Calif').
    Genuine boundary differences (e.g. 'President Kennedy' vs 'Kennedy') are
    preserved as real errors.
    """
    s = _WS.sub(" ", s.strip().casefold())
    s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
    return s.strip(_PUNCT + " ")


def _n_entities(rows: List[List[str]]) -> List[List[str]]:
    return [[_norm_span(x) for x in r] for r in rows]


def _n_mentions(rows: List[List[EntityMention]]) -> List[List[EntityMention]]:
    return [[(_norm_span(t), ty) for t, ty in r] for r in rows]


def _n_triplets(rows: List[List[Triplet]]) -> List[List[Triplet]]:
    return [[(_norm_span(h), rt, _norm_span(t)) for h, rt, t in r] for r in rows]


def _n_quints(rows: List[List[Quintuple]]) -> List[List[Quintuple]]:
    return [[(_norm_span(h), ht, rt, _norm_span(t), tt) for h, ht, rt, t, tt in r] for r in rows]


def _prf(predicted: Set, gold: Set) -> Dict[str, float]:
    if not predicted and not gold: 
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r > 0 else 0.0}


def _instance_prf(pred: List, gold: List, prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in _prf(set(pred), set(gold)).items()}


def _macro_average(metrics: List[Dict[str, float]], prefix: str) -> Dict[str, float]:
    n = len(metrics) or 1
    return {
        f"macro_{prefix}_precision": sum(m[f"{prefix}_precision"] for m in metrics) / n,
        f"macro_{prefix}_recall": sum(m[f"{prefix}_recall"] for m in metrics) / n,
        f"macro_{prefix}_f1": sum(m[f"{prefix}_f1"] for m in metrics) / n
    }


def _corpus_prf(all_predicted: List[List[Any]], all_gold: List[List[Any]], prefix: str) -> Dict[str, float]:
    """
    Computes corpus-level PRF iteratively in O(1) memory.
    """
    tp, p_len, g_len = 0, 0, 0
    
    for p, g in zip(all_predicted, all_gold):
        p_set, g_set = set(p), set(g)
        tp += len(p_set & g_set)
        p_len += len(p_set)
        g_len += len(g_set)
        
    p = tp / p_len if p_len > 0 else 0.0
    r = tp / g_len if g_len > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    
    return {f"{prefix}_{k}": v for k, v in zip(["precision", "recall", "f1"], [p, r, f1])}


def corpus_rel_boundary_f1(all_predicted: List[List[Triplet]], all_gold: List[List[Triplet]]) -> Dict[str, float]:
    return _corpus_prf(_n_triplets(all_predicted), _n_triplets(all_gold), "boundary")


def corpus_rel_strict_f1(all_predicted: List[List[Quintuple]], all_gold: List[List[Quintuple]]) -> Dict[str, float]:
    return _corpus_prf(_n_quints(all_predicted), _n_quints(all_gold), "strict")


def corpus_ner_boundary_f1(all_predicted: List[List[str]], all_gold: List[List[str]]) -> Dict[str, float]:
    return _corpus_prf(_n_entities(all_predicted), _n_entities(all_gold), "ner_boundary")


def corpus_ner_strict_f1(all_predicted: List[List[EntityMention]], all_gold: List[List[EntityMention]]) -> Dict[str, float]:
    return _corpus_prf(_n_mentions(all_predicted), _n_mentions(all_gold), "ner")


def macro_rel_boundary_f1(all_predicted: List[List[Triplet]], all_gold: List[List[Triplet]]) -> Dict[str, float]:
    return _macro_average([_instance_prf(p, g, "boundary") for p, g in zip(_n_triplets(all_predicted), _n_triplets(all_gold))], "boundary")


def macro_rel_strict_f1(all_predicted: List[List[Quintuple]], all_gold: List[List[Quintuple]]) -> Dict[str, float]:
    return _macro_average([_instance_prf(p, g, "strict") for p, g in zip(_n_quints(all_predicted), _n_quints(all_gold))], "strict")


def macro_ner_boundary_f1(all_predicted: List[List[str]], all_gold: List[List[str]]) -> Dict[str, float]:
    return _macro_average([_instance_prf(p, g, "ner_boundary") for p, g in zip(_n_entities(all_predicted), _n_entities(all_gold))], "ner_boundary")


def macro_ner_strict_f1(all_predicted: List[List[EntityMention]], all_gold: List[List[EntityMention]]) -> Dict[str, float]:
    return _macro_average([_instance_prf(p, g, "ner") for p, g in zip(_n_mentions(all_predicted), _n_mentions(all_gold))], "ner")


def compute_metrics_for_task(
    task: str,
    all_pred_triplets: Optional[List[List[Triplet]]] = None, 
    all_gold_triplets: Optional[List[List[Triplet]]] = None,
    all_pred_quintuples: Optional[List[List[Quintuple]]] = None, 
    all_gold_quintuples: Optional[List[List[Quintuple]]] = None,
    all_pred_entities: Optional[List[List[str]]] = None, 
    all_gold_entities: Optional[List[List[str]]] = None,
    all_pred_entity_mentions: Optional[List[List[EntityMention]]] = None, 
    all_gold_entity_mentions: Optional[List[List[EntityMention]]] = None,
) -> Dict[str, float]:
    if task not in {"boundary", "ner", "re", "boundary_re", "boundary_joint", "joint"}: 
        raise ValueError(f"Unknown task {task!r}.")
    
    m = {}
    if task in {"boundary", "ner", "joint"}:
        if all_pred_entities is None or all_gold_entities is None:
            raise ValueError(f"'all_pred_entities' and 'all_gold_entities' must be provided for task '{task}'.")
        m.update(corpus_ner_boundary_f1(all_pred_entities, all_gold_entities))
        m.update(macro_ner_boundary_f1(all_pred_entities, all_gold_entities))
        
    if task in {"ner", "joint"}:
        if all_pred_entity_mentions is None or all_gold_entity_mentions is None:
            raise ValueError(f"'all_pred_entity_mentions' and 'all_gold_entity_mentions' must be provided for task '{task}'.")
        m.update(corpus_ner_strict_f1(all_pred_entity_mentions, all_gold_entity_mentions))
        m.update(macro_ner_strict_f1(all_pred_entity_mentions, all_gold_entity_mentions))
        
    if task in {"re", "boundary_re", "boundary_joint", "joint"}:
        if all_pred_triplets is None or all_gold_triplets is None:
            raise ValueError(f"'all_pred_triplets' and 'all_gold_triplets' must be provided for task '{task}'.")
        m.update(corpus_rel_boundary_f1(all_pred_triplets, all_gold_triplets))
        m.update(macro_rel_boundary_f1(all_pred_triplets, all_gold_triplets))
        
    if task in {"re", "joint"}:
        if all_pred_quintuples is None or all_gold_quintuples is None:
            raise ValueError(f"'all_pred_quintuples' and 'all_gold_quintuples' must be provided for task '{task}'.")
        m.update(corpus_rel_strict_f1(all_pred_quintuples, all_gold_quintuples))
        m.update(macro_rel_strict_f1(all_pred_quintuples, all_gold_quintuples))
        
    return m