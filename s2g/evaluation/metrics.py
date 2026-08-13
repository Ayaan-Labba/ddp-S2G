"""
Evaluation metrics for the S2G model.

Macro F1 is computed REBEL-style: per-type micro PRF is computed globally
across all sentences for each type in the schema, then averaged across types.
This matches re_score() in REBEL's score.py.

Hallucinated types (predicted types not in the provided schema) are silently
discarded before metric computation, again matching REBEL's re_score(), which
only loops over known relation_types and ignores everything else.
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from s2g.linearisation import VALID_VARIANTS

Triplet = Tuple[int, str, int]              # (head_idx, rel_type, tail_idx)
Quintuple = Tuple[int, str, str, int, str]  # (head_idx, head_type, rel_type, tail_idx, tail_type)
EntityMention = Tuple[int, str]             # (head_idx, entity_type)
EntityBlock = Dict[str, Any]


def extract_from_blocks(blocks: List[EntityBlock]) -> Tuple[List[Triplet], List[Quintuple], List[int], List[EntityMention]]:
    """
    Single-pass extraction of all evaluation elements (indexed by position) from a sentence's EntityBlocks.
    """
    triplets: List[Triplet] = []
    quintuples: List[Quintuple] = []
    entities: List[int] = []
    mentions: List[EntityMention] = []

    for h_idx, ent in enumerate(blocks):
        h_type = ent.get('type', '')
        
        entities.append(h_idx)
        if h_type:
            mentions.append((h_idx, h_type))

        for rel in ent.get('relations', []):
            r_type = rel.get('type', '')
            t_idx = rel.get('tail_id')

            if t_idx is not None and t_idx < len(blocks):
                t_type = blocks[t_idx].get('type', '')
                if r_type:
                    triplets.append((h_idx, r_type, t_idx))
                    if h_type and t_type:
                        quintuples.append((h_idx, h_type, r_type, t_idx, t_type))

    return triplets, quintuples, entities, mentions


def corpus_prf(all_predicted: List[List[Any]], all_gold: List[List[Any]], prefix: str) -> Dict[str, float]:
    """
    Corpus-level (micro) PRF — accumulates TP/FP/FN across all sentences.
    """
    tp, p_len, g_len = 0, 0, 0
    for p, g in zip(all_predicted, all_gold):
        p_set, g_set = set(p), set(g)
        tp    += len(p_set & g_set)
        p_len += len(p_set)
        g_len += len(g_set)

    p   = tp / p_len if p_len > 0 else 0.0
    r   = tp / g_len if g_len > 0 else 0.0
    f1  = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {f"{prefix}_{k}": v for k, v in zip(['precision', 'recall', 'f1'], [p, r, f1])}


def per_type_macro(
    all_predicted: List[List[Any]],
    all_gold: List[List[Any]],
    type_fn,
    schema: List[str],
    prefix: str,
) -> Dict[str, float]:
    """
    REBEL-style macro PRF across types in schema.
    """
    pred_by_type = defaultdict(lambda: defaultdict(set))
    gold_by_type = defaultdict(lambda: defaultdict(set))

    for idx, (pred, gold) in enumerate(zip(all_predicted, all_gold)):
        for p in set(pred):
            pred_by_type[type_fn(p)][idx].add(p)
        for g in set(gold):
            gold_by_type[type_fn(g)][idx].add(g)

    type_ps, type_rs, type_f1s = [], [], []
    for t in schema:
        p_dict = pred_by_type[t]
        g_dict = gold_by_type[t]

        tp = sum(len(p_dict[idx] & g_dict[idx]) for idx in p_dict.keys() | g_dict.keys())
        p_cnt = sum(len(s) for s in p_dict.values())
        g_cnt = sum(len(s) for s in g_dict.values())

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



def compute_metrics_for_variant(
    variant: str,
    all_pred_blocks: List[List[EntityBlock]],
    all_gold_blocks: List[List[EntityBlock]],
    rel_schema: Optional[List[str]] = None,
    ent_schema: Optional[List[str]] = None,
) -> Dict[str, float]:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}.")

    # Single-pass extraction across all sentences
    all_pred_triplets, all_pred_quintuples, all_pred_entities, all_pred_entity_mentions = [], [], [], []
    all_gold_triplets, all_gold_quintuples, all_gold_entities, all_gold_entity_mentions = [], [], [], []

    for p_block in all_pred_blocks:
        t, q, e, m = extract_from_blocks(p_block)
        all_pred_triplets.append(t)
        all_pred_quintuples.append(q)
        all_pred_entities.append(e)
        all_pred_entity_mentions.append(m)

    for g_block in all_gold_blocks:
        t, q, e, m = extract_from_blocks(g_block)
        all_gold_triplets.append(t)
        all_gold_quintuples.append(q)
        all_gold_entities.append(e)
        all_gold_entity_mentions.append(m)

    # Filter out-of-schema predictions
    rel_set = set(rel_schema) if rel_schema else None
    ent_set = set(ent_schema) if ent_schema else None

    if rel_set is not None:
        all_pred_triplets = [[t for t in sent if t[1] in rel_set] for sent in all_pred_triplets]
        all_pred_quintuples = [[q for q in sent if q[2] in rel_set] for sent in all_pred_quintuples]

    if ent_set is not None:
        all_pred_entity_mentions = [[m for m in sent if m[1] in ent_set] for sent in all_pred_entity_mentions]

    # Compute metrics
    m: Dict[str, float] = {}

    # Entity boundary
    if variant in {'boundary_joint', 'boundary_re', 're', 'joint'}:
        m.update(corpus_prf(all_pred_entities, all_gold_entities, 'ner_boundary'))

    # Entity strict
    if variant in {'joint', 're'}:
        if ent_schema is None:
            raise ValueError(f"'ent_schema' must be provided for variant '{variant}' to get macro metrics.")

        m.update(corpus_prf(all_pred_entity_mentions, all_gold_entity_mentions, 'ner'))
        m.update(per_type_macro(all_pred_entity_mentions, all_gold_entity_mentions, lambda x: x[1], ent_schema, "ner"))

    # Relation boundary
    if variant in {'boundary_re', 'boundary_joint', 're', 'joint'}:
        if rel_schema is None:
            raise ValueError(f"'rel_schema' must be provided for variant '{variant}' to get macro metrics.")

        m.update(corpus_prf(all_pred_triplets, all_gold_triplets, 'boundary'))
        m.update(per_type_macro(all_pred_triplets, all_gold_triplets, lambda t: t[1], rel_schema, "boundary"))

    # Relation strict
    if variant in {'re', 'joint'}:
        if rel_schema is None:
            raise ValueError(f"'rel_schema' must be provided for variant '{variant}' to get macro metrics.")

        m.update(corpus_prf(all_pred_quintuples, all_gold_quintuples, 'strict'))
        m.update(per_type_macro(all_pred_quintuples, all_gold_quintuples, lambda q: q[2], rel_schema, "strict"))

    return m