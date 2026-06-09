"""
Pre-processing script for the SciERC dataset (Luan et al., 2018).

Produces two complementary output formats:

**Sentence-level** (default, always generated)
    One instance per sentence.  Within-sentence NER and relation spans
    are converted from inclusive to half-open indices.  Cross-sentence
    relations, which do not appear in the standard within-sentence
    annotation format, are absent by definition.  This is the
    **standard evaluation protocol** followed by all published SoTA
    models (SpERT, PURE, DyGIE++, PLMarker, etc.) and produces
    results directly comparable to the literature.

**Document-level** (opt-in via ``--document_level``)
    One instance per abstract.  All sentence token lists are
    concatenated into a single document token list, and all NER and
    relation spans are converted to document-level offsets.  This
    allows the model to attend to the full abstract context during both
    encoding and augmentation.  Since the standard SciERC annotation
    stores relations within sentences (not across them), cross-sentence
    relations are NOT added by this mode; the benefit is broader
    contextual encoding, not additional relation coverage.  Results are
    NOT directly comparable to sentence-level benchmarks.

    Document-level instances are written to ``{split}_doc.jsonl``
    alongside the sentence-level ``{split}.jsonl`` files.

Expected input format (one JSON object per line in each split file)::

    {
      "doc_key":   "ACL:2001/01",
      "sentences": [["This", "paper", "presents", ...], ["The", ...]],
      "ner":       [[[0, 1, "Method"], [4, 4, "Task"]], ...],
      "relations": [[[0, 1, 4, 4, "Used-for"]], ...]
    }

NER spans: ``[start, end_inclusive, type]`` within the sentence.
Relation spans: ``[h_start, h_end_inclusive, t_start, t_end_inclusive, type]``
within the sentence.  Both are converted to half-open ``[start, end)``
in output.

Usage::

    # Sentence-level only
    python -m s2g.data.preprocess_scierc \\
        --input_dir  data/raw/scierc \\
        --output_dir data/scierc

    # Sentence-level + document-level
    python -m s2g.data.preprocess_scierc \\
        --input_dir   data/raw/scierc \\
        --output_dir  data/scierc \\
        --document_level

Output (sentence-level)::

    data/scierc/
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    ├── entity.schema
    └── relation.schema

Additional output with ``--document_level``::

    data/scierc/
    ├── train_doc.jsonl
    ├── val_doc.jsonl
    └── test_doc.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ===================================================================== #
#                     SENTENCE-LEVEL CONVERSION                         #
# ===================================================================== #


def convert_document(doc: Dict) -> List[Dict]:
    """Convert a SciERC document to a list of sentence-level S2G instances.

    Each sentence that contains at least one entity becomes a separate
    instance.  Relation spans that exceed the current sentence's bounds
    are skipped with a debug-level log (handles both malformed data and
    distributions that use document-level offsets).

    Args:
        doc: Parsed JSON object with keys ``sentences``, ``ner``,
             ``relations`` (and optionally ``doc_key``).

    Returns:
        List of S2G instance dicts, one per sentence with entities.
    """
    sentences:    List[List[str]]     = doc["sentences"]
    ner_per_sent: List[List[List]]    = doc.get("ner",       [[] for _ in sentences])
    rel_per_sent: List[List[List]]    = doc.get("relations", [[] for _ in sentences])
    doc_key:      str                 = doc.get("doc_key", "")

    instances: List[Dict] = []
    for sent_idx, tokens in enumerate(sentences):
        n_tokens  = len(tokens)
        ner_spans = ner_per_sent[sent_idx] if sent_idx < len(ner_per_sent) else []
        rel_spans = rel_per_sent[sent_idx] if sent_idx < len(rel_per_sent) else []

        inst = _convert_sentence(tokens, ner_spans, rel_spans, n_tokens, doc_key, sent_idx)
        if inst is not None:
            instances.append(inst)

    return instances


def _convert_sentence(
    tokens:    List[str],
    ner_spans: List[List],
    rel_spans: List[List],
    n_tokens:  int,
    doc_key:   str,
    sent_idx:  int,
) -> Optional[Dict]:
    """Convert one sentence to an S2G instance dict.

    Spans use inclusive end indices in the SciERC format; these are
    converted to half-open ``[start, end)`` in the output.
    """
    if not ner_spans:
        return None

    text = " ".join(tokens)
    span_key_to_idx: Dict[Tuple[int, int], int] = {}
    entities: List[Dict] = []

    for span in ner_spans:
        s_inc, e_inc, ent_type = int(span[0]), int(span[1]), span[2]
        if s_inc > e_inc or e_inc >= n_tokens:
            logger.debug("%s sent %d: skipping out-of-bounds NER span [%d, %d].",
                         doc_key, sent_idx, s_inc, e_inc)
            continue
        start, end = s_inc, e_inc + 1       # → half-open
        key = (start, end)
        if key not in span_key_to_idx:
            span_key_to_idx[key] = len(entities)
            entities.append({
                "text":   " ".join(tokens[start:end]),
                "offset": [start, end],
                "type":   ent_type,
            })

    if not entities:
        return None

    relations:     List[Dict] = []
    skipped_cross: int        = 0

    for rel in rel_spans:
        h_s, h_e, t_s, t_e, rel_type = (
            int(rel[0]), int(rel[1]), int(rel[2]), int(rel[3]), rel[4]
        )
        if h_e >= n_tokens or t_e >= n_tokens:
            skipped_cross += 1
            continue
        h_key = (h_s, h_e + 1)
        t_key = (t_s, t_e + 1)
        if h_key not in span_key_to_idx or t_key not in span_key_to_idx:
            logger.debug("%s sent %d: relation references unknown entity span.",
                         doc_key, sent_idx)
            continue
        relations.append({
            "head": entities[span_key_to_idx[h_key]],
            "tail": entities[span_key_to_idx[t_key]],
            "type": rel_type,
        })

    if skipped_cross:
        logger.debug("%s sent %d: skipped %d out-of-bounds / cross-sentence relation(s).",
                     doc_key, sent_idx, skipped_cross)

    return {
        "text":         text,
        "tokens":       tokens,
        "entities":     entities,
        "relations":    relations,
        "entity_types": sorted(set(e["type"] for e in entities)),
        "rel_types":    sorted(set(r["type"] for r in relations)),
    }


# ===================================================================== #
#                    DOCUMENT-LEVEL CONVERSION                          #
# ===================================================================== #


def convert_document_level(doc: Dict) -> Optional[Dict]:
    """Convert a SciERC document to a single document-level S2G instance.

    Concatenates all sentence token lists and converts all NER and
    relation spans to document-level offsets by accumulating sentence
    lengths.  Relations remain intra-sentence after conversion (no new
    cross-sentence relations are introduced, since the standard SciERC
    annotation does not store them in the ``relations`` field).

    The benefit over sentence-level processing is broader encoder
    context: the model attends to the full abstract when encoding any
    entity or relation span.

    Args:
        doc: Parsed JSON object with keys ``sentences``, ``ner``,
             ``relations`` (and optionally ``doc_key``).

    Returns:
        A single S2G instance dict, or ``None`` if the document has no
        entities.
    """
    sentences:    List[List[str]]  = doc["sentences"]
    ner_per_sent: List[List[List]] = doc.get("ner",       [[] for _ in sentences])
    rel_per_sent: List[List[List]] = doc.get("relations", [[] for _ in sentences])
    doc_key:      str              = doc.get("doc_key", "")

    # Build flat document token list and sentence start offsets.
    doc_tokens: List[str] = []
    sent_offsets: List[int] = []
    for sent in sentences:
        sent_offsets.append(len(doc_tokens))
        doc_tokens.extend(sent)
    n_doc = len(doc_tokens)

    if n_doc == 0:
        return None

    # Collect entities with document-level offsets (deduplicate by span key).
    span_to_ent: Dict[Tuple[int, int], Dict] = {}
    entities:    List[Dict] = []

    for sent_idx, ner_spans in enumerate(ner_per_sent):
        sent_off = sent_offsets[sent_idx]
        n_sent   = len(sentences[sent_idx])
        for span in ner_spans:
            s_inc, e_inc, ent_type = int(span[0]), int(span[1]), span[2]
            if s_inc > e_inc or e_inc >= n_sent:
                logger.debug("%s sent %d: skipping out-of-bounds NER span.", doc_key, sent_idx)
                continue
            doc_start = sent_off + s_inc
            doc_end   = sent_off + e_inc + 1   # half-open
            key = (doc_start, doc_end)
            if key not in span_to_ent:
                ent = {
                    "text":   " ".join(doc_tokens[doc_start:doc_end]),
                    "offset": [doc_start, doc_end],
                    "type":   ent_type,
                }
                span_to_ent[key] = ent
                entities.append(ent)

    if not entities:
        return None

    # Collect relations with document-level offsets.
    relations: List[Dict] = []
    for sent_idx, rel_spans in enumerate(rel_per_sent):
        sent_off = sent_offsets[sent_idx]
        n_sent   = len(sentences[sent_idx])
        for rel in rel_spans:
            h_s, h_e, t_s, t_e, rel_type = (
                int(rel[0]), int(rel[1]), int(rel[2]), int(rel[3]), rel[4]
            )
            # Skip if spans exceed the current sentence (malformed data).
            if h_e >= n_sent or t_e >= n_sent:
                logger.debug("%s sent %d: skipping cross-sentence / out-of-bounds relation.",
                             doc_key, sent_idx)
                continue
            h_key = (sent_off + h_s, sent_off + h_e + 1)
            t_key = (sent_off + t_s, sent_off + t_e + 1)
            if h_key not in span_to_ent or t_key not in span_to_ent:
                continue
            relations.append({
                "head": span_to_ent[h_key],
                "tail": span_to_ent[t_key],
                "type": rel_type,
            })

    return {
        "text":         " ".join(doc_tokens),
        "tokens":       doc_tokens,
        "entities":     entities,
        "relations":    relations,
        "entity_types": sorted(set(e["type"] for e in entities)),
        "rel_types":    sorted(set(r["type"] for r in relations)),
    }


# ===================================================================== #
#                        SPLIT PROCESSING                               #
# ===================================================================== #


def process_split(
    input_path:     Path,
    sent_out_path:  Path,
    doc_out_path:   Optional[Path],
) -> Tuple[List[str], List[str]]:
    """Convert all documents in *input_path* and write output JSONL files.

    Args:
        input_path:    Input JSONL file (one document per line).
        sent_out_path: Output path for sentence-level instances.
        doc_out_path:  Output path for document-level instances, or
                       ``None`` to skip document-level processing.

    Returns:
        ``(entity_types, rel_types)`` — all types seen in this split.
    """
    seen_ent: List[str] = []
    seen_rel: List[str] = []
    sent_written = 0
    doc_written  = 0

    sent_fh = open(sent_out_path, "w", encoding="utf-8")
    doc_fh  = open(doc_out_path,  "w", encoding="utf-8") if doc_out_path else None

    try:
        with open(input_path, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)

                # Sentence-level
                for inst in convert_document(doc):
                    sent_fh.write(json.dumps(inst, ensure_ascii=False) + "\n")
                    seen_ent.extend(inst["entity_types"])
                    seen_rel.extend(inst["rel_types"])
                    sent_written += 1

                # Document-level (optional)
                if doc_fh is not None:
                    inst = convert_document_level(doc)
                    if inst is not None:
                        doc_fh.write(json.dumps(inst, ensure_ascii=False) + "\n")
                        doc_written += 1
    finally:
        sent_fh.close()
        if doc_fh is not None:
            doc_fh.close()

    msg = "%s → %s  (%d sentences written)"
    if doc_out_path:
        msg += f", {doc_written} documents → {doc_out_path.name}"
    logger.info(msg, input_path.name, sent_out_path.name, sent_written)

    return seen_ent, seen_rel


# ===================================================================== #
#                              MAIN                                     #
# ===================================================================== #


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess SciERC for S2G fine-tuning."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing train.json, dev.json, test.json.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory for output JSONL and schema files.",
    )
    parser.add_argument(
        "--document_level", action="store_true",
        help=(
            "Also produce document-level JSONL files (train_doc.jsonl, etc.) "
            "alongside the standard sentence-level files.  Results from the "
            "document-level files are NOT comparable to sentence-level benchmarks."
        ),
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train.json": "train.jsonl",
        "dev.json":   "val.jsonl",
        "test.json":  "test.jsonl",
    }

    for in_name, out_name in split_map.items():
        in_path   = input_dir  / in_name
        sent_path = output_dir / out_name
        doc_path  = (output_dir / out_name.replace(".jsonl", "_doc.jsonl")
                     if args.document_level else None)

        if not in_path.exists():
            logger.warning("Input file not found, skipping: %s", in_path)
            continue
        process_split(in_path, sent_path, doc_path)

    # Schema from training split only.
    train_path = input_dir / "train.json"
    if train_path.exists():
        train_ent, train_rel = [], []
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for inst in convert_document(json.loads(line)):
                    train_ent.extend(inst["entity_types"])
                    train_rel.extend(inst["rel_types"])
        _write_schema(output_dir / "entity.schema",   train_ent)
        _write_schema(output_dir / "relation.schema", train_rel)

    logger.info("Done. Output written to %s", output_dir)


def _write_schema(path: Path, types: List[str]) -> None:
    unique = sorted(set(types))
    with open(path, "w", encoding="utf-8") as f:
        for t in unique:
            f.write(t + "\n")
    logger.info("Schema: %s  (%d types)", path.name, len(unique))


if __name__ == "__main__":
    main()