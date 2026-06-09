"""
Pre-processing script for the CoNLL 2004 dataset.

Reads the JSON format produced by Eberts & Ulges (SpERT, 2020), which is
the standard pre-processed form used by recent joint entity-relation
extraction papers.  Converts each sentence into the S2G JSONL format and
writes entity/relation schema files.

Expected input format (one JSON array per file)::

    [
      {
        "tokens":    ["John", "Smith", "works", "for", "Microsoft", "."],
        "entities":  [{"type": "Peop", "start": 0, "end": 2},
                      {"type": "Org",  "start": 4, "end": 5}],
        "relations": [{"type": "Work_For", "head": 0, "tail": 1}]
      },
      ...
    ]

Where ``entities[i]["start"]`` and ``["end"]`` are token-level half-open
``[start, end)`` intervals, and ``relations[i]["head"]`` / ``["tail"]``
are indices into the ``entities`` list.

Usage::

    python -m s2g.data.preprocess_conll04 \\
        --input_dir  data/raw/conll04 \\
        --output_dir data/conll04

The input directory must contain ``train.json``, ``dev.json``,
``test.json``.  The output directory will contain::

    data/conll04/
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    ├── entity.schema
    └── relation.schema
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ===================================================================== #
#                        INSTANCE CONVERSION                            #
# ===================================================================== #


def convert_instance(raw: Dict) -> Optional[Dict]:
    """Convert a single SpERT-format sentence dict to S2G JSONL format.

    Args:
        raw: Dict with keys ``tokens``, ``entities``, ``relations``.

    Returns:
        S2G instance dict, or ``None`` if the sentence has no entities.
    """
    tokens: List[str] = raw["tokens"]
    text   = " ".join(tokens)

    raw_entities  = raw.get("entities",  [])
    raw_relations = raw.get("relations", [])

    if not raw_entities:
        return None

    # Build entity dicts (offset end is already exclusive in SpERT format).
    entities: List[Dict] = []
    for ent in raw_entities:
        start = int(ent["start"])
        end   = int(ent["end"])
        span  = " ".join(tokens[start:end])
        entities.append(
            {
                "text":   span,
                "offset": [start, end],
                "type":   ent["type"],
            }
        )

    # Build relation dicts using entity-index references.
    relations: List[Dict] = []
    for rel in raw_relations:
        h_idx = int(rel["head"])
        t_idx = int(rel["tail"])
        if h_idx >= len(entities) or t_idx >= len(entities):
            logger.debug("Skipping relation with out-of-range entity index.")
            continue
        relations.append(
            {
                "head": entities[h_idx],
                "tail": entities[t_idx],
                "type": rel["type"],
            }
        )

    entity_types = sorted(set(e["type"] for e in entities))
    rel_types    = sorted(set(r["type"] for r in relations))

    return {
        "text":         text,
        "tokens":       tokens,
        "entities":     entities,
        "relations":    relations,
        "entity_types": entity_types,
        "rel_types":    rel_types,
    }


# ===================================================================== #
#                        SPLIT PROCESSING                               #
# ===================================================================== #


def process_split(
    input_path: Path,
    output_path: Path,
) -> Tuple[List[str], List[str]]:
    """Convert one split file and return (entity_types, rel_types) seen.

    Args:
        input_path:  Path to the input JSON file.
        output_path: Path to the output JSONL file.

    Returns:
        ``(entity_types, rel_types)`` — sorted lists of all types seen
        in this split, for schema accumulation.
    """
    with open(input_path, encoding="utf-8") as f:
        raw_instances = json.load(f)

    seen_ent_types: List[str] = []
    seen_rel_types: List[str] = []
    written = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for raw in raw_instances:
            inst = convert_instance(raw)
            if inst is None:
                skipped += 1
                continue
            out.write(json.dumps(inst, ensure_ascii=False) + "\n")
            seen_ent_types.extend(inst["entity_types"])
            seen_rel_types.extend(inst["rel_types"])
            written += 1

    logger.info(
        "%s → %s  (%d written, %d skipped)",
        input_path.name, output_path.name, written, skipped,
    )
    return seen_ent_types, seen_rel_types


# ===================================================================== #
#                              MAIN                                     #
# ===================================================================== #


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess CoNLL 2004 for S2G fine-tuning."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing train.json, dev.json, test.json.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory for train.jsonl, val.jsonl, test.jsonl, *.schema.",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ent_types: List[str] = []
    all_rel_types: List[str] = []

    split_map = {
        "train.json": "train.jsonl",
        "dev.json":   "val.jsonl",
        "test.json":  "test.jsonl",
    }

    for in_name, out_name in split_map.items():
        in_path  = input_dir  / in_name
        out_path = output_dir / out_name
        if not in_path.exists():
            logger.warning("Input file not found, skipping: %s", in_path)
            continue
        ent_types, rel_types = process_split(in_path, out_path)
        all_ent_types.extend(ent_types)
        all_rel_types.extend(rel_types)

    # Write schema files from training split only (consistent with benchmark
    # practice: schema is defined by the training set).
    train_path = input_dir / "train.json"
    if train_path.exists():
        with open(train_path, encoding="utf-8") as f:
            train_raw = json.load(f)
        train_ent: List[str] = []
        train_rel: List[str] = []
        for raw in train_raw:
            inst = convert_instance(raw)
            if inst:
                train_ent.extend(inst["entity_types"])
                train_rel.extend(inst["rel_types"])
        _write_schema(output_dir / "entity.schema",   train_ent)
        _write_schema(output_dir / "relation.schema", train_rel)
    else:
        # Fallback: use all splits.
        _write_schema(output_dir / "entity.schema",   all_ent_types)
        _write_schema(output_dir / "relation.schema", all_rel_types)

    logger.info("Done. Output written to %s", output_dir)


def _write_schema(path: Path, types: List[str]) -> None:
    """Write a sorted, deduplicated list of type strings, one per line."""
    unique = sorted(set(types))
    with open(path, "w", encoding="utf-8") as f:
        for t in unique:
            f.write(t + "\n")
    logger.info("Schema: %s  (%d types)", path.name, len(unique))


if __name__ == "__main__":
    main()