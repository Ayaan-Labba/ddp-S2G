"""
Pre-processing script for the NYT-multi dataset.

Reads the JSON format commonly distributed for NYT-multi (same schema as
the SpERT / CoNLL 2004 format), converts each sentence to the S2G JSONL
format, and writes entity/relation schema files.

Expected input format (one JSON array per file)::

    [
      {
        "tokens":    ["Obama", "was", "born", "in", "Hawaii", "."],
        "entities":  [{"type": "PERSON",   "start": 0, "end": 1},
                      {"type": "LOCATION", "start": 4, "end": 5}],
        "relations": [{"type": "/people/person/place_of_birth",
                       "head": 0, "tail": 1}]
      },
      ...
    ]

``entities[i]["end"]`` is exclusive (half-open interval).
``relations[i]["head"]`` / ``["tail"]`` are indices into ``entities``.

Relation types are kept verbatim (e.g. ``/people/person/nationality``).
Some distributions use simplified labels; the script works with either.

Usage::

    python -m s2g.data.preprocess_nyt_multi \\
        --input_dir  data/raw/nyt_multi \\
        --output_dir data/nyt_multi

Input directory must contain ``train.json``, ``dev.json``, ``test.json``.
Output::

    data/nyt_multi/
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


# ---- INSTANCE CONVERSION ----


def convert_instance(raw: Dict) -> Optional[Dict]:
    """Convert a single SpERT-format sentence dict to S2G JSONL format.

    Identical logic to the CoNLL 2004 converter; factored here so each
    dataset script is self-contained.

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


# ---- SPLIT PROCESSING ----


def process_split(
    input_path: Path,
    output_path: Path,
) -> Tuple[List[str], List[str]]:
    """Convert one split and return (entity_types, rel_types) encountered."""
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


# ---- MAIN ----


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess NYT-multi for S2G fine-tuning."
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
        process_split(in_path, out_path)

    # Schema from training split only.
    train_path = input_dir / "train.json"
    if train_path.exists():
        with open(train_path, encoding="utf-8") as f:
            train_raw = json.load(f)
        train_ent, train_rel = [], []
        for raw in train_raw:
            inst = convert_instance(raw)
            if inst:
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