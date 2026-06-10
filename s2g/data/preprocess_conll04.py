"""
Pre-processing script for SpERT-format datasets (CoNLL-04).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


def convert_instance(raw: Dict) -> Optional[Dict]:
    if not raw.get("entities"): return None
    
    tokens = raw["tokens"]
    entities = [{
        "text": " ".join(tokens[int(e["start"]):int(e["end"])]),
        "offset": [int(e["start"]), int(e["end"])],
        "type": e["type"]
    } for e in raw["entities"]]

    relations = [{
        "head": entities[int(r["head"])], "tail": entities[int(r["tail"])], "type": r["type"]
    } for r in raw.get("relations", []) if int(r["head"]) < len(entities) and int(r["tail"]) < len(entities)]

    return {
        "text": " ".join(tokens), "tokens": tokens, "entities": entities, "relations": relations,
        "entity_types": sorted({e["type"] for e in entities}), "rel_types": sorted({r["type"] for r in relations}),
    }


def process_split(split_name: str, instances: List[Dict], output_path: Path) -> Tuple[List[str], List[str]]:
    seen_ent: Set[str] = set()
    seen_rel: Set[str] = set()
    skipped, written = 0, 0
    
    with open(output_path, "w", encoding="utf-8") as fout:
        # Iterate directly over the list of instances for this split
        for raw in instances:
            if inst := convert_instance(raw):
                fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
                seen_ent.update(inst["entity_types"])
                seen_rel.update(inst["rel_types"])
                written += 1
            else: 
                skipped += 1
            
    logger.info(
        "Split: %s → %s (%d written, %d skipped)", 
        split_name, 
        output_path.name, 
        written,
        skipped
    )
    return list(seen_ent), list(seen_rel)


def _write_schema(path: Path, types: List[str]) -> None:
    unique = sorted(set(types))
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(unique) + "\n")
    logger.info("Schema: %s (%d types)", path.name, len(unique))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Preprocess dataset for S2G fine-tuning.")
    parser.add_argument("--input_file", required=True, help="Path to coll04.json") 
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the single JSON file containing all splits
    with open(input_file, encoding="utf-8") as fin:
        dataset = json.load(fin)

    all_ent, all_rel = [], []
    
    # Map the JSON keys to your desired output file names
    split_mapping = {"train": "train.jsonl", "dev": "val.jsonl", "test": "test.jsonl"}
    
    for split_key, out_name in split_mapping.items():
        if split_key in dataset:
            e, r = process_split(split_key, dataset[split_key], output_dir / out_name)
            if split_key == "train": 
                all_ent, all_rel = e, r # Prefer training schema

    _write_schema(output_dir / "entity.schema", all_ent)
    _write_schema(output_dir / "relation.schema", all_rel)


if __name__ == "__main__":
    main()