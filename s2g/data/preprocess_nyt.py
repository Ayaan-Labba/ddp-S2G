"""
Pre-processing script for the NYT-multi dataset (from the JointRE repo) into S2G fine-tuning format.
Preprocessing logic matches REBEL's nyt_typed.py: relations are sorted by head entity start position,
and all rows are yielded unconditionally.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set

import yaml

logger = logging.getLogger(__name__)


def load_label_maps(config_map_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    if not config_map_path:
        return {}, {}
    path = Path(config_map_path)
    if not path.exists():
        logger.warning(f"Config map file {config_map_path} not found. Using raw labels.")
        return {}, {}

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        return {}, {}

    entities = config.get("entities", {}) or {}
    relations = config.get("relations", {}) or {}
    return {str(k): str(v) for k, v in entities.items()}, {str(k): str(v) for k, v in relations.items()}


def convert_instance(raw: Dict, entity_map: Dict[str, str], relation_map: Dict[str, str]) -> Dict:
    tokens = raw["tokens"]
    text = " ".join(tokens)
    entities_registry: Dict[Tuple[int, int], Dict] = {}
    relations = []

    relations_sorted = sorted(zip(raw.get("spo_list", []), raw.get("spo_details", [])), key=lambda tup: tup[1][0])
    for relation, details in relations_sorted:
        # details layout: [head_start, head_end, head_type, rel_type, tail_start, tail_end, tail_type]
        h_start, h_end, h_type = int(details[0]), int(details[1]), details[2]
        t_start, t_end, t_type = int(details[4]), int(details[5]), details[6]

        h_key = (h_start, h_end)
        if h_key not in entities_registry:
            entities_registry[h_key] = {
                "text": " ".join(tokens[h_start:h_end]),
                "offset": [h_start, h_end],
                "type": entity_map.get(h_type, h_type)
            }

        t_key = (t_start, t_end)
        if t_key not in entities_registry:
            entities_registry[t_key] = {
                "text": " ".join(tokens[t_start:t_end]),
                "offset": [t_start, t_end],
                "type": entity_map.get(t_type, t_type)
            }

        head_obj = entities_registry[h_key]
        tail_obj = entities_registry[t_key]

        raw_rel_type = relation[1].split("/")[-1]
        mapped_rel_type = relation_map.get(raw_rel_type, raw_rel_type)
        relations.append({"head": head_obj, "tail": tail_obj, "type": mapped_rel_type})

    # entities may be empty for relation-free rows — that is intentional (mirrors REBEL)
    entities = sorted(entities_registry.values(), key=lambda e: e["offset"])

    return {
        "text": text,
        "tokens": tokens,
        "entities": entities,
        "relations": relations,
        "entity_types": sorted({e["type"] for e in entities}),
        "rel_types": sorted({r["type"] for r in relations}),
    }


def process_split(
    input_path: Path, output_path: Path,
    entity_map: Dict[str, str], relation_map: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    seen_ent: Set[str] = set()
    seen_rel: Set[str] = set()
    written = 0

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for raw in json.load(fin):
            inst = convert_instance(raw, entity_map, relation_map)
            fout.write(json.dumps(inst, ensure_ascii=False) + "\n")
            seen_ent.update(inst["entity_types"])
            seen_rel.update(inst["rel_types"])
            written += 1

    logger.info("%s → %s (%d written)", input_path.name, output_path.name, written)
    return list(seen_ent), list(seen_rel)


def _write_schema(path: Path, types: List[str]) -> None:
    unique = sorted(set(types))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(unique) + "\n")
    logger.info("Schema: %s (%d types)", path.name, len(unique))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Preprocess dataset for S2G fine-tuning.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_map", default="configs/data/nyt.yaml", help="Path to the config YAML for label mapping.")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entity_map, relation_map = load_label_maps(args.config_map)

    all_ent, all_rel = [], []

    split_mapping = [
        ("train", input_dir / "train.json", "train.jsonl"),
        ("dev",   input_dir / "dev.json",   "val.jsonl"),
        ("test",  input_dir / "test.json",  "test.jsonl"),
    ]

    for split_name, input_path, out_name in split_mapping:
        e, r = process_split(input_path, output_dir / out_name, entity_map, relation_map)
        if split_name == "train":
            all_ent, all_rel = e, r

    _write_schema(output_dir / "entity.schema", all_ent)
    _write_schema(output_dir / "relation.schema", all_rel)


if __name__ == "__main__":
    main()