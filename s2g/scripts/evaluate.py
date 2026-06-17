"""
Standalone evaluation script for the S2G model.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from s2g.data import S2GDataset
from s2g.evaluation import compute_metrics_for_task
from s2g.linearisation import (
    S2GTokens, AnyTokens, EntityBlock, VARIANT_TO_TASKS,
    add_special_tokens_to_tokenizer,
    build_boundary_joint_encoder_input, build_joint_encoder_input,
    build_re_encoder_input, build_boundary_re_encoder_input,
    extract_triplets, find_all_token_spans, parse_sel,
)
from s2g.model import build_constraint_processor
from s2g.scripts.config_utils import load_config, load_entity_schema, load_schema

logger = logging.getLogger(__name__)


def _generate_batch(
    model: Any, tokenizer: Any, encoder_inputs: List[str], tokens: AnyTokens,
    max_source_length: int, max_target_length: int, eval_beams: int, device: torch.device,
    constraint_decoding: bool = False, entity_schema: Optional[List[str]] = None, rel_schema: Optional[List[str]] = None
) -> List[List[EntityBlock]]:
    tok_out = tokenizer(
        encoder_inputs, max_length=max_source_length, truncation=True, padding="longest", return_tensors="pt"
    ).to(device, non_blocking=True)
    
    gen_kwargs = {**tok_out, "num_beams": eval_beams, "max_length": max_target_length, "length_penalty": 0.0, "no_repeat_ngram_size": 0, "early_stopping": False}

    if constraint_decoding:
        gen_kwargs["logits_processor"] = [build_constraint_processor(tokenizer, tok_out["input_ids"], tokens, eval_beams, entity_schema=entity_schema, rel_schema=rel_schema)]

    dtype = next(model.parameters()).dtype
    ctx = torch.autocast(device.type, dtype) if dtype in {torch.bfloat16, torch.float16} and device.type == "cuda" else contextlib.nullcontext()
    
    with torch.inference_mode(), ctx: 
        generated = model.generate(**gen_kwargs)
    specials = [tok for tok in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token) if tok]
    all_entities = []
    
    for text in tokenizer.batch_decode(generated, skip_special_tokens=False):
        for tok in specials: 
            text = text.replace(tok, "")
        ents, _ = parse_sel(" ".join(text.split()), tok=tokens)
        all_entities.append(ents)
        
    return all_entities


def _to_spans(source_tokens: List[str], entities: List[EntityBlock]) -> List[Tuple[int, int]]:
    return list(dict.fromkeys(span for e in entities for span in find_all_token_spans(source_tokens, e["text"])))


def _to_entity_data(source_tokens: List[str], entities: List[EntityBlock], use_type: bool = True) -> List[Tuple[int, int, str]]:
    return list(dict.fromkeys(
        (*span, e["type"] if (use_type and e.get("type")) else "") 
        for e in entities if (not use_type or e.get("type")) 
        for span in find_all_token_spans(source_tokens, e["text"])
    ))


def _evaluate_pipeline(model, tokenizer, instances, entity_schema, rel_schema, tokens, max_source_length, max_target_length, batch_size, eval_beams, device, constraint_decoding, tasks=None, ssi_prompt="ssi") -> Tuple[Dict[str, Any], Dict[str, float]]:
    if tasks is None:
        tasks = ["re"]
    use_re = "re" in tasks
    use_boundary_re = "boundary_re" in tasks

    def _run(inputs):
        return [ent for i in tqdm(range(0, len(inputs), batch_size), leave=False) 
                for ent in _generate_batch(model, tokenizer, inputs[i:i+batch_size], tokens, max_source_length, max_target_length, eval_beams, device, constraint_decoding, entity_schema, rel_schema)]
    
    r_per_inst = []
    ner_maps = []
    if use_re or use_boundary_re:
        r_inputs = []
        for inst in instances:
            if use_re:
                entity_data = [(int(e["offset"][0]), int(e["offset"][1]), e.get("type", "")) for e in inst["entities"]]
                ner_maps.append({e["text"]: e.get("type", "") for e in inst["entities"]})
            else:
                entity_data = [(int(e["offset"][0]), int(e["offset"][1]), "") for e in inst["entities"]]
                ner_maps.append({e["text"]: "" for e in inst["entities"]})
            if tokens.variant == "re":
                r_inputs.append(build_re_encoder_input(entity_schema, rel_schema, inst["text"], False, tokens, ssi_prompt=ssi_prompt))
            elif tokens.variant == "boundary_re":
                r_inputs.append(build_boundary_re_encoder_input(rel_schema, inst["text"], False, tokens, ssi_prompt=ssi_prompt))
        r_per_inst = _run(r_inputs)
    else:
        r_per_inst = [[] for _ in instances]
        ner_maps = [{} for _ in instances]

    per_inst, m = [], {}
    for i, inst in enumerate(instances):
        res = {
            "text": inst["text"],
            "gold_triplets": [{"head": r["head"]["text"], "type": r["type"], "tail": r["tail"]["text"]} for r in inst["relations"]]
        }
        if use_re or use_boundary_re:
            res["re_triplets"] = [{"head": t[0], "type": t[1], "tail": t[2]} for t in extract_triplets(r_per_inst[i])]
        per_inst.append(res)
        
    if use_re:
        g_quints = [[(r["head"]["text"], r["head"].get("type",""), r["type"], r["tail"]["text"], r["tail"].get("type","")) for r in inst["relations"]] for inst in instances]
        m.update(compute_metrics_for_task("re", rel_schema=rel_schema, all_pred_triplets=[extract_triplets(r) for r in r_per_inst], all_gold_triplets=[[(r["head"]["text"], r["type"], r["tail"]["text"]) for r in inst["relations"]] for inst in instances], all_pred_quintuples=[[(e["text"], ner_maps[i].get(e["text"], ""), rel["type"], rel["tail"], ner_maps[i].get(rel["tail"], "")) for e in r_per_inst[i] for rel in e["relations"]] for i in range(len(instances))], all_gold_quintuples=g_quints))

    if use_boundary_re:
        m.update(compute_metrics_for_task("boundary_re", rel_schema=rel_schema, all_pred_triplets=[extract_triplets(r) for r in r_per_inst], all_gold_triplets=[[(r["head"]["text"], r["type"], r["tail"]["text"]) for r in inst["relations"]] for inst in instances]))

    return per_inst, m


def _evaluate_boundary_joint(model, tokenizer, instances, entity_schema, rel_schema, tokens, max_source_length, max_target_length, batch_size, eval_beams, device, constraint_decoding, tasks=None, ssi_prompt="ssi") -> Tuple[Dict[str, Any], Dict[str, float]]:
    if tasks is None:
        tasks = ["boundary_joint", "joint"]
    use_boundary_joint = "boundary_joint" in tasks
    use_joint = "joint" in tasks

    def _run(inputs):
        return [ent for i in tqdm(range(0, len(inputs), batch_size), leave=False) 
                for ent in _generate_batch(model, tokenizer, inputs[i:i+batch_size], tokens, max_source_length, max_target_length, eval_beams, device, constraint_decoding, entity_schema, rel_schema)]

    j_per_inst = []
    if use_boundary_joint:
        j_per_inst  = _run([build_boundary_joint_encoder_input(rel_schema, inst["text"], False, tokens, ssi_prompt=ssi_prompt) for inst in instances])
    else:
        j_per_inst = [[] for _ in instances]

    jp_per_inst = []
    if use_joint:
        jp_per_inst = _run([build_joint_encoder_input(entity_schema, rel_schema, inst["text"], False, tokens, ssi_prompt=ssi_prompt) for inst in instances])
    else:
        jp_per_inst = [[] for _ in instances]

    gold_trips = [[(r["head"]["text"], r["type"], r["tail"]["text"]) for r in inst["relations"]] for inst in instances]
    gold_quints = [[(r["head"]["text"], r["head"].get("type",""), r["type"], r["tail"]["text"], r["tail"].get("type","")) for r in inst["relations"]] for inst in instances]

    per_inst, m = [], {}
    for i, inst in enumerate(instances):
        res = {
            "text": inst["text"],
            "gold_triplets": [{"head": r["head"]["text"], "type": r["type"], "tail": r["tail"]["text"]} for r in inst["relations"]]
        }
        if use_boundary_joint:
            res["boundary_joint_triplets"] = [{"head": t[0], "type": t[1], "tail": t[2]} for t in extract_triplets(j_per_inst[i])]
        if use_joint:
            res["joint"] = {"entities": [{"text": e["text"], "type": e.get("type")} for e in jp_per_inst[i]], "triplets": [{"head": t[0], "type": t[1], "tail": t[2]} for t in extract_triplets(jp_per_inst[i])]}
        per_inst.append(res)

    if use_boundary_joint:
        m.update(compute_metrics_for_task("boundary_joint", rel_schema=rel_schema, all_pred_triplets=[extract_triplets(j) for j in j_per_inst], all_gold_triplets=gold_trips))
    
    if use_joint:
        jp_maps = [{e["text"]: e.get("type", "") for e in jp} for jp in jp_per_inst]
        m.update(compute_metrics_for_task(
            "joint", 
            rel_schema=rel_schema,
            entity_schema=entity_schema,
            all_pred_triplets=[extract_triplets(jp) for jp in jp_per_inst], 
            all_gold_triplets=gold_trips, 
            all_pred_quintuples=[[(e["text"], jp_maps[i].get(e["text"], ""), rel["type"], rel["tail"], rel.get("tail_type") or jp_maps[i].get(rel["tail"], "")) for e in jp_per_inst[i] for rel in e["relations"]] for i in range(len(instances))], 
            all_gold_quintuples=gold_quints, 
            all_pred_entities=[[e["text"] for e in jp] for jp in jp_per_inst], 
            all_gold_entities=[[e["text"] for e in inst["entities"]] for inst in instances], 
            all_pred_entity_mentions=[[(e["text"], e.get("type") or "") for e in jp if e.get("type")] for jp in jp_per_inst], 
            all_gold_entity_mentions=[[(e["text"], e.get("type", "")) for e in inst["entities"]] for inst in instances]
        ))

    return per_inst, m


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = load_config()
    if cfg.hardware.gpu_ids is not None: 
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.hardware.gpu_ids))
    set_seed(cfg.train.seed)

    if not (ckpt := cfg.model.pretrained_checkpoint): 
        raise ValueError("model.pretrained_checkpoint is required.")

    tokenizer, model = AutoTokenizer.from_pretrained(ckpt), AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    model_variant = (Path(ckpt) / "model_variant.txt").read_text(encoding="utf-8").strip() if (Path(ckpt) / "model_variant.txt").exists() else cfg.model.model_variant
    
    if (Path(ckpt) / "tasks.json").exists():
        with open(Path(ckpt) / "tasks.json", "r", encoding="utf-8") as f:
            tasks = json.load(f)
    elif (Path(ckpt) / "tasks.txt").exists():
        tasks = [t.strip() for t in (Path(ckpt) / "tasks.txt").read_text(encoding="utf-8").strip().split(",") if t.strip()]
    else:
        tasks = VARIANT_TO_TASKS[model_variant]

    use_rejection = "<null>" in tokenizer.get_vocab()
    tokens = S2GTokens(model_variant, use_rejection=use_rejection)
    add_special_tokens_to_tokenizer(tokenizer, tokens, model, warm=cfg.sel.warm_start)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rel_schema, entity_schema = load_schema(cfg.data.schema_file), load_entity_schema(cfg.data.entity_schema_file)
    instances = [inst for inst in S2GDataset(Path(cfg.data.data_dir) / f"{cfg.evaluation.split}.jsonl", seed=cfg.train.seed)]
    
    eval_fn = _evaluate_pipeline if model_variant in {"re", "boundary_re"} else _evaluate_boundary_joint
    per_inst_results, metrics = eval_fn(
        model, tokenizer, instances, entity_schema, rel_schema, tokens,
        cfg.tokenization.max_source_length, cfg.tokenization.max_target_length,
        cfg.validation.batch_size, cfg.generation.num_beams, device, cfg.generation.constraint_decoding,
        tasks, ssi_prompt=cfg.ssi.ssi_prompt
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        out_dir = Path(cfg.data.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{cfg.evaluation.split}_results.jsonl", "w", encoding="utf-8") as f:
            for r in per_inst_results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out_dir / f"{cfg.evaluation.split}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()