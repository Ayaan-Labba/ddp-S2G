"""
Standalone evaluation script for the S2G model.

Loads a trained checkpoint, runs sequential pipeline evaluation (or Joint
model evaluation) on a specified data split, computes all applicable
metrics, and writes structured output files.

For the Pipeline model, evaluation chains Boundary → NER → RE, using each
stage's predictions as the next stage's augmented encoder input — identical
to the evaluation logic in S2GTrainer.  The key addition here is optional
FSM constraint decoding.

Output files written to ``cfg.data.output_dir``::

    {split}_results.jsonl  — Per-instance structured predictions.
    {split}_metrics.json   — All corpus-level micro and macro metrics.

Usage::

    python -m s2g.scripts.evaluate \\
        --config configs/evaluate.yaml \\
        model.pretrained_checkpoint=outputs/finetune/conll04/best_model \\
        evaluation.split=test

    # With constraint decoding
    python -m s2g.scripts.evaluate \\
        --config configs/evaluate.yaml \\
        model.pretrained_checkpoint=outputs/finetune/conll04/best_model \\
        generation.constraint_decoding=true
"""

from __future__ import annotations

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
    AnyTokens,
    EntityBlock,
    JOINT_TOKENS,
    PIPELINE_TOKENS,
    add_special_tokens_to_tokenizer,
    build_boundary_encoder_input,
    build_joint_encoder_input,
    build_joint_plus_encoder_input,
    build_ner_encoder_input,
    build_re_encoder_input,
    extract_triplets,
    find_all_token_spans,
    find_token_span,
    get_token_ids,
    parse_sel,
)
from s2g.model import build_constraint_processor
from s2g.scripts.config_utils import (
    load_config,
    load_entity_schema,
    load_schema,
)

logger = logging.getLogger(__name__)


# ===================================================================== #
#                        GENERATION HELPER                              #
# ===================================================================== #


def _generate_batch(
    model:               Any,
    tokenizer:           Any,
    encoder_inputs:      List[str],
    tokens:              AnyTokens,
    max_source_length:   int,
    max_target_length:   int,
    eval_beams:          int,
    device:              torch.device,
    constraint_decoding: bool = False,
) -> List[List[EntityBlock]]:
    """Tokenise, generate, decode, and parse a list of encoder inputs.

    Args:
        model:               The seq2seq model (unwrapped).
        tokenizer:           HuggingFace tokeniser.
        encoder_inputs:      Pre-built encoder input strings.
        tokens:              Token registry for SEL parsing.
        max_source_length:   Encoder truncation length.
        max_target_length:   Decoder max length.
        eval_beams:          Beam width.
        device:              Torch device.
        constraint_decoding: Whether to activate FSM constraints.

    Returns:
        List of parsed entity block lists, one per input string.
    """
    tok_out = tokenizer(
        encoder_inputs,
        max_length=max_source_length,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )
    input_ids      = tok_out["input_ids"].to(device)
    attention_mask = tok_out["attention_mask"].to(device)

    gen_kwargs: Dict[str, Any] = {
        "input_ids":            input_ids,
        "attention_mask":       attention_mask,
        "num_beams":            eval_beams,
        "max_length":           max_target_length,
        "length_penalty":       0.0,
        "no_repeat_ngram_size": 0,
        "early_stopping":       False,
    }

    if constraint_decoding:
        gen_kwargs["logits_processor"] = [
            build_constraint_processor(
                tokenizer=tokenizer,
                source_ids=input_ids,
                tokens=tokens,
                num_beams=eval_beams,
            )
        ]

    with torch.no_grad():
        generated = model.generate(**gen_kwargs)

    all_entities: List[List[EntityBlock]] = []
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
    for text in decoded:
        for tok in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token):
            if tok:
                text = text.replace(tok, "")
        text = " ".join(text.split())
        ents, _ = parse_sel(text, tok=tokens)
        all_entities.append(ents)
    return all_entities


def _to_spans(
    source_tokens: List[str],
    entities:      List[EntityBlock],
) -> List[Tuple[int, int]]:
    """Return ALL token-index span occurrences for each predicted entity."""
    spans: List[Tuple[int, int]] = []
    seen: set = set()
    for e in entities:
        for span in find_all_token_spans(source_tokens, e["text"]):
            if span not in seen:
                seen.add(span)
                spans.append(span)
    return spans


def _to_entity_data(
    source_tokens: List[str],
    entities:      List[EntityBlock],
) -> List[Tuple[int, int, str]]:
    """Return ALL ``(start, end, type)`` occurrences for each typed entity."""
    data: List[Tuple[int, int, str]] = []
    seen: set = set()
    for e in entities:
        if not e.get("type"):
            continue
        for span in find_all_token_spans(source_tokens, e["text"]):
            if span not in seen:
                seen.add(span)
                data.append((span[0], span[1], e["type"]))
    return data


# ===================================================================== #
#                    PIPELINE EVALUATION                                #
# ===================================================================== #


def _evaluate_pipeline(
    model:               Any,
    tokenizer:           Any,
    instances:           List[Dict],
    entity_schema:       List[str],
    rel_schema:          List[str],
    tokens:              AnyTokens,
    max_source_length:   int,
    max_target_length:   int,
    batch_size:          int,
    eval_beams:          int,
    device:              torch.device,
    constraint_decoding: bool,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run Boundary → NER → RE and return (per_instance_results, metrics)."""

    def _run(inputs):
        all_ents = []
        for i in tqdm(range(0, len(inputs), batch_size), leave=False):
            all_ents.extend(
                _generate_batch(
                    model, tokenizer, inputs[i:i+batch_size], tokens,
                    max_source_length, max_target_length,
                    eval_beams, device, constraint_decoding,
                )
            )
        return all_ents

    # Boundary
    b_inputs   = [build_boundary_encoder_input(inst["text"], tok=tokens) for inst in instances]
    b_per_inst = _run(b_inputs)

    # NER
    n_inputs = [
        build_ner_encoder_input(
            entity_schema,
            inst["tokens"],
            _to_spans(inst["tokens"], b_ents),
            random_order=False, tok=tokens,
        )
        for inst, b_ents in zip(instances, b_per_inst)
    ]
    n_per_inst = _run(n_inputs)

    # RE
    ner_type_maps = [{e["text"]: e.get("type", "") for e in n} for n in n_per_inst]
    r_inputs = [
        build_re_encoder_input(
            rel_schema,
            inst["tokens"],
            _to_entity_data(inst["tokens"], n_ents),
            random_order=False, tok=tokens,
        )
        for inst, n_ents in zip(instances, n_per_inst)
    ]
    r_per_inst = _run(r_inputs)

    # Per-instance results
    per_inst = []
    for i, inst in enumerate(instances):
        b_ents  = b_per_inst[i]
        n_ents  = n_per_inst[i]
        r_ents  = r_per_inst[i]
        nm      = ner_type_maps[i]
        triplets = extract_triplets(r_ents)
        per_inst.append({
            "text":           inst["text"],
            "boundary_spans": [e["text"] for e in b_ents],
            "ner_entities":   [{"text": e["text"], "type": e.get("type")} for e in n_ents],
            "re_triplets":    [{"head": t[0], "type": t[1], "tail": t[2]} for t in triplets],
            "gold_triplets":  [
                {"head": r["head"]["text"], "type": r["type"], "tail": r["tail"]["text"]}
                for r in inst["relations"]
            ],
        })

    # Metrics
    pred_triplets   = [extract_triplets(r) for r in r_per_inst]
    pred_quintuples = [
        [(e["text"], ner_type_maps[i].get(e["text"], ""), rel["type"],
          rel["tail"], ner_type_maps[i].get(rel["tail"], ""))
         for e in r_per_inst[i] for rel in e["relations"]]
        for i in range(len(instances))
    ]
    gold_triplets   = [
        [(r["head"]["text"], r["type"], r["tail"]["text"]) for r in inst["relations"]]
        for inst in instances
    ]
    gold_quintuples = [
        [(r["head"]["text"], r["head"].get("type",""), r["type"],
          r["tail"]["text"], r["tail"].get("type","")) for r in inst["relations"]]
        for inst in instances
    ]

    m: Dict[str, float] = {}
    m.update({f"boundary_{k}": v for k, v in compute_metrics_for_task(
        "boundary",
        all_pred_entities=[[e["text"] for e in b] for b in b_per_inst],
        all_gold_entities=[[e["text"] for e in inst["entities"]] for inst in instances],
    ).items()})
    m.update({f"ner_{k}": v for k, v in compute_metrics_for_task(
        "ner",
        all_pred_entities=[[e["text"] for e in n] for n in n_per_inst],
        all_gold_entities=[[e["text"] for e in inst["entities"]] for inst in instances],
        all_pred_entity_mentions=[
            [(e["text"], e.get("type") or "") for e in n if e.get("type")]
            for n in n_per_inst
        ],
        all_gold_entity_mentions=[
            [(e["text"], e.get("type","")) for e in inst["entities"]]
            for inst in instances
        ],
    ).items()})
    m.update({f"re_{k}": v for k, v in compute_metrics_for_task(
        "re",
        all_pred_triplets=pred_triplets,
        all_gold_triplets=gold_triplets,
        all_pred_quintuples=pred_quintuples,
        all_gold_quintuples=gold_quintuples,
    ).items()})

    return per_inst, m


# ===================================================================== #
#                      JOINT EVALUATION                                 #
# ===================================================================== #


def _evaluate_joint(
    model:               Any,
    tokenizer:           Any,
    instances:           List[Dict],
    entity_schema:       List[str],
    rel_schema:          List[str],
    tokens:              AnyTokens,
    max_source_length:   int,
    max_target_length:   int,
    batch_size:          int,
    eval_beams:          int,
    device:              torch.device,
    constraint_decoding: bool,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Run Joint + Joint+ and return (per_instance_results, metrics)."""

    def _run(inputs):
        all_ents = []
        for i in tqdm(range(0, len(inputs), batch_size), leave=False):
            all_ents.extend(
                _generate_batch(
                    model, tokenizer, inputs[i:i+batch_size], tokens,
                    max_source_length, max_target_length,
                    eval_beams, device, constraint_decoding,
                )
            )
        return all_ents

    j_inputs    = [build_joint_encoder_input(rel_schema, inst["text"], random_order=False, tok=tokens) for inst in instances]
    j_per_inst  = _run(j_inputs)
    jp_inputs   = [build_joint_plus_encoder_input(entity_schema, rel_schema, inst["text"], random_order=False, tok=tokens) for inst in instances]
    jp_per_inst = _run(jp_inputs)

    gold_triplets = [
        [(r["head"]["text"], r["type"], r["tail"]["text"]) for r in inst["relations"]]
        for inst in instances
    ]
    gold_quintuples = [
        [(r["head"]["text"], r["head"].get("type",""), r["type"],
          r["tail"]["text"], r["tail"].get("type","")) for r in inst["relations"]]
        for inst in instances
    ]

    per_inst = []
    for i, inst in enumerate(instances):
        per_inst.append({
            "text":           inst["text"],
            "joint_triplets": [{"head": t[0], "type": t[1], "tail": t[2]}
                               for t in extract_triplets(j_per_inst[i])],
            "joint_plus":     {
                "entities": [{"text": e["text"], "type": e.get("type")} for e in jp_per_inst[i]],
                "triplets": [{"head": t[0], "type": t[1], "tail": t[2]}
                             for t in extract_triplets(jp_per_inst[i])],
            },
            "gold_triplets": [
                {"head": r["head"]["text"], "type": r["type"], "tail": r["tail"]["text"]}
                for r in inst["relations"]
            ],
        })

    jp_type_maps = [{e["text"]: e.get("type", "") for e in jp} for jp in jp_per_inst]
    jp_triplets  = [extract_triplets(jp) for jp in jp_per_inst]
    jp_quintuples = [
        [(e["text"], jp_type_maps[i].get(e["text"], ""), rel["type"],
          rel["tail"], jp_type_maps[i].get(rel["tail"], ""))
         for e in jp_per_inst[i] for rel in e["relations"]]
        for i in range(len(instances))
    ]

    m: Dict[str, float] = {}
    m.update({f"joint_{k}": v for k, v in compute_metrics_for_task(
        "joint",
        all_pred_triplets=[extract_triplets(j) for j in j_per_inst],
        all_gold_triplets=gold_triplets,
    ).items()})
    m.update({f"joint_plus_{k}": v for k, v in compute_metrics_for_task(
        "joint+",
        all_pred_triplets=jp_triplets,
        all_gold_triplets=gold_triplets,
        all_pred_quintuples=jp_quintuples,
        all_gold_quintuples=gold_quintuples,
        all_pred_entities=[[e["text"] for e in jp] for jp in jp_per_inst],
        all_gold_entities=[[e["text"] for e in inst["entities"]] for inst in instances],
        all_pred_entity_mentions=[
            [(e["text"], e.get("type") or "") for e in jp if e.get("type")]
            for jp in jp_per_inst
        ],
        all_gold_entity_mentions=[
            [(e["text"], e.get("type","")) for e in inst["entities"]]
            for inst in instances
        ],
    ).items()})

    return per_inst, m


# ===================================================================== #
#                              MAIN                                     #
# ===================================================================== #


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config()
    logger.info("Configuration loaded: %s", cfg.config_path)

    if cfg.hardware.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in cfg.hardware.gpu_ids)

    set_seed(cfg.train.seed)

    # ---- Load model ----
    ckpt = cfg.model.pretrained_checkpoint
    if ckpt is None:
        raise ValueError("model.pretrained_checkpoint is required for evaluation.")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model     = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

    # Detect variant from saved metadata; fall back to config.
    variant_file = Path(ckpt) / "model_variant.txt"
    model_variant = (
        variant_file.read_text(encoding="utf-8").strip()
        if variant_file.exists()
        else cfg.model.model_variant
    )
    tokens = PIPELINE_TOKENS if model_variant == "pipeline" else JOINT_TOKENS
    add_special_tokens_to_tokenizer(tokenizer, tokens, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()
    logger.info("Loaded %s model from %s on %s", model_variant, ckpt, device)

    # ---- Data and schema ----
    rel_schema    = load_schema(cfg.data.schema_file)
    entity_schema = load_entity_schema(cfg.data.entity_schema_file)

    split      = cfg.evaluation.split
    split_file = {"val": "val.jsonl", "test": "test.jsonl"}[split]
    dataset    = S2GDataset(Path(cfg.data.data_dir) / split_file, seed=cfg.train.seed)
    instances  = [dataset[i] for i in range(len(dataset))]
    logger.info("%s set: %d instances", split, len(instances))

    # ---- Evaluate ----
    eval_fn     = _evaluate_pipeline if model_variant == "pipeline" else _evaluate_joint
    kwargs = dict(
        model=model, tokenizer=tokenizer, instances=instances,
        entity_schema=entity_schema, rel_schema=rel_schema, tokens=tokens,
        max_source_length=cfg.tokenization.max_source_length,
        max_target_length=cfg.tokenization.max_target_length,
        batch_size=cfg.validation.batch_size,
        eval_beams=cfg.generation.num_beams,
        device=device,
        constraint_decoding=cfg.generation.constraint_decoding,
    )
    per_inst_results, metrics = eval_fn(**kwargs)

    # ---- Write outputs ----
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{split}_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in per_inst_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics_path = output_dir / f"{split}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("%s metrics:\n%s", split, json.dumps(metrics, indent=2))
    logger.info("Output written to %s", output_dir)


if __name__ == "__main__":
    main()