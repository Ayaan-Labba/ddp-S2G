"""
Inference script for interactive or batch S2G knowledge graph extraction.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from s2g.linearisation import (
    JOINT_TOKENS, PIPELINE_TOKENS, add_special_tokens_to_tokenizer,
    build_boundary_encoder_input, build_joint_encoder_input, build_joint_plus_encoder_input,
    build_ner_encoder_input, build_re_encoder_input, extract_triplets, find_all_token_spans, parse_sel,
)
from s2g.model import build_constraint_processor
from s2g.scripts.config_utils import load_entity_schema, load_schema

logger = logging.getLogger(__name__)

# EFFICIENCY FIX: Cache NLTK checks globally to eliminate filesystem polling on every call
_NLTK_READY = False

def _ensure_nltk_punkt() -> None:
    global _NLTK_READY
    if _NLTK_READY: 
        return
        
    try: 
        nltk.data.find('tokenizers/punkt')
    except LookupError: 
        nltk.download('punkt', quiet=True)
    try: 
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError: 
        nltk.download('punkt_tab', quiet=True)
        
    _NLTK_READY = True


def _generate_single(model, tokenizer, encoder_input, tokens, num_beams, max_src, max_tgt, device, constraint_decoding=False) -> str:
    # EFFICIENCY FIX: non_blocking=True for pipeline batching
    tok_out = tokenizer([encoder_input], max_length=max_src, truncation=True, return_tensors="pt").to(device, non_blocking=True)
    gen_kwargs = {**tok_out, "num_beams": num_beams, "max_length": max_tgt, "length_penalty": 0.0, "no_repeat_ngram_size": 0, "early_stopping": False}
    
    if constraint_decoding: 
        gen_kwargs["logits_processor"] = [build_constraint_processor(tokenizer, tok_out["input_ids"], tokens, num_beams)]

    dtype = next(model.parameters()).dtype
    ctx = torch.autocast(device.type, dtype) if dtype in {torch.bfloat16, torch.float16} and device.type == "cuda" else contextlib.nullcontext()
    
    with torch.inference_mode(), ctx: 
        generated = model.generate(**gen_kwargs)
    
    raw = tokenizer.decode(generated[0], skip_special_tokens=False)
    for tok in filter(None, [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token]): 
        raw = raw.replace(tok, "")
    return " ".join(raw.split())


def extract_pipeline(text: str, model: Any, tokenizer: Any, entity_schema: List[str], rel_schema: List[str], tokens: Any, num_beams: int=4, max_source_length: int=300, max_target_length: int=200, device: Optional[Any]=None, constraint_decoding: bool=False) -> Dict[str, Any]:
    _ensure_nltk_punkt()
    device = device or next(model.parameters()).device
    src_toks = nltk.word_tokenize(text)

    b_ents, _ = parse_sel(_generate_single(model, tokenizer, build_boundary_encoder_input(text, tok=tokens), tokens, num_beams, max_source_length, max_target_length, device, constraint_decoding), tok=tokens)
    
    n_ents, _ = parse_sel(_generate_single(model, tokenizer, build_ner_encoder_input(entity_schema, src_toks, list(dict.fromkeys(s for e in b_ents for s in find_all_token_spans(src_toks, e["text"]))), False, tokens), tokens, num_beams, max_source_length, max_target_length, device, constraint_decoding), tok=tokens)
    
    r_ents, _ = parse_sel(_generate_single(model, tokenizer, build_re_encoder_input(rel_schema, src_toks, list(dict.fromkeys((*s, e["type"]) for e in n_ents if e.get("type") for s in find_all_token_spans(src_toks, e["text"]))), False, tokens), tokens, num_beams, max_source_length, max_target_length, device, constraint_decoding), tok=tokens)

    return {"text": text, "tokens": src_toks, "boundary_spans": [e["text"] for e in b_ents], "ner_entities": [{"text": e["text"], "type": e.get("type")} for e in n_ents], "re_triplets": [{"head": t[0], "type": t[1], "tail": t[2]} for t in extract_triplets(r_ents)]}


def extract_joint(text: str, model: Any, tokenizer: Any, entity_schema: List[str], rel_schema: List[str], tokens: Any, num_beams: int=4, max_source_length: int=300, max_target_length: int=200, device: Optional[Any]=None, constraint_decoding: bool=False) -> Dict[str, Any]:
    device = device or next(model.parameters()).device
    j_ents, _ = parse_sel(_generate_single(model, tokenizer, build_joint_encoder_input(rel_schema, text, False, tokens), tokens, num_beams, max_source_length, max_target_length, device, constraint_decoding), tok=tokens)
    jp_ents, _ = parse_sel(_generate_single(model, tokenizer, build_joint_plus_encoder_input(entity_schema, rel_schema, text, False, tokens), tokens, num_beams, max_source_length, max_target_length, device, constraint_decoding), tok=tokens)

    return {"text": text, "joint_triplets": [{"head": t[0], "type": t[1], "tail": t[2]} for t in extract_triplets(j_ents)], "joint_plus": {"entities": [{"text": e["text"], "type": e.get("type")} for e in jp_ents], "triplets": [{"head": t[0], "type": t[1], "tail": t[2]} for t in extract_triplets(jp_ents)]}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--schema_file", required=True); parser.add_argument("--entity_schema_file", default=None); parser.add_argument("--input_file", default=None); parser.add_argument("--output_file", default=None); parser.add_argument("--constraint_decoding", default="false"); parser.add_argument("--num_beams", type=int, default=4); parser.add_argument("--max_source_length", type=int, default=300); parser.add_argument("--max_target_length", type=int, default=200)
    args = parser.parse_args()

    tokenizer, model = AutoTokenizer.from_pretrained(args.checkpoint), AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    model_variant = (Path(args.checkpoint) / "model_variant.txt").read_text(encoding="utf-8").strip() if (Path(args.checkpoint) / "model_variant.txt").exists() else "pipeline"
    tokens = PIPELINE_TOKENS if model_variant == "pipeline" else JOINT_TOKENS
    add_special_tokens_to_tokenizer(tokenizer, tokens, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    kwargs = {"model": model, "tokenizer": tokenizer, "entity_schema": load_entity_schema(args.entity_schema_file), "rel_schema": load_schema(args.schema_file), "tokens": tokens, "num_beams": args.num_beams, "max_source_length": args.max_source_length, "max_target_length": args.max_target_length, "device": device, "constraint_decoding": args.constraint_decoding.lower() in ("true", "1", "yes")}
    extract_fn = extract_pipeline if model_variant == "pipeline" else extract_joint

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f: sentences = [ln.strip() for ln in f if ln.strip()]
        out = open(args.output_file, "w", encoding="utf-8") if args.output_file else sys.stdout
        try:
            for sent in sentences: out.write(json.dumps(extract_fn(text=sent, **kwargs), ensure_ascii=False) + "\n")
        finally:
            if args.output_file: out.close()
    else:
        print(f"\n=== S2G Interactive Inference ({model_variant}) ===")
        while (text := input(">>> ").strip()) and text.lower() not in ("quit", "exit", "q"):
            print(json.dumps(extract_fn(text=text, **kwargs), indent=2))

if __name__ == "__main__":
    main()