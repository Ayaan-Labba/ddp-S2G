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
    S2GTokens, add_special_tokens_to_tokenizer,
    build_boundary_joint_encoder_input, build_joint_encoder_input,
    build_re_encoder_input, build_boundary_re_encoder_input,
    extract_triplets, find_all_token_spans, parse_sel,
    VARIANT_TO_TASKS,
)
from s2g.model import build_constraint_processor
from s2g.scripts.config_utils import load_entity_schema, load_schema

logger = logging.getLogger(__name__)
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


def _generate_single(model, tokenizer, encoder_input, tokens, num_beams, max_src, max_tgt, device, constraint_decoding=False, entity_schema=None, rel_schema=None) -> str:
    tok_out = tokenizer([encoder_input], max_length=max_src, truncation=True, return_tensors="pt").to(device, non_blocking=True)
    gen_kwargs = {**tok_out, "num_beams": num_beams, "max_length": max_tgt, "length_penalty": 0.0, "no_repeat_ngram_size": 0, "early_stopping": False}
    
    if constraint_decoding: 
        gen_kwargs["logits_processor"] = [build_constraint_processor(tokenizer, tok_out["input_ids"], tokens, num_beams, entity_schema=entity_schema, rel_schema=rel_schema)]

    dtype = next(model.parameters()).dtype
    ctx = torch.autocast(device.type, dtype) if dtype in {torch.bfloat16, torch.float16} and device.type == "cuda" else contextlib.nullcontext()
    
    with torch.inference_mode(), ctx: 
        generated = model.generate(**gen_kwargs)
    
    raw = tokenizer.decode(generated[0], skip_special_tokens=False)
    for tok in filter(None, [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token]): 
        raw = raw.replace(tok, "")
    return " ".join(raw.split())


def extract_re(text: str, model: Any, tokenizer: Any, entity_schema: List[str], rel_schema: List[str], tokens: Any, num_beams: int=4, max_source_length: int=300, max_target_length: int=200, device: Optional[Any]=None, constraint_decoding: bool=False, tasks=None, ssi_prompt: str="ssi") -> Dict[str, Any]:
    _ensure_nltk_punkt()
    device = device or next(model.parameters()).device
    src_toks = nltk.word_tokenize(text)

    if tasks is None:
        tasks = ["re"]
    use_re = "re" in tasks
    use_boundary_re = "boundary_re" in tasks

    r_ents = []
    if use_re or use_boundary_re:
        if tokens.variant == "re":
            re_input = build_re_encoder_input(entity_schema, rel_schema, text, False, tokens, ssi_prompt=ssi_prompt)
        elif tokens.variant == "boundary_re":
            re_input = build_boundary_re_encoder_input(rel_schema, text, False, tokens, ssi_prompt=ssi_prompt)
        r_ents, _ = parse_sel(_generate_single(model, tokenizer, re_input, tokens, num_beams, max_source_length, max_target_length, device, constraint_decoding, entity_schema=entity_schema, rel_schema=rel_schema), tok=tokens)

    res = {"text": text, "tokens": src_toks}
    if use_re or use_boundary_re:
        res["re_triplets"] = [{"head": t[0], "type": t[1], "tail": t[2]} for t in extract_triplets(r_ents)]
    return res


def extract_boundary_joint(text: str, model: Any, tokenizer: Any, entity_schema: List[str], rel_schema: List[str], tokens: Any, num_beams: int=4, max_source_length: int=300, max_target_length: int=200, device: Optional[Any]=None, constraint_decoding: bool=False, tasks=None, ssi_prompt: str="ssi") -> Dict[str, Any]:
    device = device or next(model.parameters()).device
    if tasks is None:
        tasks = ["boundary_joint", "joint"]
    use_boundary_joint = "boundary_joint" in tasks
    use_joint = "joint" in tasks

    res = {"text": text}
    if use_boundary_joint:
        j_ents, _ = parse_sel(_generate_single(model, tokenizer, build_boundary_joint_encoder_input(rel_schema, text, False, tokens, ssi_prompt=ssi_prompt), tokens, num_beams, max_source_length, max_target_length, device, constraint_decoding, entity_schema=entity_schema, rel_schema=rel_schema), tok=tokens)
        res["boundary_joint_triplets"] = [{"head": t[0], "type": t[1], "tail": t[2]} for t in extract_triplets(j_ents)]
    if use_joint:
        jp_ents, _ = parse_sel(_generate_single(model, tokenizer, build_joint_encoder_input(entity_schema, rel_schema, text, False, tokens, ssi_prompt=ssi_prompt), tokens, num_beams, max_source_length, max_target_length, device, constraint_decoding, entity_schema=entity_schema, rel_schema=rel_schema), tok=tokens)
        res["joint"] = {"entities": [{"text": e["text"], "type": e.get("type")} for e in jp_ents], "triplets": [{"head": t[0], "type": t[1], "tail": t[2]} for t in extract_triplets(jp_ents)]}
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--schema_file", required=True); parser.add_argument("--entity_schema_file", default=None); parser.add_argument("--input_file", default=None); parser.add_argument("--output_file", default=None); parser.add_argument("--constraint_decoding", default="false"); parser.add_argument("--num_beams", type=int, default=4); parser.add_argument("--max_source_length", type=int, default=300); parser.add_argument("--max_target_length", type=int, default=200); parser.add_argument("--ssi_prompt", default="ssi", choices=["ssi", "natural", "false"])
    args = parser.parse_args()

    tokenizer, model = AutoTokenizer.from_pretrained(args.checkpoint), AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, torch_dtype="auto")
    if hasattr(model.generation_config, "forced_bos_token_id"):
        model.generation_config.forced_bos_token_id = None
    variant_file = Path(args.checkpoint) / "model_variant.txt"
    if not variant_file.exists():
        raise FileNotFoundError(
            f"model_variant.txt not found in checkpoint '{args.checkpoint}'. "
            f"Cannot determine the model variant for inference."
        )
    model_variant = variant_file.read_text(encoding="utf-8").strip()
    
    if (Path(args.checkpoint) / "tasks.json").exists():
        with open(Path(args.checkpoint) / "tasks.json", "r", encoding="utf-8") as f:
            tasks = json.load(f)
    elif (Path(args.checkpoint) / "tasks.txt").exists():
        tasks = [t.strip() for t in (Path(args.checkpoint) / "tasks.txt").read_text(encoding="utf-8").strip().split(",") if t.strip()]
    else:
        tasks = VARIANT_TO_TASKS[model_variant]

    use_rejection = "<extra_id_6>" in tokenizer.get_vocab()
    tokens = S2GTokens(model_variant, use_rejection=use_rejection)
    add_special_tokens_to_tokenizer(tokenizer, tokens, model, warm=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    kwargs = {"model": model, "tokenizer": tokenizer, "entity_schema": load_entity_schema(args.entity_schema_file), "rel_schema": load_schema(args.schema_file), "tokens": tokens, "num_beams": args.num_beams, "max_source_length": args.max_source_length, "max_target_length": args.max_target_length, "device": device, "constraint_decoding": args.constraint_decoding.lower() in ("true", "1", "yes"), "tasks": tasks, "ssi_prompt": args.ssi_prompt}
    extract_fn = extract_re if model_variant in {"re", "boundary_re"} else extract_boundary_joint

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