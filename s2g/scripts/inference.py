"""
Inference script for the S2G model.

Provides a simple interface for extracting structured knowledge-graph
output from arbitrary text using a trained S2G checkpoint.  Supports
interactive mode and batch mode.

For the Pipeline model, the full Boundary → NER → RE chain is run on
each input sentence.  For the Joint model, Joint and Joint+ are run
independently.

Usage::

    # Interactive (Pipeline model)
    python -m s2g.scripts.inference \\
        --checkpoint outputs/finetune/conll04/best_model \\
        --schema_file data/conll04/relation.schema \\
        --entity_schema_file data/conll04/entity.schema

    # Batch mode (one sentence per line)
    python -m s2g.scripts.inference \\
        --checkpoint outputs/finetune/conll04/best_model \\
        --schema_file data/conll04/relation.schema \\
        --entity_schema_file data/conll04/entity.schema \\
        --input_file sentences.txt \\
        --output_file predictions.jsonl
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
    AnyTokens,
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
    parse_sel,
)
from s2g.model import build_constraint_processor
from s2g.scripts.config_utils import load_entity_schema, load_schema

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Word-tokenize *text* using NLTK."""
    return nltk.word_tokenize(text)


def _generate_single(
    model:               Any,
    tokenizer:           Any,
    encoder_input:       str,
    tokens:              AnyTokens,
    num_beams:           int,
    max_source_length:   int,
    max_target_length:   int,
    device:              torch.device,
    constraint_decoding: bool = False,
) -> str:
    """Generate and decode one SEL string from one encoder input."""
    tok_out = tokenizer(
        [encoder_input],
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = tok_out["input_ids"].to(device)
    attention_mask = tok_out["attention_mask"].to(device)

    gen_kwargs: Dict[str, Any] = {
        "input_ids":            input_ids,
        "attention_mask":       attention_mask,
        "num_beams":            num_beams,
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
                num_beams=num_beams,
            )
        ]

    param_dtype = next(model.parameters()).dtype
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=param_dtype)
        if param_dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        generated = model.generate(**gen_kwargs)

    raw = tokenizer.decode(generated[0], skip_special_tokens=False)
    for tok in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token):
        if tok:
            raw = raw.replace(tok, "")
    return " ".join(raw.split())


def extract_pipeline(
    text:                str,
    model:               Any,
    tokenizer:           Any,
    entity_schema:       List[str],
    rel_schema:          List[str],
    tokens:              AnyTokens,
    num_beams:           int          = 4,
    max_source_length:   int          = 300,
    max_target_length:   int          = 200,
    device:              Optional[Any] = None,
    constraint_decoding: bool          = False,
) -> Dict[str, Any]:
    """Extract entities and relations from *text* using the Pipeline model.

    Runs Boundary → NER → RE sequentially, using each stage's predicted
    output as the next stage's augmented encoder input.

    Returns:
        Dict with keys: ``text``, ``tokens``, ``boundary_spans``,
        ``ner_entities``, ``re_triplets``.
    """
    if device is None:
        device = next(model.parameters()).device

    src_tokens = _tokenize(text)

    # Step 1: Boundary
    b_sel  = _generate_single(model, tokenizer,
                               build_boundary_encoder_input(text, tok=tokens),
                               tokens, num_beams, max_source_length,
                               max_target_length, device, constraint_decoding)
    b_ents, _ = parse_sel(b_sel, tok=tokens)

    # Step 2: NER — mark ALL occurrences of each predicted entity span
    spans: List[Tuple[int, int]] = []
    seen_spans: set = set()
    for e in b_ents:
        for s in find_all_token_spans(src_tokens, e["text"]):
            if s not in seen_spans:
                seen_spans.add(s)
                spans.append(s)
    n_sel   = _generate_single(model, tokenizer,
                                build_ner_encoder_input(entity_schema, src_tokens, spans,
                                                        random_order=False, tok=tokens),
                                tokens, num_beams, max_source_length,
                                max_target_length, device, constraint_decoding)
    n_ents, _ = parse_sel(n_sel, tok=tokens)

    # Step 3: RE — mark ALL occurrences of each predicted NER entity
    entity_data: List[Tuple[int, int, str]] = []
    seen_re: set = set()
    for e in n_ents:
        if not e.get("type"):
            continue
        for s in find_all_token_spans(src_tokens, e["text"]):
            if s not in seen_re:
                seen_re.add(s)
                entity_data.append((s[0], s[1], e["type"]))
    r_sel  = _generate_single(model, tokenizer,
                               build_re_encoder_input(rel_schema, src_tokens, entity_data,
                                                      random_order=False, tok=tokens),
                               tokens, num_beams, max_source_length,
                               max_target_length, device, constraint_decoding)
    r_ents, _ = parse_sel(r_sel, tok=tokens)
    triplets   = extract_triplets(r_ents)

    return {
        "text":           text,
        "tokens":         src_tokens,
        "boundary_spans": [e["text"] for e in b_ents],
        "ner_entities":   [{"text": e["text"], "type": e.get("type")} for e in n_ents],
        "re_triplets":    [{"head": t[0], "type": t[1], "tail": t[2]} for t in triplets],
    }


def extract_joint(
    text:                str,
    model:               Any,
    tokenizer:           Any,
    entity_schema:       List[str],
    rel_schema:          List[str],
    tokens:              AnyTokens,
    num_beams:           int          = 4,
    max_source_length:   int          = 300,
    max_target_length:   int          = 200,
    device:              Optional[Any] = None,
    constraint_decoding: bool          = False,
) -> Dict[str, Any]:
    """Extract entities and relations from *text* using the Joint model.

    Runs Joint and Joint+ independently on raw text.

    Returns:
        Dict with keys: ``text``, ``joint_triplets``, ``joint_plus``.
    """
    if device is None:
        device = next(model.parameters()).device

    j_sel  = _generate_single(model, tokenizer,
                               build_joint_encoder_input(rel_schema, text, random_order=False, tok=tokens),
                               tokens, num_beams, max_source_length,
                               max_target_length, device, constraint_decoding)
    jp_sel = _generate_single(model, tokenizer,
                               build_joint_plus_encoder_input(entity_schema, rel_schema, text,
                                                              random_order=False, tok=tokens),
                               tokens, num_beams, max_source_length,
                               max_target_length, device, constraint_decoding)

    j_ents,  _  = parse_sel(j_sel,  tok=tokens)
    jp_ents, _  = parse_sel(jp_sel, tok=tokens)

    return {
        "text":          text,
        "joint_triplets": [{"head": t[0], "type": t[1], "tail": t[2]}
                           for t in extract_triplets(j_ents)],
        "joint_plus": {
            "entities": [{"text": e["text"], "type": e.get("type")} for e in jp_ents],
            "triplets": [{"head": t[0], "type": t[1], "tail": t[2]}
                         for t in extract_triplets(jp_ents)],
        },
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="S2G inference — extract entities and relations from text."
    )
    parser.add_argument("--checkpoint",          required=True)
    parser.add_argument("--schema_file",         required=True)
    parser.add_argument("--entity_schema_file",  default=None)
    parser.add_argument("--input_file",          default=None)
    parser.add_argument("--output_file",         default=None)
    parser.add_argument("--constraint_decoding", default="false")
    parser.add_argument("--num_beams",           type=int, default=4)
    parser.add_argument("--max_source_length",   type=int, default=300)
    parser.add_argument("--max_target_length",   type=int, default=200)
    args = parser.parse_args()

    logger.info("Loading model from %s", args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)

    variant_file   = Path(args.checkpoint) / "model_variant.txt"
    model_variant  = (
        variant_file.read_text(encoding="utf-8").strip()
        if variant_file.exists() else "pipeline"
    )
    tokens = PIPELINE_TOKENS if model_variant == "pipeline" else JOINT_TOKENS
    add_special_tokens_to_tokenizer(tokenizer, tokens, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()
    logger.info("Loaded %s model on %s", model_variant, device)

    rel_schema    = load_schema(args.schema_file)
    entity_schema = load_entity_schema(args.entity_schema_file)

    use_cd     = args.constraint_decoding.lower() in ("true", "1", "yes")
    extract_fn = extract_pipeline if model_variant == "pipeline" else extract_joint

    common = dict(
        model=model, tokenizer=tokenizer,
        entity_schema=entity_schema, rel_schema=rel_schema, tokens=tokens,
        num_beams=args.num_beams,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        device=device,
        constraint_decoding=use_cd,
    )

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            sentences = [ln.strip() for ln in f if ln.strip()]
        logger.info("Processing %d sentences...", len(sentences))
        out = open(args.output_file, "w", encoding="utf-8") if args.output_file else sys.stdout
        try:
            for sent in sentences:
                result = extract_fn(text=sent, **common)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
        finally:
            if args.output_file:
                out.close()
        logger.info("Done. Output: %s", args.output_file or "stdout")
    else:
        print(f"\n=== S2G Interactive Inference ({model_variant} model) ===")
        print(f"Schema: {len(rel_schema)} relation types, {len(entity_schema)} entity types")
        print('Type a sentence and press Enter. Type "quit" to exit.\n')

        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not text or text.lower() in ("quit", "exit", "q"):
                break

            result = extract_fn(text=text, **common)
            if model_variant == "pipeline":
                print(f"\n  Boundary spans: {result['boundary_spans']}")
                print(f"  NER entities:   {result['ner_entities']}")
                if result["re_triplets"]:
                    print(f"  Triplets:")
                    for t in result["re_triplets"]:
                        print(f"    ({t['head']}, {t['type']}, {t['tail']})")
                else:
                    print("  No triplets extracted.")
            else:
                if result["joint_triplets"]:
                    print(f"  Joint triplets:")
                    for t in result["joint_triplets"]:
                        print(f"    ({t['head']}, {t['type']}, {t['tail']})")
                jp = result["joint_plus"]
                if jp["entities"]:
                    print(f"  Joint+ entities: {jp['entities']}")
            print()


if __name__ == "__main__":
    main()