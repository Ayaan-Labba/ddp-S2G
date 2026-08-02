"""
Decoupled S2GEvaluator class for single-pass SEL parsing, metric evaluation,
and streaming evaluation loops.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from s2g.evaluation import compute_metrics_for_variant
from s2g.model import build_constraint_processor
from s2g.linearisation import EntityBlock, S2GTokens, parse_graph, organise_filter_and_block


logger = logging.getLogger(__name__)


class S2GEvaluator:
    """
    Evaluator for S2G models. Handles text cleaning, single-pass graph parsing,
    per-instance formatting, metric calculations, and streaming DataLoader evaluation.
    """

    def __init__(
        self,
        tokenizer: Any,
        tokens: S2GTokens,
        model_variant: str,
        rel_schema: List[str],
        entity_schema: List[str],
    ) -> None:
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.model_variant = model_variant
        self.rel_schema = rel_schema
        self.entity_schema = entity_schema

        self.specials = [
            tok for tok in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token) if tok
        ]

    def clean_text(self, text: str) -> str:
        """Strips special control tokens and normalizes whitespace."""
        for tok in self.specials:
            text = text.replace(tok, "")
        return " ".join(text.split())

    def parse_text(self, text: str) -> Tuple[List[EntityBlock], List[Any]]:
        """Cleans raw text and parses graph entity blocks."""
        cleaned = self.clean_text(text)
        return parse_graph(cleaned, tok=self.tokens)

    def process_batch_predictions(
        self, batch_instances: List[Dict[str, Any]], generated_ids: torch.Tensor
    ) -> Tuple[List[List[EntityBlock]], List[Dict[str, Any]]]:
        """
        Decodes generated token IDs once, parses SEL blocks, and formats per-instance records.
        """
        preds = generated_ids.cpu().numpy()
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)

        batch_pred_blocks = []
        batch_records = []

        for inst, raw_pred in zip(batch_instances, decoded_preds):
            parsed_ents, rejected = self.parse_text(raw_pred)
            batch_pred_blocks.append(parsed_ents)
            batch_records.append(
                {
                    "text": inst.get("text", ""),
                    "prediction_raw": self.clean_text(raw_pred),
                    "parsed_blocks": parsed_ents,
                    "rejected": rejected,
                    "gold_entities": inst.get("entities", []),
                    "gold_relations": inst.get("relations", []),
                }
            )

        return batch_pred_blocks, batch_records

    def compute_final_metrics(
        self, all_pred_blocks: List[List[EntityBlock]], all_gold_blocks: List[List[EntityBlock]]
    ) -> Dict[str, float]:
        """Computes corpus-level macro PRF metrics for the configured variant."""
        return compute_metrics_for_variant(
            variant=self.model_variant,
            all_pred_blocks=all_pred_blocks,
            all_gold_blocks=all_gold_blocks,
            rel_schema=self.rel_schema,
            entity_schema=self.entity_schema,
        )

    def run_evaluation(
        self,
        model: Any,
        dataloader: DataLoader,
        dataset: Any,
        collator: Any,
        out_dir: Path,
        split: str,
        device: torch.device,
        max_target_length: int,
        num_beams: int,
        constraint_decoding: bool = False,
    ) -> Dict[str, float]:
        """
        Executes a memory-efficient, streaming evaluation over a DataLoader.
        Flushes prediction records directly to disk to maintain constant memory usage.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / f"{split}_results.jsonl"
        metrics_file = out_dir / f"{split}_metrics.json"

        all_pred_blocks: List[List[EntityBlock]] = []
        all_gold_blocks: List[List[EntityBlock]] = []

        batch_size = dataloader.batch_size or 1
        logger.info("Evaluating split '%s' using streaming DataLoader...", split)

        prep_fn = getattr(collator, f"_prepare_{self.model_variant}")

        with open(results_file, "w", encoding="utf-8") as f_out, torch.inference_mode():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating ({split})")):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(dataset))
                batch_instances = [dataset[i] for i in range(start_idx, end_idx)]

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                dtype = next(model.parameters()).dtype
                ctx = (
                    torch.autocast(device.type, dtype)
                    if dtype in {torch.bfloat16, torch.float16} and device.type == "cuda"
                    else contextlib.nullcontext()
                )

                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_length": max_target_length,
                    "num_beams": num_beams,
                }

                if constraint_decoding:
                    gen_kwargs["logits_processor"] = [
                            build_constraint_processor(
                                self.tokenizer,
                                input_ids,
                                self.tokens,
                                num_beams,
                                entity_schema=self.entity_schema,
                                rel_schema=self.rel_schema,
                            )
                        ]

                with ctx:
                    generated_ids = model.generate(**gen_kwargs)

                pred_blocks, batch_records = self.process_batch_predictions(
                    batch_instances, generated_ids
                )
                all_pred_blocks.extend(pred_blocks)

                ent_schema_set = set(self.entity_schema)
                rel_schema_set = set(self.rel_schema)

                for inst in batch_instances:
                    gold_ents = organise_filter_and_block(
                        inst["entities"], inst["relations"], ent_schema_set, rel_schema_set
                    )
                    all_gold_blocks.append(gold_ents)

                for record in batch_records:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        metrics = self.compute_final_metrics(all_pred_blocks, all_gold_blocks)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Saved evaluation metrics to %s", metrics_file)
        logger.info("Saved streaming predictions to %s", results_file)
        return metrics