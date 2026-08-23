"""
Decoupled S2GEvaluator class for single-pass SEL parsing, metric evaluation,
and streaming evaluation loops.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from s2g.evaluation.metrics import compute_metrics_for_variant
from s2g.evaluation.gold import build_gold_blocks, build_gold_offsets
from s2g.evaluation.offsets import project_blocks
from s2g.model import build_constraint_processor
from s2g.linearisation import EntityBlock, S2GTokens, parse_graph


logger = logging.getLogger(__name__)


class S2GEvaluator:
    """
    Evaluator for S2G models. 
    Handles text cleaning, single-pass graph parsing, per-instance formatting, metric calculations, and streaming 
    DataLoader evaluation.
    """
    def __init__(
        self,
        tokenizer: Any,
        tokens: S2GTokens,
        variant: str,
        rel_schema: List[str],
        ent_schema: List[str],
        dedup: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.variant = variant
        self.rel_schema = rel_schema
        self.ent_schema = ent_schema
        self.dedup = dedup
        self.special_toks = [tok for tok in (tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token) if tok]

    def clean_text(self, text: str) -> str:
        """Strips special control tokens and normalizes whitespace."""
        for tok in self.special_toks:
            text = text.replace(tok, "")

        return " ".join(text.split())

    def parse_text(self, text: str) -> Tuple[List[EntityBlock], List[Any]]:
        """Cleans raw text and parses graph entity blocks."""
        cleaned = self.clean_text(text)
        return parse_graph(cleaned, tok=self.tokens)

    def build_gold(self, instance: Dict[str, Any]) -> Tuple[List[EntityBlock], Tuple]:
        """Gold blocks and gold offset tuples for one preprocessed instance."""
        return (
            build_gold_blocks(instance, self.variant, self.dedup),
            build_gold_offsets(instance, self.variant),
        )

    def process_batch_outputs(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        instances: List[Dict[str, Any]],
    ) -> Tuple[List[List[EntityBlock]], List[List[EntityBlock]], List[Tuple], List[Tuple], List[Dict[str, Any]]]:
        """
        Decodes generated IDs, parses predicted graph blocks and projects them onto
        source offsets, and pairs them with gold taken from the preprocessed
        instances.  The decoded labels are retained only as a debugging record.
        """
        preds = generated_ids.cpu().numpy()
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)

        lbls = labels.cpu().numpy()
        lbls = np.where(lbls != -100, lbls, self.tokenizer.pad_token_id)
        decoded_golds = self.tokenizer.batch_decode(lbls, skip_special_tokens=False)

        inps = input_ids.cpu().numpy()
        inps = np.where(inps != -100, inps, self.tokenizer.pad_token_id)
        decoded_inputs = self.tokenizer.batch_decode(inps, skip_special_tokens=True)

        batch_pred_blocks = []
        batch_gold_blocks = []
        batch_pred_offsets = []
        batch_gold_offsets = []
        batch_records = []

        for raw_inp, raw_pred, raw_gold, inst in zip(decoded_inputs, decoded_preds, decoded_golds, instances):
            pred_ents, rejected = self.parse_text(raw_pred)
            gold_ents, gold_offsets = self.build_gold(inst)
            pred_offsets, offset_map = project_blocks(pred_ents, inst.get('tokens', []))

            batch_pred_blocks.append(pred_ents)
            batch_gold_blocks.append(gold_ents)
            batch_pred_offsets.append(pred_offsets)
            batch_gold_offsets.append(gold_offsets)

            batch_records.append(
                {
                    'text': inst.get('text', self.clean_text(raw_inp)),
                    'encoder_input': self.clean_text(raw_inp),
                    'prediction_raw': self.clean_text(raw_pred),
                    'gold_raw': self.clean_text(raw_gold),
                    'parsed_pred_blocks': pred_ents,
                    'gold_blocks': gold_ents,
                    # Negative offsets are hallucinated mentions: text the model
                    # produced that occurs nowhere in the source.
                    'pred_offsets': offset_map,
                    'rejected': rejected,
                }
            )

        return batch_pred_blocks, batch_gold_blocks, batch_pred_offsets, batch_gold_offsets, batch_records

    def compute_final_metrics(
        self, 
        all_pred_blocks: List[List[EntityBlock]], 
        all_gold_blocks: List[List[EntityBlock]],
        all_pred_offsets: Optional[List[Tuple]] = None,
        all_gold_offsets: Optional[List[Tuple]] = None,
    ) -> Dict[str, float]:
        """Computes corpus-level micro and macro PRF metrics for the configured variant."""
        return compute_metrics_for_variant(
            variant=self.variant,
            all_pred_blocks=all_pred_blocks,
            all_gold_blocks=all_gold_blocks,
            rel_schema=self.rel_schema,
            ent_schema=self.ent_schema,
            all_pred_offsets=all_pred_offsets,
            all_gold_offsets=all_gold_offsets,
        )

    def run_evaluation(
        self,
        dataset: Any,
        split: str,
        dataloader: DataLoader,
        out_dir: Path,
        model: Any,
        max_target_length: int,
        num_beams: int,
        device: torch.device,
        constraint_decoding: bool = False,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        no_repeat_ngram_size: Optional[int] = None
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
        all_pred_offsets: List[Tuple] = []
        all_gold_offsets: List[Tuple] = []

        logger.info("Evaluating split '%s' using streaming DataLoader...", split)

        # Gold comes from ``dataset`` by position. Both callers build the loader
        # with shuffle=False and no drop_last, so batch n is a contiguous slice.
        cursor = 0

        with open(results_file, 'w', encoding='utf-8') as f_out, torch.inference_mode():
            for batch in tqdm(dataloader, desc=f"Evaluating ({split})"):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels']

                dtype = next(model.parameters()).dtype
                ctx = (
                    torch.autocast(device.type, dtype)
                    if dtype in {torch.bfloat16, torch.float16} and device.type == 'cuda' else contextlib.nullcontext()
                )

                gen_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'max_length': max_target_length,
                    'num_beams': num_beams,
                }
                
                if num_beams > 1:
                    if length_penalty is not None: gen_kwargs['length_penalty'] = length_penalty
                    if early_stopping is not None: gen_kwargs['early_stopping'] = early_stopping
                    if no_repeat_ngram_size is not None: gen_kwargs['no_repeat_ngram_size'] = no_repeat_ngram_size

                if constraint_decoding:
                    gen_kwargs['logits_processor'] = [
                            build_constraint_processor(
                                self.tokenizer,
                                input_ids,
                                self.tokens,
                                num_beams,
                                ent_schema=self.ent_schema,
                                rel_schema=self.rel_schema,
                            )
                        ]

                with ctx:
                    generated_ids = model.generate(**gen_kwargs)

                batch_size = input_ids.size(0)
                instances = [dataset[cursor + i] for i in range(batch_size)]
                cursor += batch_size

                pred_blocks, gold_blocks, pred_offsets, gold_offsets, batch_records = self.process_batch_outputs(
                    input_ids=input_ids,
                    generated_ids=generated_ids,
                    labels=labels,
                    instances=instances,
                )
                all_pred_blocks.extend(pred_blocks)
                all_gold_blocks.extend(gold_blocks)
                all_pred_offsets.extend(pred_offsets)
                all_gold_offsets.extend(gold_offsets)

                for record in batch_records:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        if cursor != len(dataset):
            logger.warning(
                "Consumed %d instances but the dataset holds %d; gold alignment may be off.",
                cursor, len(dataset),
            )

        metrics = self.compute_final_metrics(
            all_pred_blocks, all_gold_blocks, all_pred_offsets, all_gold_offsets
        )
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        logger.info("Saved evaluation metrics to %s", metrics_file)
        logger.info("Saved streaming predictions to %s", results_file)
        return metrics