"""
Training callbacks for the S2G pipeline.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import wandb
from transformers import (
    EarlyStoppingCallback, 
    PreTrainedTokenizerBase,
    AutoModelForSeq2SeqLM,
    TrainerCallback, 
    TrainerControl, 
    TrainerState
)

from s2g.data import S2GCollator
from s2g.linearisation import extract_triplets, parse_graph

logger = logging.getLogger(__name__)


class StepTrackingCallback(TrainerCallback):
    def __init__(self, collator: S2GCollator) -> None: 
        self.collator = collator
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        self.collator.current_step = state.global_step


class GenerateTextSamplesCallback(TrainerCallback):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        variant: str, 
        sample_batch: List[Dict], 
        collator: S2GCollator,
        interval: int = 1_000,
        num_beams: int = 3,
        max_target_length: int = 256
    ) -> None:
        if variant not in {'re', 'boundary_re', 'joint', 'boundary_joint'}: 
            raise ValueError(f"Unknown variant {variant!r}.")

        self.tokenizer = tokenizer
        self.sample_batch = sample_batch
        self.collator = collator
        self.variant = variant
        self.tok = collator.tok
        self.interval = interval
        self.num_beams = num_beams
        self.max_target_length =  max_target_length
        self.last_logged = -1

    def on_step_end(
            self, 
            args, 
            state: TrainerState, 
            control: TrainerControl, 
            model: AutoModelForSeq2SeqLM = None, 
            **kwargs
        ) -> None:
        if not state.is_world_process_zero or \
            state.global_step in {0, self.last_logged} or state.global_step % self.interval != 0: 
            return
            
        self.last_logged = state.global_step
        if model is None:
            logger.warning("GenerateTextSamplesCallback: no model at step %d.", state.global_step)
            return
        
        try: 
            self.log_samples(model, state, is_initial=False)
        except Exception: 
            logger.exception("GenerateTextSamplesCallback failed at step %d.", state.global_step)

    def on_train_begin(
            self, 
            args, 
            state: TrainerState, 
            control: TrainerControl, 
            model: AutoModelForSeq2SeqLM = None, 
            **kwargs
        ) -> None:
        if not state.is_world_process_zero: 
            return
            
        if model is None: 
            logger.warning("GenerateTextSamplesCallback: no model at train begin.")
            return
            
        try: 
            self.log_samples(model, state, is_initial=True)
            self.last_logged = 0
        except Exception: 
            logger.exception("GenerateTextSamplesCallback failed at train begin.")

    def log_samples(self, model: AutoModelForSeq2SeqLM, state: TrainerState, is_initial: bool = False) -> None:
        if wandb.run is None: 
            return

        batch = self.collator(self.sample_batch)
        device = next(model.parameters()).device
        k, dtype = self.variant, next(model.parameters()).dtype

        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attn_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        ctx = torch.autocast(device.type, dtype) \
        if dtype in {torch.bfloat16, torch.float16} and device.type == 'cuda' else contextlib.nullcontext()
        
        model.eval()
        with torch.inference_mode(), ctx:
            gen_kwargs = {
                'input_ids': input_ids,
                'attention_mask': attn_mask,
                'num_beams': self.num_beams,
                'max_length': self.max_target_length
            }

            generated_ids = (model.module if hasattr(model, 'module') else model).generate(**gen_kwargs) # model.module for DDP

        model.train()

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        g_ids = labels.clone()
        g_ids.masked_fill_(g_ids == -100, pad_id)
        pred_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        gold_texts = self.tokenizer.batch_decode(g_ids, skip_special_tokens=False)

        # Get prompts
        prompts = []
        for seq in input_ids:
            seq_nopad = seq[seq != pad_id]
            prompts.append(self.tokenizer.decode(seq_nopad, skip_special_tokens=False))

        specials_to_remove = [t for t in (self.tokenizer.pad_token, self.tokenizer.eos_token, self.tokenizer.bos_token) if t]
        rows = []
        
        cols = [
            'Source', 
            'Encoder Input', 
            'Predicted Entities', 
            'Gold Entities', 
            'Predicted Triplets', 
            'Gold Triplets', 
            'Predicted Graph', 
            'Gold Graph'
        ]
        
        for i, inst in enumerate(self.sample_batch):
            p_graph, g_graph = pred_texts[i], gold_texts[i]
            
            for tok in specials_to_remove:
                p_graph = p_graph.replace(tok, "")
                g_graph = g_graph.replace(tok, "")
                
            p_graph = " ".join(p_graph.split())
            g_graph = " ".join(g_graph.split())
            
            p_ent, _ = parse_graph(p_graph, tok=self.tok)
            g_ent, _ = parse_graph(g_graph, tok=self.tok)

            row = [inst['text'], prompts[i]]
            
            include_types = self.variant in {'joint', 're'}
            
            # Format entities (with types for joint/re, without types for boundary_*)
            if include_types:
                p_e = "\n".join([f"{e['text']} [{e.get('type')}]" for e in p_ent]) if p_ent else "(none)"
                g_e = "\n".join([f"{e['text']} [{e.get('type')}]" for e in g_ent]) if g_ent else "(none)"
            else:
                p_e = "\n".join([f"{e['text']}" for e in p_ent]) if p_ent else "(none)"
                g_e = "\n".join([f"{e['text']}" for e in g_ent]) if g_ent else "(none)"

            # Format triplets
            p_triplets = extract_triplets(p_ent, include_types=include_types)
            g_triplets = extract_triplets(g_ent, include_types=include_types)
            p_t = "\n".join([f"({t[0]}, {t[1]}, {t[2]})" for t in p_triplets]) if p_triplets else "(none)"
            g_t = "\n".join([f"({t[0]}, {t[1]}, {t[2]})" for t in g_triplets]) if g_triplets else "(none)"

            row.extend([p_e, g_e, p_t, g_t, p_graph, g_graph])
            rows.append(row)

        wandb.log(
            {f"samples/{self.variant}": wandb.Table(columns=cols, data=rows)}, 
            step=0 if is_initial else state.global_step
        )
        logger.info("Logged %d %s samples to W&B at step %d.", len(rows), self.variant, 0 if is_initial else state.global_step)


class PeriodicCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: str, every_n_steps: int = 5000, wandb_run_id: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir)
        self.every_n_steps = every_n_steps
        self.wandb_run_id = wandb_run_id
        self.last_saved = -1

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if state.global_step in {0, self.last_saved} or state.global_step % self.every_n_steps != 0: 
            return
            
        self.last_saved, control.should_save = state.global_step, True
        
        if self.wandb_run_id and state.is_world_process_zero:
            m_path = self.output_dir / "run_metadata.json"
            m_path.parent.mkdir(parents=True, exist_ok=True) # verify if needed
            with open(m_path, 'w', encoding='utf-8') as f: 
                json.dump({'wandb_run_id': self.wandb_run_id, 'last_step': state.global_step}, f, indent=2)


class S2GEarlyStoppingCallback(EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        """
        Override to prevent early stopping counter from incrementing if the best metric so far is still <= 0.0
        """
        super().check_metric_value(args, state, control, metric_value)
        if args.greater_is_better and (state.best_metric is None or state.best_metric <= 0.0):
            self.early_stopping_patience_counter = 0


def load_run_metadata(output_dir: str) -> Optional[Dict[str, Any]]:
    m_path = Path(output_dir) / "run_metadata.json"
    if m_path.exists():
        with open(m_path, 'r', encoding='utf-8') as f: return json.load(f)
    else:
        raise FileNotFoundError(f"Metaadata file not found: {m_path}")

    return