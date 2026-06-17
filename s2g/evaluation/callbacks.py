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
from transformers import EarlyStoppingCallback, PreTrainedTokenizerBase, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from s2g.linearisation import S2GTokens, AnyTokens, extract_triplets, parse_sel

logger = logging.getLogger(__name__)

_TASK_TO_KEY = {
    "re": "re",
    "boundary_re": "boundary_re",
    "boundary_joint": "boundary_joint",
    "joint": "joint",
}


class StepTrackingCallback(TrainerCallback):
    def __init__(self, collator: Any) -> None: 
        self.collator = collator
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        self.collator.current_step = state.global_step


class GenerateTextSamplesCallback(TrainerCallback):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, sample_batch: List[Dict], collator: Any, task: str,
        interval: int = 10_000, eval_beams: int = 3, max_target_length: int = 150,
    ) -> None:
        if task not in _TASK_TO_KEY: 
            raise ValueError(f"Unknown task {task!r}.")
        self.tokenizer = tokenizer
        self.sample_batch = sample_batch
        self.collator = collator
        self._task, self._task_key = task, _TASK_TO_KEY[task]
        self._tok = collator._tok
        self.interval, self.eval_beams, self.max_target_length, self._last_logged = interval, eval_beams, max_target_length, -1

    def on_step_end(
            self, args: TrainingArguments, state: TrainerState, control: TrainerControl, 
            model: Optional[Any] = None, **kwargs: Any
        ) -> None:
        if not state.is_world_process_zero or state.global_step in {0, self._last_logged} or state.global_step % self.interval != 0: 
            return
            
        self._last_logged = state.global_step
        if model is None: 
            logger.warning("GenerateTextSamplesCallback: no model at step %d.", state.global_step)
            return
        
        try: 
            self._log_samples(model, state, is_initial=False)
        except Exception: 
            logger.exception("GenerateTextSamplesCallback failed at step %d.", state.global_step)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: Optional[Any] = None, **kwargs: Any) -> None:
        if not state.is_world_process_zero: 
            return
            
        if model is None: 
            logger.warning("GenerateTextSamplesCallback: no model at train begin.")
            return
            
        try: 
            self._log_samples(model, state, is_initial=True)
            self._last_logged = 0
        except Exception: 
            logger.exception("GenerateTextSamplesCallback failed at train begin.")

    def _log_samples(self, model: Any, state: TrainerState, is_initial: bool = False) -> None:
        try: 
            import wandb
        except ImportError: 
            return
        if wandb.run is None: 
            return

        batch = self.collator(self.sample_batch)
        device = next(model.parameters()).device
        k, dtype = self._task_key, next(model.parameters()).dtype
        
        input_ids = batch[f"{k}_input_ids"].to(device, non_blocking=True)
        attn_mask = batch[f"{k}_attention_mask"].to(device, non_blocking=True)
        labels = batch[f"{k}_labels"].to(device, non_blocking=True)

        ctx = torch.autocast(device.type, dtype) if dtype in {torch.bfloat16, torch.float16} and device.type == "cuda" else contextlib.nullcontext()
        
        model.eval()
        with torch.inference_mode(), ctx:
            generated_ids = (model.module if hasattr(model, "module") else model).generate(
                input_ids=input_ids, attention_mask=attn_mask, num_beams=self.eval_beams, max_length=self.max_target_length,
                length_penalty=0.0, no_repeat_ngram_size=0, early_stopping=False,
            )
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
        
        cols = ["Source", "Encoder Input"]
        if self._task in {"joint", "boundary_joint"}:
            cols.extend(["Predicted Entities", "Gold Entities"])
        if self._task in {"re", "boundary_re", "joint", "boundary_joint"}:
            cols.extend(["Predicted Triplets", "Gold Triplets"])
        cols.extend(["Predicted SEL", "Gold SEL"])
        
        for i, inst in enumerate(self.sample_batch):
            p_sel, g_sel = pred_texts[i], gold_texts[i]
            
            for tok in specials_to_remove:
                p_sel = p_sel.replace(tok, "")
                g_sel = g_sel.replace(tok, "")
                
            p_sel = " ".join(p_sel.split())
            g_sel = " ".join(g_sel.split())
            
            p_ent, _ = parse_sel(p_sel, tok=self._tok)
            g_ent, _ = parse_sel(g_sel, tok=self._tok)

            row = [inst["text"], prompts[i]]
            
            if self._task in {"joint", "boundary_joint"}:
                p_e = "\n".join([f"{e['text']} [{e.get('type') or '?'}]" for e in p_ent]) if p_ent else "(none)"
                g_e = "\n".join([f"{e['text']} [{e.get('type') or '?'}]" for e in g_ent]) if g_ent else "(none)"
                row.extend([p_e, g_e])


            if self._task in {"re", "boundary_re", "joint", "boundary_joint"}:
                p_triplets = extract_triplets(p_ent, include_types=(self._task == "re"))
                g_triplets = extract_triplets(g_ent, include_types=(self._task == "re"))
                p_t = "\n".join([f"({t[0]}, {t[1]}, {t[2]})" for t in p_triplets]) if p_triplets else "(none)"
                g_t = "\n".join([f"({t[0]}, {t[1]}, {t[2]})" for t in g_triplets]) if g_triplets else "(none)"
                row.extend([p_t, g_t])
                
            row.extend([p_sel, g_sel])
            rows.append(row)

        wandb.log(
            {f"samples/{self._task}": wandb.Table(columns=cols, data=rows)}, 
            step=0 if is_initial else state.global_step
        )
        logger.info("Logged %d %s samples to W&B at step %d.", len(rows), self._task, 0 if is_initial else state.global_step)


class PeriodicCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir: str, every_n_steps: int = 5000, wandb_run_id: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir)
        self.every_n_steps = every_n_steps
        self.wandb_run_id = wandb_run_id
        self._last_saved = -1

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        if state.global_step in {0, self._last_saved} or state.global_step % self.every_n_steps != 0: 
            return
            
        self._last_saved, control.should_save = state.global_step, True
        
        if self.wandb_run_id and state.is_world_process_zero:
            m_path = self.output_dir / "run_metadata.json"
            m_path.parent.mkdir(parents=True, exist_ok=True)
            with open(m_path, "w", encoding="utf-8") as f: 
                json.dump({"wandb_run_id": self.wandb_run_id, "last_step": state.global_step}, f, indent=2)





def load_run_metadata(output_dir: str) -> Optional[Dict[str, Any]]:
    m_path = Path(output_dir) / "run_metadata.json"
    if m_path.exists():
        with open(m_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


class S2GEarlyStoppingCallback(EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        super().check_metric_value(args, state, control, metric_value)
        # Prevent early stopping counter from incrementing if the best metric so far is still <= 0.0
        # (meaning the model hasn't started increasing/improving from its initial value)
        if args.greater_is_better and (state.best_metric is None or state.best_metric <= 0.0):
            self.early_stopping_patience_counter = 0