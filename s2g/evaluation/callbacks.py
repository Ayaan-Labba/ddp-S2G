"""
Training callbacks for the S2G pipeline.

1. **StepTrackingCallback** — writes ``state.global_step`` to the
   collator's ``current_step`` property after each optimiser update.
   Budget-mode collators ignore the value; the callback is retained for
   Experiment 3 pre-training compatibility.

2. **GenerateTextSamplesCallback** — periodically generates predictions
   on a fixed sample batch, parses the SEL, and logs a W&B table
   comparing source text, predicted output, and gold output.  The
   display format adapts to the task: entity spans for Boundary/NER,
   triplets for RE/Joint/Joint+.

3. **PeriodicCheckpointCallback** — saves a resumable checkpoint at a
   fixed step interval, independent of the metric-based top-k saves.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from s2g.linearisation import (
    JOINT_TOKENS,
    PIPELINE_TOKENS,
    AnyTokens,
    extract_triplets,
    parse_sel,
)

logger = logging.getLogger(__name__)

# Map task names to the batch-dict key prefix used by S2GCollator.
# "joint+" uses "joint_plus" because "+" is not a valid Python identifier.
_TASK_TO_KEY: Dict[str, str] = {
    "boundary": "boundary",
    "ner":      "ner",
    "re":       "re",
    "joint":    "joint",
    "joint+":   "joint_plus",
}

# Pipeline tasks use PIPELINE_TOKENS; Joint tasks use JOINT_TOKENS.
_TASK_TO_TOK: Dict[str, AnyTokens] = {
    "boundary": PIPELINE_TOKENS,
    "ner":      PIPELINE_TOKENS,
    "re":       PIPELINE_TOKENS,
    "joint":    JOINT_TOKENS,
    "joint+":   JOINT_TOKENS,
}


# ---- STEP TRACKING CALLBACK ----


class StepTrackingCallback(TrainerCallback):
    """Synchronise the collator's step counter with the Trainer.

    Writes ``state.global_step`` to ``collator.current_step`` after each
    optimiser update.  In budget mode the collator's setter is a no-op;
    the callback is retained so Experiment 3 schedule-mode pre-training
    continues to work without changes to pretrain.py.

    Args:
        collator: The :class:`~s2g.data.S2GCollator` instance.
    """

    def __init__(self, collator: Any) -> None:
        self.collator = collator

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self.collator.current_step = state.global_step


# ---- GENERATE TEXT SAMPLES CALLBACK ----


class GenerateTextSamplesCallback(TrainerCallback):
    """Log predicted-vs-gold comparisons to W&B at regular intervals.

    Every *interval* global steps, this callback:

    1. Retrieves task-specific tensors from the collated sample batch.
    2. Runs autoregressive generation (no constraint decoding — this is
       a training-time qualitative check, not a test-time evaluation).
    3. Parses the generated and gold SEL strings using the token registry
       appropriate for *task*.
    4. Logs a W&B table with source text, predicted output, gold output,
       and both raw SEL strings.

    The display format is task-adapted:

    - **Boundary** — entity surface spans only.
    - **NER** — entity spans with type labels.
    - **RE / Joint / Joint+** — ``(head, rel_type, tail)`` triplets.

    Args:
        tokenizer:         HuggingFace tokeniser with S2G tokens.
        sample_batch:      Fixed list of raw instances from the
                           validation set.
        collator:          :class:`~s2g.data.S2GCollator`
                           used to tokenise *sample_batch*.
        task:              One of ``"boundary"``, ``"ner"``, ``"re"``,
                           ``"joint"``, ``"joint+"``.
        interval:          Log every *interval* global steps.
        eval_beams:        Beams for generation.
        max_target_length: Maximum generated sequence length.
    """

    def __init__(
        self,
        tokenizer:         PreTrainedTokenizerBase,
        sample_batch:      List[Dict],
        collator:          Any,
        task:              str,
        interval:          int = 10_000,
        eval_beams:        int = 3,
        max_target_length: int = 150,
    ) -> None:
        if task not in _TASK_TO_KEY:
            raise ValueError(
                f"Unknown task {task!r}. "
                f"Expected one of: {sorted(_TASK_TO_KEY)}."
            )
        self.tokenizer         = tokenizer
        self.sample_batch      = sample_batch
        self.collator          = collator
        self._task             = task
        self._task_key         = _TASK_TO_KEY[task]
        self._tok: AnyTokens   = _TASK_TO_TOK[task]
        self.interval          = interval
        self.eval_beams        = eval_beams
        self.max_target_length = max_target_length
        self._last_logged_step = -1

    def on_step_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        model:   Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero:
            return
        step = state.global_step
        if step == 0 or step == self._last_logged_step:
            return
        if step % self.interval != 0:
            return
        self._last_logged_step = step
        if model is None:
            logger.warning(
                "GenerateTextSamplesCallback: model not available at step %d.",
                step,
            )
            return
        try:
            self._log_samples(model, state)
        except Exception:
            logger.exception(
                "GenerateTextSamplesCallback failed at step %d.", step
            )

    def _log_samples(self, model: Any, state: TrainerState) -> None:
        """Generate predictions for the sample batch and log to W&B."""
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed; skipping sample generation.")
            return
        if wandb.run is None:
            return

        # ---- Collate and extract task-specific tensors ----
        batch       = self.collator(self.sample_batch)
        device      = next(model.parameters()).device
        key         = self._task_key
        input_ids   = batch[f"{key}_input_ids"].to(device)
        attn_mask   = batch[f"{key}_attention_mask"].to(device)
        labels      = batch[f"{key}_labels"].to(device)

        param_dtype  = next(model.parameters()).dtype
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=param_dtype)
            if param_dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"
            else contextlib.nullcontext()
        )
        model.eval()
        with torch.inference_mode(), autocast_ctx:
            unwrapped = model.module if hasattr(model, "module") else model
            generated_ids = unwrapped.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                num_beams=self.eval_beams,
                max_length=self.max_target_length,
                length_penalty=0.0,
                no_repeat_ngram_size=0,
                early_stopping=False,
            )
        model.train()

        # ---- Decode, parse, and format ----
        rows: List[List[str]] = []
        for i in range(len(self.sample_batch)):
            source_text = self.sample_batch[i]["text"]

            # Predicted SEL.
            pred_sel  = self.tokenizer.decode(generated_ids[i], skip_special_tokens=False)
            pred_sel  = _clean_decoded(pred_sel, self.tokenizer)
            pred_ents, _ = parse_sel(pred_sel, tok=self._tok)
            pred_trips   = extract_triplets(pred_ents)

            # Gold SEL (recover from labels; -100 → pad for decoding).
            gold_ids = labels[i].clone()
            gold_ids[gold_ids == -100] = self.tokenizer.pad_token_id
            gold_sel  = self.tokenizer.decode(gold_ids, skip_special_tokens=False)
            gold_sel  = _clean_decoded(gold_sel, self.tokenizer)
            gold_ents, _ = parse_sel(gold_sel, tok=self._tok)
            gold_trips    = extract_triplets(gold_ents)

            # Task-aware display string.
            pred_display = _format_output(self._task, pred_ents, pred_trips)
            gold_display = _format_output(self._task, gold_ents, gold_trips)

            rows.append([
                source_text,
                pred_display,
                gold_display,
                pred_sel.strip(),
                gold_sel.strip(),
            ])

        table = wandb.Table(
            columns=["Source", "Predicted", "Gold", "Predicted SEL", "Gold SEL"],
            data=rows,
        )
        wandb.log(
            {f"samples/{self._task}": table},
            step=state.global_step,
        )
        logger.info(
            "Logged %d %s samples to W&B at step %d.",
            len(rows), self._task, state.global_step,
        )


# ---- PERIODIC CHECKPOINT CALLBACK ----


class PeriodicCheckpointCallback(TrainerCallback):
    """Save a full resumable checkpoint at a fixed step interval.

    Provides a safety net against SSH disconnections.  Only one periodic
    checkpoint is kept at a time (overwritten each interval); this is
    separate from the top-k metric-based checkpoints.

    Args:
        output_dir:    Base output directory.
        every_n_steps: Save every *every_n_steps* global steps.
        wandb_run_id:  W&B run ID to persist for seamless resumption.
    """

    def __init__(
        self,
        output_dir:    str,
        every_n_steps: int = 5000,
        wandb_run_id:  Optional[str] = None,
    ) -> None:
        self.output_dir    = Path(output_dir)
        self.every_n_steps = every_n_steps
        self.wandb_run_id  = wandb_run_id
        self._last_saved   = -1

    def on_step_end(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        if step == 0 or step == self._last_saved:
            return
        if step % self.every_n_steps != 0:
            return
        self._last_saved      = step
        control.should_save   = True
        if self.wandb_run_id and state.is_world_process_zero:
            self._save_run_metadata(step)

    def _save_run_metadata(self, step: int) -> None:
        meta_path = self.output_dir / "run_metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"wandb_run_id": self.wandb_run_id, "last_step": step},
                f, indent=2,
            )


# ---- HELPERS ----


def _clean_decoded(text: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """Strip decoder artefacts from a decoded SEL string."""
    for tok in [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token]:
        if tok:
            text = text.replace(tok, "")
    return " ".join(text.split())


def _format_output(task: str, entities: List[Dict], triplets: List[tuple]) -> str:
    """Format parsed SEL output as a human-readable string.

    For Boundary: list entity surface spans.
    For NER: list entity spans with type annotations.
    For RE / Joint / Joint+: list ``(head, rel_type, tail)`` triplets.
    """
    if task == "boundary":
        lines = [e["text"] for e in entities] if entities else ["(none)"]
        return "\n".join(lines)

    if task == "ner":
        if not entities:
            return "(none)"
        lines = [
            f"{e['text']} [{e.get('type') or '?'}]"
            for e in entities
        ]
        return "\n".join(lines)

    # RE / Joint / Joint+
    return _format_triplets(triplets)


def _format_triplets(triplets: List[tuple]) -> str:
    """Format ``(head, rel_type, tail)`` triplets as multi-line string."""
    if not triplets:
        return "(none)"
    return "\n".join(f"({t[0]}, {t[1]}, {t[2]})" for t in triplets)


def load_run_metadata(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load W&B run metadata from *output_dir* for training resumption.

    Args:
        output_dir: Base output directory.

    Returns:
        Metadata dict or ``None`` if the file does not exist.
    """
    meta_path = Path(output_dir) / "run_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)