"""
Standalone VRAM measurement script for S2G training and evaluation.

Measures:
  - Training peak VRAM: binary-searches for the maximum train batch size that
    fits in VRAM (forward + backward, all task heads simultaneously).
  - Eval peak VRAM: binary-searches for the maximum eval batch size that fits
    within GPU memory, using model.generate() at max target length.

Usage:
    python -m s2g.scripts.measure_vram --config configs/finetune.yaml
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from s2g.linearisation import JOINT_TOKENS, PIPELINE_TOKENS, add_special_tokens_to_tokenizer
from s2g.scripts.config_utils import load_config

logger = logging.getLogger(__name__)

# Tasks exercised per model variant — must match compute_loss and _run_generation
_TRAIN_TASK_KEYS: Dict[str, Tuple[str, ...]] = {
    "pipeline": ("boundary", "ner", "re"),
    "joint":    ("joint", "joint_plus"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _mb(n_bytes: int) -> float:
    return n_bytes / 1024 ** 2


def _reset(device: torch.device) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _peak_allocated_mb(device: torch.device) -> float:
    return _mb(torch.cuda.max_memory_allocated(device))


def _peak_reserved_mb(device: torch.device) -> float:
    return _mb(torch.cuda.max_memory_reserved(device))


def _make_train_batch(
    task_keys: Tuple[str, ...],
    batch_size: int,
    max_src: int,
    max_tgt: int,
    pad_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch = {}
    for k in task_keys:
        batch[f"{k}_input_ids"]      = torch.full((batch_size, max_src), pad_id,  dtype=torch.long, device=device)
        batch[f"{k}_attention_mask"] = torch.ones ((batch_size, max_src),          dtype=torch.long, device=device)
        batch[f"{k}_labels"]         = torch.full ((batch_size, max_tgt), -100,   dtype=torch.long, device=device)
    return batch


def _make_gen_batch(
    batch_size: int,
    max_src: int,
    pad_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "input_ids":      torch.full((batch_size, max_src), pad_id, dtype=torch.long, device=device),
        "attention_mask": torch.ones ((batch_size, max_src),         dtype=torch.long, device=device),
    }


# ── Train check ───────────────────────────────────────────────────────────────

def check_train_vram(
    model: Any,
    pad_id: int,
    task_keys: Tuple[str, ...],
    batch_size: int,
    max_src: int,
    max_tgt: int,
    device: torch.device,
    precision: str = "fp32",
) -> Optional[Tuple[float, float]]:
    """
    Runs a full forward + backward at worst-case (max) lengths.
    Returns (peak_allocated_mb, peak_reserved_mb) on success, None on OOM.
    """
    model.train()
    _reset(device)
    batch = _make_train_batch(task_keys, batch_size, max_src, max_tgt, pad_id, device)

    autocast_ctx = _get_autocast_ctx(precision, device)

    try:
        with autocast_ctx:
            total_loss = None
            for k in task_keys:
                out = model(
                    input_ids=batch[f"{k}_input_ids"],
                    attention_mask=batch[f"{k}_attention_mask"],
                    labels=batch[f"{k}_labels"],
                )
                total_loss = out.loss if total_loss is None else total_loss + out.loss
                del out
            total_loss.backward()

        return _peak_allocated_mb(device), _peak_reserved_mb(device)

    except torch.cuda.OutOfMemoryError:
        return None

    finally:
        model.zero_grad(set_to_none=True)
        _reset(device)


def _populate_optimizer_states(
    model: Any,
    optimizer: torch.optim.Optimizer,
    pad_id: int,
    task_keys: Tuple[str, ...],
    max_src: int,
    max_tgt: int,
    device: torch.device,
    precision: str = "fp32",
) -> None:
    """
    Runs one forward → backward → optimizer.step() at batch_size=1, then
    zeros gradients WITHOUT calling empty_cache.

    This leaves the AdamW moment tensors (~2× model size) resident in VRAM,
    matching the memory footprint present when evaluation fires during real
    training.  Gradients themselves are released by zero_grad(set_to_none=True).
    """
    model.train()
    batch = _make_train_batch(task_keys, 1, max_src, max_tgt, pad_id, device)
    autocast_ctx = _get_autocast_ctx(precision, device)

    with autocast_ctx:
        total_loss = None
        for k in task_keys:
            out = model(
                input_ids=batch[f"{k}_input_ids"],
                attention_mask=batch[f"{k}_attention_mask"],
                labels=batch[f"{k}_labels"],
            )
            total_loss = out.loss if total_loss is None else total_loss + out.loss
            del out
        total_loss.backward()

    optimizer.step()
    # zero_grad without empty_cache — moments stay in VRAM, gradients are freed
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)


def binary_search_max_train_batch(
    model: Any,
    pad_id: int,
    task_keys: Tuple[str, ...],
    max_src: int,
    max_tgt: int,
    device: torch.device,
    precision: str = "fp32",
    search_max: int = 256,
) -> Tuple[int, Optional[float], Optional[float]]:
    """
    Binary-searches for the largest train batch size that fits in VRAM.
    Returns (max_batch_size, peak_allocated_mb, peak_reserved_mb).
    peak_* are None if even batch_size=1 OOMs.
    """
    result = check_train_vram(model, pad_id, task_keys, 1, max_src, max_tgt, device, precision)
    if result is None:
        return 0, None, None

    lo, hi = 1, search_max
    best, best_result = 1, result

    while lo <= hi:
        mid = (lo + hi) // 2
        r = check_train_vram(model, pad_id, task_keys, mid, max_src, max_tgt, device, precision)
        if r is not None:
            best, best_result = mid, r
            lo = mid + 1
        else:
            hi = mid - 1

    return best, best_result[0], best_result[1]


# ── Eval check ────────────────────────────────────────────────────────────────

def check_eval_vram_at(
    model: Any,
    pad_id: int,
    batch_size: int,
    max_src: int,
    max_tgt: int,
    num_beams: int,
    device: torch.device,
    precision: str = "fp32",
) -> Optional[Tuple[float, float]]:
    """
    Runs model.generate() at worst-case lengths for a given eval batch size.
    Returns (peak_allocated_mb, peak_reserved_mb) on success, None on OOM.
    """
    model.eval()
    _reset(device)
    batch = _make_gen_batch(batch_size, max_src, pad_id, device)

    gen_kwargs: Dict[str, Any] = {
        "num_beams": num_beams,
        "max_new_tokens": max_tgt,
        "min_new_tokens": max_tgt,   # Forces decoder to run all max_tgt steps;
                                     # without this, dummy padding inputs cause early EOS
                                     # and the KV cache never grows to its true peak size.
    }
    if num_beams > 1:
        gen_kwargs.update({"length_penalty": 0.0, "early_stopping": False})

    autocast_ctx = _get_autocast_ctx(precision, device)

    try:
        with torch.inference_mode(), autocast_ctx:
            model.generate(**batch, **gen_kwargs)

        return _peak_allocated_mb(device), _peak_reserved_mb(device)

    except torch.cuda.OutOfMemoryError:
        return None

    finally:
        _reset(device)


def binary_search_max_eval_batch(
    model: Any,
    pad_id: int,
    max_src: int,
    max_tgt: int,
    num_beams: int,
    device: torch.device,
    precision: str = "fp32",
    search_max: int = 256,
) -> Tuple[int, Optional[float], Optional[float]]:
    """
    Binary-searches for the largest eval batch size that fits in VRAM.
    Returns (max_batch_size, peak_allocated_mb, peak_reserved_mb).
    peak_* are None if even batch_size=1 OOMs.
    """
    # First check that batch_size=1 actually fits
    result = check_eval_vram_at(model, pad_id, 1, max_src, max_tgt, num_beams, device, precision)
    if result is None:
        return 0, None, None

    lo, hi = 1, search_max
    best, best_result = 1, result

    while lo <= hi:
        mid = (lo + hi) // 2
        r = check_eval_vram_at(model, pad_id, mid, max_src, max_tgt, num_beams, device, precision)
        if r is not None:
            best, best_result = mid, r
            lo = mid + 1
        else:
            hi = mid - 1

    return best, best_result[0], best_result[1]


# ── Precision context ─────────────────────────────────────────────────────────

def _get_autocast_ctx(precision: str, device: torch.device):
    import contextlib
    if device.type == "cuda":
        if precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()

    if not torch.cuda.is_available():
        logger.error("No CUDA device found — VRAM measurement requires a GPU.")
        return

    if cfg.hardware.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.hardware.gpu_ids[0])
    device = torch.device("cuda:0")

    logger.info("Loading model and tokenizer: %s", cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_checkpoint or cfg.model.name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.pretrained_checkpoint or cfg.model.name)
    tokens    = PIPELINE_TOKENS if cfg.model.model_variant == "pipeline" else JOINT_TOKENS
    add_special_tokens_to_tokenizer(tokenizer, tokens, model)
    model.to(device)

    task_keys = _TRAIN_TASK_KEYS[cfg.model.model_variant]
    pad_id    = tokenizer.pad_token_id
    max_src   = cfg.tokenization.max_source_length
    max_tgt   = cfg.tokenization.max_target_length
    precision = cfg.train.precision
    num_beams = cfg.generation.num_beams

    total_vram_mb = _mb(torch.cuda.get_device_properties(device).total_memory)
    logger.info("GPU total VRAM: %.0f MB", total_vram_mb)
    logger.info(
        "Config: variant=%s | tasks=%s | max_src=%d | max_tgt=%d | precision=%s | beams=%d",
        cfg.model.model_variant, list(task_keys), max_src, max_tgt, precision, num_beams,
    )

    # ── 1. Training VRAM — binary search ─────────────────────────────────────
    logger.info("─" * 60)
    logger.info("TRAIN CHECK — binary-searching max train batch size (forward+backward, all task heads)")

    max_train_bs, alloc_mb, reserved_mb = binary_search_max_train_batch(
        model, pad_id, task_keys, max_src, max_tgt, device, precision
    )

    if max_train_bs == 0:
        logger.error("TRAIN OOM — even batch_size=1 does not fit at max_src=%d, max_tgt=%d.", max_src, max_tgt)
    else:
        logger.info(
            "TRAIN max batch_size: %d — peak allocated: %.0f MB | reserved: %.0f MB | "
            "headroom: %.0f MB (%.1f%% of total)",
            max_train_bs, alloc_mb, reserved_mb,
            total_vram_mb - alloc_mb,
            100.0 * (total_vram_mb - alloc_mb) / total_vram_mb,
        )
        logger.info(
            "Suggested config: train.batch_size=%d  "
            "(set lower to leave headroom; account for gradient_acc_steps in effective batch size)",
            max_train_bs,
        )

    # ── 2. Eval VRAM — binary search with optimizer states resident ───────────
    logger.info("─" * 60)
    logger.info(
        "EVAL CHECK — populating AdamW optimizer states before search "
        "to match VRAM footprint present when eval fires during real training"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2),
        eps=cfg.optimizer.adam_epsilon,
        weight_decay=cfg.optimizer.weight_decay,
    )
    _populate_optimizer_states(model, optimizer, pad_id, task_keys, max_src, max_tgt, device, precision)

    baseline_mb = _mb(torch.cuda.memory_allocated(device))
    logger.info(
        "Optimizer states resident — VRAM baseline: %.0f MB | headroom for eval: %.0f MB",
        baseline_mb, total_vram_mb - baseline_mb,
    )
    logger.info(
        "Binary-searching max eval batch size (generate, num_beams=%d, max_tgt=%d)",
        num_beams, max_tgt,
    )

    max_eval_bs, alloc_mb, reserved_mb = binary_search_max_eval_batch(
        model, pad_id, max_src, max_tgt, num_beams, device, precision
    )

    if max_eval_bs == 0:
        logger.error("EVAL OOM — even batch_size=1 does not fit at max_src=%d, max_tgt=%d.", max_src, max_tgt)
    else:
        logger.info(
            "EVAL max batch_size: %d — peak allocated: %.0f MB | reserved: %.0f MB | "
            "headroom: %.0f MB (%.1f%% of total)",
            max_eval_bs, alloc_mb, reserved_mb,
            total_vram_mb - alloc_mb,
            100.0 * (total_vram_mb - alloc_mb) / total_vram_mb,
        )
        logger.info(
            "Suggested config: validation.batch_size=%d  "
            "(set lower to leave headroom for other GPU activity)",
            max_eval_bs,
        )

    logger.info("─" * 60)


if __name__ == "__main__":
    main()