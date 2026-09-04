"""
Standalone VRAM measurement script for S2G training and evaluation.

Measures:
  - Training peak VRAM: binary-searches for the maximum train batch size that
    fits in VRAM (forward + backward pass).
  - Eval peak VRAM: binary-searches for the maximum eval batch size that fits
    within GPU memory, using model.generate() at max target length.

Usage:
    python -m s2g.scripts.measure_vram --config configs/finetune.yaml
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from s2g.linearisation import S2GTokens, verify_token_integrity
from s2g.scripts.config_utils import load_config

logger = logging.getLogger(__name__)


def _mb(n_bytes: int) -> float:
    return n_bytes / 1024 ** 2


def _reset(device: torch.device) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _peak_allocated_mb(device: torch.device) -> float:
    return _mb(torch.cuda.max_memory_allocated(device))


def _peak_reserved_mb(device: torch.device) -> float:
    return _mb(torch.cuda.max_memory_reserved(device))


def _get_autocast_ctx(precision: str, device: torch.device):
    if device.type == "cuda":
        if precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _make_train_batch(
    batch_size: int,
    max_src: int,
    max_tgt: int,
    pad_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.full((batch_size, max_src), pad_id, dtype=torch.long, device=device),
        "attention_mask": torch.ones((batch_size, max_src), dtype=torch.long, device=device),
        "labels": torch.full((batch_size, max_tgt), pad_id, dtype=torch.long, device=device),
    }


def _make_gen_batch(
    batch_size: int,
    max_src: int,
    pad_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.full((batch_size, max_src), pad_id, dtype=torch.long, device=device),
        "attention_mask": torch.ones((batch_size, max_src), dtype=torch.long, device=device),
    }


def check_train_vram(
    model: Any,
    pad_id: int,
    batch_size: int,
    max_src: int,
    max_tgt: int,
    device: torch.device,
    precision: str = "fp32",
    gradient_checkpointing: bool = False,
) -> Optional[Tuple[float, float]]:
    """
    Runs a full forward + backward pass at maximum lengths.
    Returns (peak_allocated_mb, peak_reserved_mb) on success, None on OOM.
    """
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()

    model.train()
    _reset(device)
    batch = _make_train_batch(batch_size, max_src, max_tgt, pad_id, device)
    autocast_ctx = _get_autocast_ctx(precision, device)

    try:
        with autocast_ctx:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss
            del out
            loss.backward()

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
    max_src: int,
    max_tgt: int,
    device: torch.device,
    precision: str = "fp32",
    gradient_checkpointing: bool = False,
) -> None:
    """
    Runs one forward -> backward -> optimizer.step() at batch_size=1, then
    zeros gradients without emptying cache. This keeps optimizer moment tensors
    resident in VRAM to accurately measure headroom during evaluation.
    """
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()

    model.train()
    batch = _make_train_batch(1, max_src, max_tgt, pad_id, device)
    autocast_ctx = _get_autocast_ctx(precision, device)

    with autocast_ctx:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss
        del out
        loss.backward()

    optimizer.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)


def binary_search_max_train_batch(
    model: Any,
    pad_id: int,
    max_src: int,
    max_tgt: int,
    device: torch.device,
    precision: str = "fp32",
    gradient_checkpointing: bool = False,
    search_max: int = 256,
) -> Tuple[int, Optional[float], Optional[float]]:
    result = check_train_vram(model, pad_id, 1, max_src, max_tgt, device, precision, gradient_checkpointing)
    if result is None:
        return 0, None, None

    lo, hi = 1, search_max
    best, best_result = 1, result

    while lo <= hi:
        mid = (lo + hi) // 2
        r = check_train_vram(model, pad_id, mid, max_src, max_tgt, device, precision, gradient_checkpointing)
        if r is not None:
            best, best_result = mid, r
            lo = mid + 1
        else:
            hi = mid - 1

    return best, best_result[0], best_result[1]


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
    model.eval()
    _reset(device)
    batch = _make_gen_batch(batch_size, max_src, pad_id, device)

    gen_kwargs: Dict[str, Any] = {
        "num_beams": num_beams,
        "max_new_tokens": max_tgt,
        "min_new_tokens": max_tgt,
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()

    if cfg.hardware.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.hardware.gpu_ids[0])

    if not torch.cuda.is_available():
        logger.error("No CUDA device found — VRAM measurement requires a GPU.")
        return

    device = torch.device("cuda:0")

    logger.info("Loading model and tokenizer: %s", cfg.model.pretrained_checkpoint or cfg.model.name)
    precision_to_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    dtype = precision_to_dtype.get(cfg.train.precision, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_checkpoint or cfg.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model.pretrained_checkpoint or cfg.model.name, dtype=dtype
    )
    if hasattr(model.generation_config, "forced_bos_token_id"):
        model.generation_config.forced_bos_token_id = None

    tokens = S2GTokens(
        cfg.model.variant, use_rejection=cfg.graph.use_rejection,
        inline_none=cfg.graph.inline_none,
    )
    verify_token_integrity(tokenizer)
    model.to(device)

    variant = cfg.model.variant
    pad_id = tokenizer.pad_token_id
    max_src = cfg.tokenizer.max_source_length
    max_tgt = cfg.tokenizer.max_target_length
    precision = cfg.train.precision
    num_beams = cfg.generation.num_beams

    total_vram_mb = _mb(torch.cuda.get_device_properties(device).total_memory)
    logger.info("GPU total VRAM: %.0f MB", total_vram_mb)
    logger.info(
        "Config: variant=%s | max_src=%d | max_tgt=%d | precision=%s | beams=%d",
        variant, max_src, max_tgt, precision, num_beams,
    )
    logger.info("─" * 60)
    logger.info(
        "TRAIN CHECK — binary-searching max train batch size (forward+backward%s)",
        ", gradient_checkpointing=True" if cfg.train.gradient_checkpointing else "",
    )

    max_train_bs, alloc_mb, reserved_mb = binary_search_max_train_batch(
        model,
        pad_id,
        max_src,
        max_tgt,
        device,
        precision,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
    )

    if max_train_bs == 0:
        logger.error("TRAIN OOM — even batch_size=1 does not fit at max_src=%d, max_tgt=%d.", max_src, max_tgt)
    else:
        logger.info(
            "TRAIN max batch_size: %d — peak allocated: %.0f MB | reserved: %.0f MB | headroom: %.0f MB (%.1f%% of total)",
            max_train_bs,
            alloc_mb,
            reserved_mb,
            total_vram_mb - alloc_mb,
            100.0 * (total_vram_mb - alloc_mb) / total_vram_mb,
        )
        logger.info(
            "Suggested config: train.batch_size=%d (set lower to leave headroom for gradient_acc_steps)",
            max_train_bs,
        )
    logger.info("─" * 60)
    logger.info(
        "EVAL CHECK — populating AdamW optimizer states before search to match real training VRAM footprint"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(cfg.optimizer, "lr", 5e-5),
        betas=(getattr(cfg.optimizer, "adam_beta1", 0.9), getattr(cfg.optimizer, "adam_beta2", 0.999)),
        eps=getattr(cfg.optimizer, "adam_epsilon", 1e-8),
        weight_decay=getattr(cfg.optimizer, "weight_decay", 0.0),
    )
    _populate_optimizer_states(
        model,
        optimizer,
        pad_id,
        max_src,
        max_tgt,
        device,
        precision,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
    )

    model.gradient_checkpointing_disable()
    model.config.use_cache = True

    baseline_mb = _mb(torch.cuda.memory_allocated(device))
    logger.info(
        "Optimizer states resident — VRAM baseline: %.0f MB | headroom for eval: %.0f MB",
        baseline_mb,
        total_vram_mb - baseline_mb,
    )
    logger.info("Binary-searching max eval batch size (generate, num_beams=%d, max_tgt=%d)", num_beams, max_tgt)

    max_eval_bs, alloc_mb, reserved_mb = binary_search_max_eval_batch(
        model, pad_id, max_src, max_tgt, num_beams, device, precision
    )

    if max_eval_bs == 0:
        logger.error("EVAL OOM — even batch_size=1 does not fit at max_src=%d, max_tgt=%d.", max_src, max_tgt)
    else:
        logger.info(
            "EVAL max batch_size: %d — peak allocated: %.0f MB | reserved: %.0f MB | headroom: %.0f MB (%.1f%% of total)",
            max_eval_bs,
            alloc_mb,
            reserved_mb,
            total_vram_mb - alloc_mb,
            100.0 * (total_vram_mb - alloc_mb) / total_vram_mb,
        )
        logger.info("Suggested config: validation.batch_size=%d", max_eval_bs)

    logger.info("─" * 60)


if __name__ == "__main__":
    main()