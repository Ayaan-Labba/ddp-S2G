"""
Configuration loader for the S2G pipeline (OmegaConf-based).

Defines a fully typed nested schema (``S2GConfig`` and its subsection
dataclasses) and merges three layers in resolution order:

    1. Dataclass defaults  — schema, source of truth for fields.
    2. YAML overlay        — per-experiment values (configs/*.yaml).
    3. CLI dotlist         — per-run overrides (key=value, dotted for
                              nested fields, e.g. ``optimizer.lr=3e-5``).

Because the schema is enforced in struct mode, unknown keys raise a clear
error at load time rather than silently shadowing the real value.

S2G changes from Vanilla S2G
------------------------------
- ``ModelConfig``: new ``model_variant`` field (``"pipeline"`` | ``"joint"``).
- ``DataConfig``: new ``entity_schema_file`` field.
- ``SSIConfig``: schedule-mode fields removed (budget mode only);
  ``max_types_in_prompt`` split into ``max_rel_types_in_prompt`` and
  ``max_ent_types_in_prompt``.
- ``TypedSELConfig`` removed (not part of Experiment 1 architecture).
- ``EvaluationConfig.mode`` removed; ``compute_metrics_for_task`` uses the
  task to select metrics automatically.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ===================================================================== #
#                       NESTED CONFIG SCHEMA                             #
# ===================================================================== #


@dataclass
class ModelConfig:
    """Backbone selection and fine-tuning starting point."""
    name:                  str          = "google/flan-t5-base"
    pretrained_checkpoint: Optional[str] = None
    model_variant:         str          = "pipeline"  # "pipeline" | "joint"


@dataclass
class TokenizationConfig:
    """Encoder / decoder length budgets in subword tokens."""
    max_source_length: int = 400
    max_target_length: int = 200


@dataclass
class OptimizerConfig:
    """Optimiser type and hyperparameters."""
    optim:       str   = "adamw_torch"
    lr:          float = 5e-5
    weight_decay: float = 0.0
    adam_beta1:  float = 0.9
    adam_beta2:  float = 0.999
    adam_epsilon: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning-rate schedule.

    ``type = "inverse_sqrt"``  — for REBEL pre-training (Experiment 3).
    ``type = "cosine"``        — for benchmark fine-tuning (Experiment 1);
                                  set ``lr_scheduler_type`` in
                                  ``TrainingArguments`` accordingly and
                                  let the parent Trainer build it.
    """
    type:         str = "cosine"
    warmup_steps: int = 1_000


@dataclass
class TrainConfig:
    """Training-loop hyperparameters."""
    max_steps:              int   = 20_000
    batch_size:             int   = 8        # Per-device
    gradient_acc_steps:     int   = 4
    gradient_clip_value:    float = 10.0
    gradient_checkpointing: bool  = False
    precision:              str   = "bf16"   # "16", "bf16", or "32"
    seed:                   int   = 0


@dataclass
class ValidationConfig:
    """Validation loop and early stopping."""
    check_interval:             int            = 1_000
    percent_check:              float          = 1.0    # Fraction of val set
    train_eval_percent_check:   Optional[float] = None   # null = disabled
    batch_size:                 int            = 32
    early_stopping_patience:    int            = 10
    early_stopping_metric:      str            = "re_rel_boundary_f1"


@dataclass
class GenerationConfig:
    """Beam-search settings used at validation and evaluation."""
    num_beams:             int   = 4
    length_penalty:        float = 0.0
    no_repeat_ngram_size:  int   = 0
    early_stopping:        bool  = False
    constraint_decoding:   bool  = False


@dataclass
class SSIConfig:
    """Schema-Sensitive-Input sampling parameters.

    ``mode = "budget"``   (Experiment 1 fine-tuning)
        All gold-positive types always included; remaining budget filled
        with uniformly sampled negatives.  Schedule fields are ignored.

    ``mode = "bernoulli"``  (Experiment 3 REBEL pre-training)
        Each gold-positive type is independently included with
        probability ``positive_rate(t)``; each negative type is
        independently included with probability ``negative_rate(t)``,
        capped at ``k(t)`` total negatives.  All three quantities follow
        linear schedules from their ``_start`` to ``_end`` value over
        ``max_steps`` training steps.

    For Joint+, entity types and relation types are sampled
    independently using ``max_ent_types_in_prompt`` and
    ``max_rel_types_in_prompt`` as hard caps.  Null means no cap.
    """
    mode: str = "budget"   # "budget" | "bernoulli"

    # ---- Budget mode ----
    max_ent_types_in_prompt: Optional[int] = None
    max_rel_types_in_prompt: Optional[int] = None

    # ---- Bernoulli mode (schedule fields) ----
    max_steps:            int   = 150_000  # total training steps T
    positive_rate_start:  float = 0.9
    positive_rate_end:    float = 0.9
    negative_rate_start:  float = 0.1
    negative_rate_end:    float = 0.1
    negative_max_start:   int   = 1        # k(0)
    negative_max_end:     int   = 20       # k(T)

    # ---- Shared ----
    random_prompt: bool = False
    random_sel:    bool = False


@dataclass
class CheckpointConfig:
    """Checkpointing and resumption."""
    save_top_k:    int          = 3
    every_n_steps: int          = 500
    resume_from:   Optional[str] = None


@dataclass
class CallbacksConfig:
    """Custom-callback intervals."""
    sample_generation_interval: int = 5_000


@dataclass
class WandbConfig:
    """Weights & Biases run metadata."""
    project:  str          = "s2g"
    entity:   Optional[str] = None
    run_name: Optional[str] = None


@dataclass
class DataConfig:
    """Data and output paths."""
    data_dir:           Optional[str] = None
    schema_file:        Optional[str] = None  # relation.schema
    entity_schema_file: Optional[str] = None  # entity.schema (null = empty schema)
    output_dir:         Optional[str] = None


@dataclass
class HardwareConfig:
    """GPU selection and dataloader workers."""
    num_workers:        int            = 0
    persistent_workers: bool           = False
    gpu_ids:            Optional[List[int]] = None


@dataclass
class EvaluationConfig:
    """Final-evaluation settings (consumed by ``evaluate.py`` only).

    Only ``evaluate.py`` reads this section.  ``pretrain.py`` and
    ``finetune.py`` ignore it.
    """
    split: str = "test"    # "val" or "test"


@dataclass
class S2GConfig:
    """Top-level config aggregating every nested subsection."""
    model:        ModelConfig        = field(default_factory=ModelConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    optimizer:    OptimizerConfig    = field(default_factory=OptimizerConfig)
    scheduler:    SchedulerConfig    = field(default_factory=SchedulerConfig)
    train:        TrainConfig        = field(default_factory=TrainConfig)
    validation:   ValidationConfig   = field(default_factory=ValidationConfig)
    generation:   GenerationConfig   = field(default_factory=GenerationConfig)
    ssi:          SSIConfig          = field(default_factory=SSIConfig)
    checkpoint:   CheckpointConfig   = field(default_factory=CheckpointConfig)
    callbacks:    CallbacksConfig    = field(default_factory=CallbacksConfig)
    wandb:        WandbConfig        = field(default_factory=WandbConfig)
    data:         DataConfig         = field(default_factory=DataConfig)
    hardware:     HardwareConfig     = field(default_factory=HardwareConfig)
    evaluation:   EvaluationConfig   = field(default_factory=EvaluationConfig)
    # Provenance: filled in by load_config().
    config_path:  Optional[str]      = None


# ===================================================================== #
#                             LOADER                                     #
# ===================================================================== #


def load_config(
    config_path: Optional[str] = None,
    cli_args:    Optional[List[str]] = None,
) -> DictConfig:
    """Build a typed nested config from defaults, YAML overlay, and CLI.

    Resolution order (later layers override earlier):
        1. Dataclass defaults.
        2. YAML at *config_path* (or ``--config <path>`` from *cli_args*).
        3. CLI dotlist overrides — bare ``key=value`` pairs, dotted for
           nested fields (e.g. ``optimizer.lr=3e-5``).

    Returns:
        An OmegaConf ``DictConfig`` in struct mode.

    Raises:
        FileNotFoundError: if the YAML file does not exist.
        ValueError:        if a CLI arg is malformed.
    """
    if cli_args is None:
        cli_args = sys.argv[1:]

    yaml_path, remaining = _extract_config_flag(cli_args, config_path)
    _validate_dotlist(remaining)

    cfg = OmegaConf.structured(S2GConfig)

    if yaml_path is not None:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
        logger.info("Loaded config from %s", path)
    else:
        logger.warning("No --config path provided; using schema defaults only.")

    if remaining:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(remaining))
        logger.info("Applied %d CLI override(s).", len(remaining))

    cfg.config_path = str(yaml_path) if yaml_path else None
    return cfg


# ===================================================================== #
#                           SCHEMA LOADERS                              #
# ===================================================================== #


def load_schema(schema_path: str) -> List[str]:
    """Load a schema file (one type string per line).

    Args:
        schema_path: Path to the ``relation.schema`` or ``entity.schema``
                     file.

    Returns:
        Sorted list of type strings (order matches the file).
    """
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_entity_schema(entity_schema_file: Optional[str]) -> List[str]:
    """Load the entity schema, returning an empty list if path is null.

    Args:
        entity_schema_file: Path to ``entity.schema``, or ``None`` when
                            entity types are not used (e.g. REBEL).

    Returns:
        List of entity-type strings, or ``[]`` when path is ``None``.
    """
    if entity_schema_file is None:
        return []
    return load_schema(entity_schema_file)


# ===================================================================== #
#                       PRIVATE HELPERS                                  #
# ===================================================================== #


def _extract_config_flag(
    cli_args:      List[str],
    explicit_path: Optional[str],
) -> Tuple[Optional[str], List[str]]:
    yaml_path: Optional[str] = explicit_path
    remaining: List[str] = []
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if arg == "--config":
            if i + 1 >= len(cli_args):
                raise ValueError("--config flag requires a path argument.")
            if explicit_path is None:
                yaml_path = cli_args[i + 1]
            i += 2
            continue
        if arg.startswith("--config="):
            if explicit_path is None:
                yaml_path = arg.split("=", 1)[1]
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return yaml_path, remaining


def _validate_dotlist(args: List[str]) -> None:
    for arg in args:
        if arg.startswith("-"):
            raise ValueError(
                f"Unrecognised CLI flag: '{arg}'.  Overrides must be in "
                "dotlist form, e.g. 'optimizer.lr=3e-5' (no leading dashes). "
                "The only flag accepted is '--config <path>'."
            )
        if "=" not in arg:
            raise ValueError(
                f"Malformed override: '{arg}'.  Expected 'key=value' "
                "(use dotted keys for nested fields)."
            )