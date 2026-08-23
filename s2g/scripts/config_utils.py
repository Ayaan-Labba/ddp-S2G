"""
Configuration loader for the S2G pipeline (OmegaConf-based).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    data_dir: Optional[str] = None
    rel_schema: Optional[str] = None
    ent_schema: Optional[str] = None
    output_dir: Optional[str] = None

@dataclass
class ModelConfig:
    name: Optional[str] = "google/flan-t5-base"
    pretrained_checkpoint: Optional[str] = None
    variant: str = 're'

@dataclass
class TokenizerConfig:
    max_source_length: int = 256
    max_target_length: int = 256

@dataclass
class PromptConfig:
    mode: str = 'budget'
    type: str = 'natural'
    random_prompt: bool = False
    max_ent_types: Optional[int] = None   # null = use the full entity schema
    max_rel_types: Optional[int] = None   # null = use the full relation schema
    pos_rate: Optional[float] = 0.5
    neg_rate: Optional[float] = 0.5
    pos_rate_start: Optional[float] = 0.9
    pos_rate_end: Optional[float] = 0.5
    neg_rate_start: Optional[float] = 0.1
    neg_rate_end: Optional[float] = 0.5
    pos_max_start: Optional[int] = 1
    pos_max_end: Optional[int] = 10
    neg_max_start: Optional[int] = 1
    neg_max_end: Optional[int] = 10

@dataclass
class GraphConfig:
    random_graph: bool = False
    use_rejection: bool = False
    use_nesting: bool = True
    dedup: bool = True

@dataclass
class OptimizerConfig:
    optim: str = 'adamw_torch'
    lr: float = 3e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

@dataclass
class SchedulerConfig:
    type: str = 'cosine'
    warmup_steps: int = 1_000

@dataclass
class TrainConfig:
    max_steps: int = 10_000
    steps_per_log: int = 100
    batch_size: int = 8
    gradient_acc_steps: int = 4
    gradient_clip_value: float = 10.0
    gradient_checkpointing: bool = False
    precision: str = 'bf16'
    seed: int = 0
    warm_start: bool = False

@dataclass
class ValidationConfig:
    check_interval: int = 1_000
    percent_check: float = 1.0
    train_percent_check: Optional[float] = None
    batch_size: int = 32
    num_beams: int = 1
    early_stopping_patience: int = 10
    early_stopping_metric: str = 'boundary_f1'

@dataclass
class GenerationConfig:
    num_beams: int = 3
    constraint_decoding: bool = False
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None
    no_repeat_ngram_size: Optional[int] = None

@dataclass
class EvaluationConfig:
    split: str = 'test'
    batch_size: int = 32

@dataclass
class CheckpointConfig:
    save_top_k: int = 3
    every_n_steps: int = 1000
    resume_from: Optional[str] = None

@dataclass
class CallbacksConfig:
    sample_generation_interval: int = 1000

@dataclass
class WandbConfig:
    project: str = 's2g'
    entity: Optional[str] = None
    run_name: Optional[str] = None

@dataclass
class HardwareConfig:
    num_workers: int = 0
    persistent_workers: bool = True
    gpu_ids: Optional[List[int]] = None

@dataclass
class S2GConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


def load_config() -> DictConfig:
    yaml_path, remaining = extract_config_flag(sys.argv[1:])
    validate_dotlist(remaining)

    cfg = OmegaConf.structured(S2GConfig)
    if yaml_path:
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))

    if remaining:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(remaining))
        logger.info("Applied %d CLI override(s).", len(remaining))

    return cfg


def load_schema(schema_path: str) -> List[str]:
    if not (p := Path(schema_path)).exists(): raise FileNotFoundError(f"Schema not found: {p}")
    with open(p, 'r', encoding='utf-8') as f: return [ln.strip() for ln in f if ln.strip()]


def load_ent_schema(entity_schema_path: Optional[str]) -> List[str]:
    return load_schema(entity_schema_path) if entity_schema_path else []


def extract_config_flag(cli_args: List[str]) -> Tuple[Optional[str], List[str]]:
    yaml_path, remaining, i = None, [], 0
    while i < len(cli_args):
        if cli_args[i] == "--config":
            if i + 1 >= len(cli_args): raise ValueError("--config flag requires a path argument.")
            yaml_path = cli_args[i + 1]
            i += 2
        elif cli_args[i].startswith("--config="):
            yaml_path = cli_args[i].split("=", 1)[1]
            i += 1
        else:
            remaining.append(cli_args[i])
            i += 1

    return yaml_path, remaining


def validate_dotlist(args: List[str]) -> None:
    for arg in args:
        if arg.startswith("-"): raise ValueError(f"Unrecognised CLI flag: '{arg}'. Overrides must be in dotlist form (no dashes).")
        if "=" not in arg: raise ValueError(f"Malformed override: '{arg}'. Expected 'key=value'.")