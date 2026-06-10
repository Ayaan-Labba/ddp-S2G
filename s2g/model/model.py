"""
S2G Model wrapper for Pipeline and Joint models.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, 
    PreTrainedTokenizerBase, LogitsProcessorList
)

from s2g.linearisation import (
    AnyTokens, JOINT_TOKENS, PIPELINE_TOKENS, 
    add_special_tokens_to_tokenizer, get_token_ids
)

logger = logging.getLogger(__name__)


class S2GModel:
    def __init__(self, model_name_or_path: str = "google/flan-t5-base", model_variant: str = "pipeline") -> None:
        if model_variant not in {"pipeline", "joint"}:
            raise ValueError(f"model_variant must be 'pipeline' or 'joint', got {model_variant!r}.")
        
        self._variant = model_variant
        self._tokens: AnyTokens = PIPELINE_TOKENS if model_variant == "pipeline" else JOINT_TOKENS

        logger.info("Loading tokenizer and model from %s", model_name_or_path)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Load exactly as specified without auto-expanding to fp32 overhead
        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype="auto")

        num_added = add_special_tokens_to_tokenizer(self.tokenizer, self._tokens, self.model)
        logger.info("Variant=%s — added %d special tokens.", model_variant, num_added)
        self.token_ids: Dict[str, int] = get_token_ids(self.tokenizer, self._tokens)

    def generate(
            self, 
            input_ids: torch.Tensor, 
            attention_mask: torch.Tensor, 
            constraint_decoding: bool = False, 
            source_ids: Optional[torch.Tensor] = None, 
            **kwargs: Any
        ) -> torch.Tensor:
        
        if constraint_decoding:
            source_ids = source_ids if source_ids is not None else input_ids
            from s2g.model.constraint_decoder import build_constraint_processor
            
            processor = build_constraint_processor(
                self.tokenizer, 
                source_ids, 
                self._tokens, 
                kwargs.get("num_beams", 1)
            )
            
            # Use LogitsProcessorList to append rather than destructively overwrite
            if "logits_processor" not in kwargs:
                kwargs["logits_processor"] = LogitsProcessorList()
            elif not isinstance(kwargs["logits_processor"], LogitsProcessorList):
                kwargs["logits_processor"] = LogitsProcessorList(kwargs["logits_processor"])
                
            kwargs["logits_processor"].append(processor)

        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def save_pretrained(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "model_variant.txt").write_text(self._variant, encoding="utf-8")
        logger.info("Saved %s model to %s", self._variant, path)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], model_variant: Optional[str] = None) -> "S2GModel":
        path = Path(path)
        model_variant = model_variant or (path / "model_variant.txt").read_text(encoding="utf-8").strip()

        instance = cls.__new__(cls)
        instance._variant = model_variant
        instance._tokens = PIPELINE_TOKENS if model_variant == "pipeline" else JOINT_TOKENS
        
        instance.tokenizer = AutoTokenizer.from_pretrained(str(path))
        instance.model = AutoModelForSeq2SeqLM.from_pretrained(str(path), torch_dtype="auto")

        add_special_tokens_to_tokenizer(instance.tokenizer, instance._tokens)
        instance.token_ids = get_token_ids(instance.tokenizer, instance._tokens)
        
        logger.info("Loaded %s model from %s", model_variant, path)
        return instance