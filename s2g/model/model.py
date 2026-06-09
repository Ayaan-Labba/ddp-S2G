"""
S2G Model — seq2seq wrapper for the Pipeline and Joint models.

Manages the lifecycle of the underlying Flan-T5 model:

1. **Initialisation** — loads a Flan-T5 checkpoint, registers the
   correct S2G special tokens for the chosen model variant, and resizes
   the embedding matrix.

2. **Generation** — provides a ``generate`` method that optionally
   activates the FSM-based constraint decoder.

The wrapper deliberately does **not** subclass ``nn.Module``.  It holds
a reference to the HuggingFace model and tokeniser; the ``S2GTrainer``
(a ``Seq2SeqTrainer`` subclass) manages the ``nn.Module`` lifecycle.

Model variants
--------------
``"pipeline"``  Boundary + NER + RE.  Uses ``PIPELINE_TOKENS``.
``"joint"``     Joint + Joint+.        Uses ``JOINT_TOKENS``.

The variant determines which special-token registry is registered with
the tokeniser and passed to the constraint decoder.  The two registries
have different task-delimiter tokens, so they must **not** be swapped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from s2g.linearisation import (
    AnyTokens,
    JOINT_TOKENS,
    PIPELINE_TOKENS,
    add_special_tokens_to_tokenizer,
    get_token_ids,
)

logger = logging.getLogger(__name__)


class S2GModel:
    """Wrapper around a HuggingFace seq2seq model with S2G special tokens.

    Args:
        model_name_or_path: HuggingFace model ID or local checkpoint path.
        model_variant:      ``"pipeline"`` (Boundary/NER/RE) or
                            ``"joint"`` (Joint/Joint+).
    """

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-base",
        model_variant: str = "pipeline",
    ) -> None:
        if model_variant not in ("pipeline", "joint"):
            raise ValueError(
                f"model_variant must be 'pipeline' or 'joint', "
                f"got {model_variant!r}."
            )
        self._variant: str = model_variant
        self._tokens: AnyTokens = (
            PIPELINE_TOKENS if model_variant == "pipeline" else JOINT_TOKENS
        )

        logger.info("Loading tokenizer from %s", model_name_or_path)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name_or_path,
        )

        logger.info("Loading model from %s", model_name_or_path)
        self.model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
        )

        num_added = add_special_tokens_to_tokenizer(
            self.tokenizer, self._tokens, self.model
        )
        logger.info(
            "Variant=%s — added %d special tokens.",
            model_variant, num_added,
        )

        # Cache token IDs for fast lookup downstream.
        self.token_ids: Dict[str, int] = get_token_ids(
            self.tokenizer, self._tokens
        )

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        constraint_decoding: bool = False,
        source_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run autoregressive generation with optional FSM constraint decoding.

        When *constraint_decoding* is ``True``, an FSM-based
        ``LogitsProcessor`` is injected that restricts the decoder
        vocabulary at each step according to the task-specific SEL grammar.
        The task is identified automatically from the task delimiter token
        in each row of *source_ids*.

        Args:
            input_ids:           Encoder input token IDs ``(batch, src_len)``.
            attention_mask:      Encoder attention mask ``(batch, src_len)``.
            constraint_decoding: Activate FSM constraints if ``True``.
            source_ids:          Encoder input IDs for source-copy constraint
                                 and per-instance trie construction.  Must be
                                 provided when *constraint_decoding* is
                                 ``True`` and should equal *input_ids* in
                                 standard usage.
            **kwargs:            Forwarded to ``model.generate()``.

        Returns:
            Generated token IDs ``(batch, tgt_len)``.
        """
        gen_kwargs: Dict[str, Any] = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }

        if constraint_decoding:
            if source_ids is None:
                raise ValueError(
                    "constraint_decoding=True requires source_ids."
                )
            from s2g.model.constraint_decoder import (
                build_constraint_processor,
            )
            processor = build_constraint_processor(
                tokenizer=self.tokenizer,
                source_ids=source_ids,
                tokens=self._tokens,
                num_beams=kwargs.get("num_beams", 1),
            )
            gen_kwargs["logits_processor"] = [processor]

        return self.model.generate(**gen_kwargs)

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save model, tokeniser, and variant metadata to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "model_variant.txt").write_text(self._variant, encoding="utf-8")
        logger.info("Saved %s model to %s", self._variant, path)

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        model_variant: Optional[str] = None,
    ) -> "S2GModel":
        """Load a previously saved S2G model.

        Special tokens are already in the saved tokeniser, so
        ``add_special_tokens_to_tokenizer`` is called with ``model=None``
        to refresh the token ID cache without resizing embeddings.

        Args:
            path:          Directory produced by :meth:`save_pretrained`.
            model_variant: ``"pipeline"`` or ``"joint"``.  If ``None``,
                           read from ``model_variant.txt`` in *path*.
        """
        path = Path(path)
        if model_variant is None:
            variant_file = path / "model_variant.txt"
            if not variant_file.exists():
                raise FileNotFoundError(
                    f"model_variant.txt not found in {path}.  "
                    "Pass model_variant explicitly."
                )
            model_variant = variant_file.read_text(encoding="utf-8").strip()

        instance = cls.__new__(cls)
        instance._variant = model_variant
        instance._tokens = (
            PIPELINE_TOKENS if model_variant == "pipeline" else JOINT_TOKENS
        )

        path_str = str(path)
        instance.tokenizer = AutoTokenizer.from_pretrained(path_str)
        instance.model     = AutoModelForSeq2SeqLM.from_pretrained(path_str)

        # Tokens already in the saved vocab — no resize needed.
        add_special_tokens_to_tokenizer(instance.tokenizer, instance._tokens)
        instance.token_ids = get_token_ids(instance.tokenizer, instance._tokens)

        logger.info("Loaded %s model from %s", model_variant, path)
        return instance