"""
Special token registry for the S2G model.

Two separate, immutable registries are defined — one per model variant —
because the Pipeline and Joint models use distinct tokenizers with distinct
special token inventories.  Sharing a tokenizer is incorrect: the two
models carry different task delimiter tokens, and the constraint decoder
relies on the delimiter token ID to select the correct FSM at inference
time.

Pipeline model (9 tokens)
--------------------------
    Task delimiters  (encoder-only):       <bound>, <ner>, <re>
    Structural       (encoder + decoder):  <type>, <rel>, <ent>, </ent>
    Structural       (decoder-only):       <tail>, <null>

Joint model (8 tokens)
-----------------------
    Task delimiters  (encoder-only):       <joint>, <joint+>
    Structural       (encoder + decoder):  <type>, <rel>, <ent>, </ent>
    Structural       (decoder-only):       <tail>, <null>

Design notes
------------
- Both registries are frozen dataclasses so token strings are defined in
  exactly one place.  Every other module imports PIPELINE_TOKENS or
  JOINT_TOKENS rather than hard-coding strings.
- ``as_dict()`` maps field names to token strings and drives both
  ``add_special_tokens_to_tokenizer`` and ``get_token_ids``, so adding a
  new token only requires updating the dataclass and its ``as_dict``.
- The ``task_delimiters`` property returns the ordered list of delimiter
  tokens for the model.  The constraint decoder walks the encoder input
  until it finds one of these to identify the active task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from transformers import PreTrainedModel, PreTrainedTokenizerBase


# ===================================================================== #
#                       TOKEN REGISTRY CLASSES                          #
# ===================================================================== #


@dataclass(frozen=True)
class PipelineTokens:
    """Immutable token registry for the Pipeline model.

    The Pipeline model is trained jointly on three tasks: Boundary, NER,
    and RE.  Each task has its own delimiter token that immediately follows
    the SSI prefix (or opens the sequence for Boundary, which has no SSI).

    Attributes
    ----------
    bound:      Task delimiter for Boundary.  Encoder-only.
    ner:        Task delimiter for NER.  Encoder-only.
    re:         Task delimiter for RE.  Encoder-only.
    type_:      Introduces one entity type in the NER SSI; annotates the
                entity type inline in the RE augmented text; emits the
                entity type label in NER SEL output.  Encoder + decoder.
    rel:        Introduces one relation type in the RE SSI; opens a
                relation block in the RE SEL output.  Encoder + decoder.
    ent_start:  Marks the opening of an entity span in the NER/RE
                augmented encoder text; opens an entity block in SEL.
                Encoder + decoder.
    ent_end:    Closes the entity span in augmented encoder text; closes
                the entity block in SEL.  Encoder + decoder.
    tail:       Introduces the tail-entity span within a relation block
                in SEL.  Decoder-only.
    null:       Opens the rejection block in SEL.  Decoder-only.
    """

    # Task delimiter tokens (encoder-only)
    bound: str = "<bound>"
    ner:   str = "<ner>"
    re:    str = "<re>"

    # Structural tokens (encoder + decoder)
    type_:     str = "<type>"
    rel:       str = "<rel>"
    ent_start: str = "<ent>"
    ent_end:   str = "</ent>"

    # Structural tokens (decoder-only)
    tail: str = "<tail>"
    null: str = "<null>"

    # ----- Derived collections -----

    @property
    def task_delimiters(self) -> List[str]:
        """Task delimiter tokens, in task order (Boundary, NER, RE)."""
        return [self.bound, self.ner, self.re]

    @property
    def all_tokens(self) -> List[str]:
        """All special tokens in a stable, deterministic order."""
        return [
            self.bound, self.ner, self.re,
            self.type_, self.rel,
            self.ent_start, self.ent_end,
            self.tail, self.null,
        ]

    def as_dict(self) -> Dict[str, str]:
        """Map from field name to token string.

        Used by :func:`add_special_tokens_to_tokenizer` and
        :func:`get_token_ids` to iterate over all tokens without
        hard-coding field names at the call sites.
        """
        return {
            "bound":     self.bound,
            "ner":       self.ner,
            "re":        self.re,
            "type_":     self.type_,
            "rel":       self.rel,
            "ent_start": self.ent_start,
            "ent_end":   self.ent_end,
            "tail":      self.tail,
            "null":      self.null,
        }


@dataclass(frozen=True)
class JointTokens:
    """Immutable token registry for the Joint model.

    The Joint model is trained jointly on two tasks: Joint and Joint+.
    Both tasks take raw text as input; the SSI lists relation types (Joint)
    or entity types followed by relation types (Joint+).

    Attributes
    ----------
    joint:      Task delimiter for Joint.  Encoder-only.
    joint_plus: Task delimiter for Joint+.  Encoder-only.
    type_:      Introduces one entity type in the Joint+ SSI; emits the
                entity type label in Joint+ SEL output.  Encoder + decoder.
    rel:        Introduces one relation type in the Joint/Joint+ SSI;
                opens a relation block in SEL.  Encoder + decoder.
    ent_start:  Opens an entity block in SEL.  Decoder-only in the Joint
                model (no entity-augmented encoder text).
    ent_end:    Closes an entity block in SEL.  Decoder-only.
    tail:       Introduces the tail-entity span within a relation block
                in SEL.  Decoder-only.
    null:       Opens the rejection block in SEL.  Decoder-only.
    """

    # Task delimiter tokens (encoder-only)
    joint:      str = "<joint>"
    joint_plus: str = "<joint+>"

    # Structural tokens (encoder + decoder)
    type_:     str = "<type>"
    rel:       str = "<rel>"
    ent_start: str = "<ent>"
    ent_end:   str = "</ent>"

    # Structural tokens (decoder-only)
    tail: str = "<tail>"
    null: str = "<null>"

    # ----- Derived collections -----

    @property
    def task_delimiters(self) -> List[str]:
        """Task delimiter tokens, in task order (Joint, Joint+)."""
        return [self.joint, self.joint_plus]

    @property
    def all_tokens(self) -> List[str]:
        """All special tokens in a stable, deterministic order."""
        return [
            self.joint, self.joint_plus,
            self.type_, self.rel,
            self.ent_start, self.ent_end,
            self.tail, self.null,
        ]

    def as_dict(self) -> Dict[str, str]:
        """Map from field name to token string."""
        return {
            "joint":      self.joint,
            "joint_plus": self.joint_plus,
            "type_":      self.type_,
            "rel":        self.rel,
            "ent_start":  self.ent_start,
            "ent_end":    self.ent_end,
            "tail":       self.tail,
            "null":       self.null,
        }


# ===================================================================== #
#                       MODULE-LEVEL SINGLETONS                         #
# ===================================================================== #

PIPELINE_TOKENS: PipelineTokens = PipelineTokens()
JOINT_TOKENS:    JointTokens    = JointTokens()

# Type alias used by the utility functions below.
AnyTokens = Union[PipelineTokens, JointTokens]


# ===================================================================== #
#                         UTILITY FUNCTIONS                             #
# ===================================================================== #


def add_special_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    tokens: AnyTokens,
    model: Optional[PreTrainedModel] = None,
) -> int:
    """Register S2G special tokens with *tokenizer* and optionally resize *model*.

    Calls ``tokenizer.add_special_tokens`` with all tokens from *tokens*
    and, when *model* is provided, resizes its embedding matrix to
    accommodate the new vocabulary entries.

    Args:
        tokenizer: A HuggingFace tokeniser instance (not yet containing
                   the S2G tokens).
        tokens:    Registry to use — ``PIPELINE_TOKENS`` or
                   ``JOINT_TOKENS``.  Must match the model variant.
        model:     If provided, ``model.resize_token_embeddings`` is
                   called after token registration.

    Returns:
        The number of tokens actually added (0 if they were already
        present, e.g. when loading from a saved checkpoint).
    """
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": tokens.all_tokens}
    )

    if model is not None and num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    return num_added


def get_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    tokens: AnyTokens,
) -> Dict[str, int]:
    """Return a mapping from field name to integer token ID.

    Must be called *after* :func:`add_special_tokens_to_tokenizer`.

    Args:
        tokenizer: HuggingFace tokeniser with S2G tokens already
                   registered.
        tokens:    Registry used during registration — ``PIPELINE_TOKENS``
                   or ``JOINT_TOKENS``.

    Returns:
        Dict mapping each field name in ``tokens.as_dict()`` to its
        integer token ID, e.g.
        ``{"bound": 32100, "ner": 32101, ..., "null": 32108}``.
    """
    return {
        name: tokenizer.convert_tokens_to_ids(token_str)
        for name, token_str in tokens.as_dict().items()
    }