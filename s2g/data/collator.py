"""
S2G Data Collator — multi-task SSI construction with budget and Bernoulli modes.

Produces tokenised model batches for multi-task fine-tuning (Pipeline or
Joint model).  Two SSI sampling modes are supported:

Budget mode  (Experiment 1 — benchmark fine-tuning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All gold-positive types are always included.  The remaining budget
(``max_ent_types_in_prompt`` or ``max_rel_types_in_prompt``) is filled
with uniformly sampled negatives.  Step-independent.

Bernoulli mode  (Experiment 3 — REBEL pre-training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each gold-positive type is independently included with probability
``positive_rate(t)`` (so some positives may be absent).  Each negative
type is independently included with probability ``negative_rate(t)``,
then the negative set is randomly trimmed to at most ``k(t)`` items.
All three quantities follow linear schedules over ``max_steps`` steps:

    value(t) = start + (t / max_steps) * (end - start)

``current_step`` must be updated after every optimizer step via
``StepTrackingCallback`` (or equivalent).

Multi-task key names
~~~~~~~~~~~~~~~~~~~~
Pipeline: ``boundary_*``, ``ner_*``, ``re_*``
Joint:    ``joint_*``, ``joint_plus_*``

Gold augmented text
~~~~~~~~~~~~~~~~~~~
The NER and RE encoder inputs use gold entity spans and types (teacher
forcing).  At evaluation time the S2GTrainer overrides evaluate() to
build downstream inputs from upstream predictions instead; no separate
eval collator is needed.

Required config keys (both modes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``model_variant``            "pipeline" | "joint"
``max_source_length``        int
``max_target_length``        int
``max_ent_types_in_prompt``  Optional[int]  (hard cap, None = full schema)
``max_rel_types_in_prompt``  Optional[int]
``random_prompt``            bool
``random_sel``               bool
``mode``                     "budget" | "bernoulli"  (default "budget")

Additional keys for Bernoulli mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``max_steps``            int
``positive_rate_start``  float
``positive_rate_end``    float
``negative_rate_start``  float
``negative_rate_end``    float
``negative_max_start``   int   — k(0)
``negative_max_end``     int   — k(T)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from transformers import PreTrainedTokenizerBase

from s2g.linearisation import (
    JOINT_TOKENS,
    PIPELINE_TOKENS,
    AnyTokens,
    build_boundary_encoder_input,
    build_joint_encoder_input,
    build_joint_plus_encoder_input,
    build_ner_encoder_input,
    build_re_encoder_input,
    build_sel,
    organize_by_entity,
)

_PIPELINE_TASKS: Tuple[str, ...] = ("boundary", "ner", "re")
_JOINT_TASKS:    Tuple[str, ...] = ("joint", "joint_plus")


class S2GCollator:
    """Data collator for multi-task fine-tuning of Pipeline and Joint models.

    Args:
        tokenizer:     HuggingFace tokeniser with S2G special tokens
                       already registered.
        entity_schema: Complete entity-type list for the dataset.
        rel_schema:    Complete relation-type list for the dataset.
        config:        Configuration dict — see module docstring.
    """

    def __init__(
        self,
        tokenizer:     PreTrainedTokenizerBase,
        entity_schema: List[str],
        rel_schema:    List[str],
        config:        Dict[str, Any],
    ) -> None:
        variant = config.get("model_variant")
        if variant not in ("pipeline", "joint"):
            raise ValueError(
                f"model_variant must be 'pipeline' or 'joint', got {variant!r}."
            )
        mode = config.get("mode", "budget")
        if mode not in ("budget", "bernoulli"):
            raise ValueError(
                f"mode must be 'budget' or 'bernoulli', got {mode!r}."
            )

        self._tokenizer:          PreTrainedTokenizerBase = tokenizer
        self._entity_schema:      List[str]               = list(entity_schema)
        self._entity_schema_set:  Set[str]                = set(entity_schema)
        self._rel_schema:         List[str]               = list(rel_schema)
        self._rel_schema_set:     Set[str]                = set(rel_schema)
        self._cfg:     Dict[str, Any] = config
        self._variant: str            = variant
        self._mode:    str            = mode

        self._tok: AnyTokens = (
            PIPELINE_TOKENS if variant == "pipeline" else JOINT_TOKENS
        )

        self._random_prompt: bool = config.get("random_prompt", False)
        self._random_sel:    bool = config.get("random_sel",    False)

        # Step counter — updated by StepTrackingCallback in Bernoulli mode.
        self._step: int = 0

    # ------------------------------------------------------------------ #
    #  Step tracking                                                       #
    # ------------------------------------------------------------------ #

    @property
    def current_step(self) -> int:
        return self._step

    @current_step.setter
    def current_step(self, value: int) -> None:
        """Update the global step.  Used by StepTrackingCallback."""
        self._step = value

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate *batch* into a multi-task dict of tokenised tensors."""
        if self._variant == "pipeline":
            return self._collate_pipeline(batch)
        return self._collate_joint(batch)

    # ------------------------------------------------------------------ #
    #  Variant-level collation                                             #
    # ------------------------------------------------------------------ #

    def _collate_pipeline(self, batch: List[Dict]) -> Dict[str, Any]:
        b_enc, b_dec = [], []
        n_enc, n_dec = [], []
        r_enc, r_dec = [], []

        for inst in batch:
            bi, bo = self._prepare_boundary(inst)
            ni, no = self._prepare_ner(inst)
            ri, ro = self._prepare_re(inst)
            b_enc.append(bi); b_dec.append(bo)
            n_enc.append(ni); n_dec.append(no)
            r_enc.append(ri); r_dec.append(ro)

        result: Dict[str, Any] = {}
        result.update(self._tokenize_task("boundary", b_enc, b_dec))
        result.update(self._tokenize_task("ner",      n_enc, n_dec))
        result.update(self._tokenize_task("re",       r_enc, r_dec))
        return result

    def _collate_joint(self, batch: List[Dict]) -> Dict[str, Any]:
        j_enc,  j_dec  = [], []
        jp_enc, jp_dec = [], []

        for inst in batch:
            ji, jo   = self._prepare_joint(inst)
            jpi, jpo = self._prepare_joint_plus(inst)
            j_enc.append(ji);   j_dec.append(jo)
            jp_enc.append(jpi); jp_dec.append(jpo)

        result: Dict[str, Any] = {}
        result.update(self._tokenize_task("joint",      j_enc,  j_dec))
        result.update(self._tokenize_task("joint_plus", jp_enc, jp_dec))
        return result

    # ------------------------------------------------------------------ #
    #  Per-task instance preparation                                       #
    # ------------------------------------------------------------------ #

    def _prepare_boundary(self, inst: Dict) -> Tuple[str, str]:
        entity_blocks = organize_by_entity(inst["entities"], inst["relations"])
        enc = build_boundary_encoder_input(inst["text"], tok=self._tok)
        dec = build_sel(entity_blocks, "boundary", self._tok,
                        random_sel=self._random_sel)
        return enc, dec

    def _prepare_ner(self, inst: Dict) -> Tuple[str, str]:
        pos_ent, neg_ent = self._sample_types(
            inst["entity_types"],
            self._entity_schema,
            self._entity_schema_set,
            self._cfg.get("max_ent_types_in_prompt"),
        )
        entity_spans = [
            (int(e["offset"][0]), int(e["offset"][1]))
            for e in inst["entities"]
        ]
        entity_blocks = organize_by_entity(inst["entities"], inst["relations"])
        enc = build_ner_encoder_input(
            pos_ent + neg_ent, inst["tokens"], entity_spans,
            random_order=self._random_prompt, tok=self._tok,
        )
        dec = build_sel(entity_blocks, "ner", self._tok,
                        rejected_ent_types=neg_ent,
                        random_sel=self._random_sel)
        return enc, dec

    def _prepare_re(self, inst: Dict) -> Tuple[str, str]:
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"],
            self._rel_schema,
            self._rel_schema_set,
            self._cfg.get("max_rel_types_in_prompt"),
        )
        entity_data = [
            (int(e["offset"][0]), int(e["offset"][1]), e["type"])
            for e in inst["entities"]
        ]
        entity_blocks = organize_by_entity(inst["entities"], inst["relations"])
        enc = build_re_encoder_input(
            pos_rel + neg_rel, inst["tokens"], entity_data,
            random_order=self._random_prompt, tok=self._tok,
        )
        dec = build_sel(entity_blocks, "re", self._tok,
                        rejected_rel_types=neg_rel,
                        random_sel=self._random_sel)
        return enc, dec

    def _prepare_joint(self, inst: Dict) -> Tuple[str, str]:
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"],
            self._rel_schema,
            self._rel_schema_set,
            self._cfg.get("max_rel_types_in_prompt"),
        )
        entity_blocks = organize_by_entity(inst["entities"], inst["relations"])
        enc = build_joint_encoder_input(
            pos_rel + neg_rel, inst["text"],
            random_order=self._random_prompt, tok=self._tok,
        )
        dec = build_sel(entity_blocks, "joint", self._tok,
                        rejected_rel_types=neg_rel,
                        random_sel=self._random_sel)
        return enc, dec

    def _prepare_joint_plus(self, inst: Dict) -> Tuple[str, str]:
        pos_ent, neg_ent = self._sample_types(
            inst["entity_types"],
            self._entity_schema,
            self._entity_schema_set,
            self._cfg.get("max_ent_types_in_prompt"),
        )
        pos_rel, neg_rel = self._sample_types(
            inst["rel_types"],
            self._rel_schema,
            self._rel_schema_set,
            self._cfg.get("max_rel_types_in_prompt"),
        )
        entity_blocks = organize_by_entity(inst["entities"], inst["relations"])
        enc = build_joint_plus_encoder_input(
            pos_ent + neg_ent, pos_rel + neg_rel, inst["text"],
            random_order=self._random_prompt, tok=self._tok,
        )
        dec = build_sel(entity_blocks, "joint+", self._tok,
                        rejected_ent_types=neg_ent,
                        rejected_rel_types=neg_rel,
                        random_sel=self._random_sel)
        return enc, dec

    # ------------------------------------------------------------------ #
    #  SSI sampling — mode dispatch                                        #
    # ------------------------------------------------------------------ #

    def _sample_types(
        self,
        instance_types: List[str],
        schema:         List[str],
        schema_set:     Set[str],
        max_types:      Optional[int],
    ) -> Tuple[List[str], List[str]]:
        """Dispatch to budget or Bernoulli sampling based on ``self._mode``."""
        if self._mode == "budget":
            return self._sample_budget(instance_types, schema, schema_set, max_types)
        return self._sample_bernoulli(instance_types, schema, schema_set, max_types)

    def _sample_budget(
        self,
        instance_types: List[str],
        schema:         List[str],
        schema_set:     Set[str],
        max_types:      Optional[int],
    ) -> Tuple[List[str], List[str]]:
        """Budget-mode sampling: all positives, uniformly sampled negatives.

        All gold-positive types are always included.  Negatives are drawn
        uniformly without replacement to fill the remaining budget
        ``max_types - len(positives)``.  Positives are never truncated:
        when the positive set alone meets or exceeds *max_types*, no
        negatives are added.

        Args:
            instance_types: Gold type strings for this instance.
            schema:         Ordered full schema list.
            schema_set:     Set of full schema for O(1) exclusion test.
            max_types:      Maximum total types in the SSI (None = no cap).

        Returns:
            ``(positives, negatives)``
        """
        instance_set  = set(instance_types)
        negative_pool = [t for t in schema if t not in instance_set]

        if max_types is None:
            sampled_neg = negative_pool
        else:
            neg_budget = max(0, max_types - len(instance_types))
            n_neg      = min(neg_budget, len(negative_pool))
            sampled_neg = (
                random.sample(negative_pool, n_neg) if n_neg > 0 else []
            )
        return list(instance_types), sampled_neg

    def _sample_bernoulli(
        self,
        instance_types: List[str],
        schema:         List[str],
        schema_set:     Set[str],
        max_types:      Optional[int],
    ) -> Tuple[List[str], List[str]]:
        """Bernoulli-mode sampling with step-dependent rates and cap.

        Rates and negative cap are computed from the current step:

            pos_rate(t) = pos_start + frac * (pos_end - pos_start)
            neg_rate(t) = neg_start + frac * (neg_end - neg_start)
            k(t)        = round(k_start + frac * (k_end - k_start))

        where ``frac = min(step, max_steps) / max_steps``.

        Each gold-positive type is independently included with
        ``pos_rate(t)``; each negative type is independently included
        with ``neg_rate(t)``.  The negative set is then randomly trimmed
        to ``k(t)`` (and optionally further capped by *max_types*).

        Args:
            instance_types: Gold type strings for this instance.
            schema:         Ordered full schema list.
            schema_set:     Set for O(1) exclusion test.
            max_types:      Hard cap on total types (None = no cap).

        Returns:
            ``(included_positives, included_negatives)``
        """
        pos_rate, neg_rate, k = self._schedule_values()
        instance_set = set(instance_types)

        # Bernoulli sample positives.
        included_pos = [t for t in instance_types if random.random() < pos_rate]

        # Bernoulli sample negatives then trim to k(t).
        neg_pool      = [t for t in schema if t not in instance_set]
        candidate_neg = [t for t in neg_pool if random.random() < neg_rate]
        if len(candidate_neg) > k:
            candidate_neg = random.sample(candidate_neg, k)

        # Apply hard cap.
        if max_types is not None:
            remaining = max(0, max_types - len(included_pos))
            if len(candidate_neg) > remaining:
                candidate_neg = random.sample(candidate_neg, remaining)

        return included_pos, candidate_neg

    def _schedule_values(self) -> Tuple[float, float, int]:
        """Compute ``(pos_rate, neg_rate, k)`` at the current training step."""
        T    = max(int(self._cfg.get("max_steps", 1)), 1)
        frac = min(self._step, T) / T

        def _lerp(start: float, end: float) -> float:
            return start + frac * (end - start)

        pos_rate = _lerp(
            self._cfg.get("positive_rate_start", 0.9),
            self._cfg.get("positive_rate_end",   0.9),
        )
        neg_rate = _lerp(
            self._cfg.get("negative_rate_start", 0.1),
            self._cfg.get("negative_rate_end",   0.1),
        )
        k = round(_lerp(
            float(self._cfg.get("negative_max_start", 1)),
            float(self._cfg.get("negative_max_end",   20)),
        ))
        return pos_rate, neg_rate, k

    # ------------------------------------------------------------------ #
    #  Tokenisation                                                        #
    # ------------------------------------------------------------------ #

    def _tokenize_task(
        self,
        task_key:        str,
        encoder_inputs:  List[str],
        decoder_targets: List[str],
    ) -> Dict[str, Any]:
        """Tokenise and pad one task's inputs and targets.

        Returns a dict with three keys:
        ``{task_key}_{input_ids|attention_mask|labels}``.
        Padding tokens in ``labels`` are replaced with -100.
        """
        max_src = self._cfg["max_source_length"]
        max_tgt = self._cfg["max_target_length"]

        model_inputs = self._tokenizer(
            encoder_inputs,
            max_length=max_src,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

        with self._tokenizer.as_target_tokenizer():
            label_enc = self._tokenizer(
                decoder_targets,
                max_length=max_tgt,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )

        label_ids = label_enc["input_ids"].clone()
        label_ids[label_ids == self._tokenizer.pad_token_id] = -100

        return {
            f"{task_key}_input_ids":      model_inputs["input_ids"],
            f"{task_key}_attention_mask": model_inputs["attention_mask"],
            f"{task_key}_labels":         label_ids,
        }