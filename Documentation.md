# S2G (Sentence-to-Graph) Codebase Documentation

This document provides a comprehensive, exhaustive technical reference for the **S2G (Sentence-to-Graph)** codebase (`sentinel` branch). It is designed to furnish complete context for developers working on or extending this project.

---

## 1. High-Level Architecture Overview

**S2G** frames joint entity and relation extraction (IE) as a sequence-to-sequence (Text-to-Text) translation problem built upon **Flan-T5** (Base/Large).

By default, S2G uses natural language instruction prompts for the encoder input and generates a linearised token graph using rolling sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, ...) for entities along with dedicated vocabulary special tokens (`<e_type>`, `<r_type>`, `<nr_type>`, `<tail>`, `<null>`).

```
                            ┌──────────────────────────────────────────────────────────┐
                            │                  Encoder Input (Prompt)                  │
                            │  Extract all entities of type [...] and find relations   │
                            │  of type [...] among the extracted entities.             │
                            │                                                          │
                            │  Text: <Source Text>                                     │
                            └────────────────────────────┬─────────────────────────────┘
                                                         │
                                                         ▼
                                             ┌───────────────────────┐
                                             │    Flan-T5 Encoder    │
                                             └───────────┬───────────┘
                                                         │
                                                         ▼
                                             ┌───────────────────────┐
                                             │    Flan-T5 Decoder    │
                                             └───────────┬───────────┘
                                                         │
                                                         ▼
                            ┌──────────────────────────────────────────────────────────┐
                            │             Decoder Output (Nested Graph)                │
                            │  <extra_id_0> e1 <e_type> type <r_type> rel <tail> e2    │
                            │  <extra_id_1> e2 <e_type> type                           │
                            └────────────────────────────┬─────────────────────────────┘
                                                         │
                                                         ▼
                                             ┌───────────────────────┐
                                             │   Deterministic       │
                                             │   Graph Parser &      │
                                             │   Evaluator           │
                                             └───────────┬───────────┘
```

### Core Concepts

1. **Prompt**: The encoder prompt formats the input text with task instructions and target entity/relation schema types, e.g. `"Extract all entities of type [...] and find relations of type [...] among the extracted entities. Text: ..."`. Setting `prompt.type: false` disables the instruction and feeds the raw source text instead (ablation only).
2. **Graph (Nested Sentinel Scheme)**: Linearised target representation where each entity mention is co-located with its outgoing relations.
   - Introduced by rolling sentinels (`<extra_id_0>`, `<extra_id_1>`, ...).
   - The first relation is introduced by `<r_type> rel_type <tail> tail_text` (in `re`, also `<e_type> tail_type`).
   - Subsequent relations for the same head entity are introduced by `<nr_type> rel_type <tail> tail_text`.
   - Entities with **no outgoing relations** simply omit relation tokens (ending directly after entity mention/type).
3. **Vocabulary Special Tokens**: Explicit special tokens `<e_type>`, `<r_type>`, `<nr_type>`, `<tail>`, and `<null>` are added to the tokenizer vocabulary. Sentinel tokens (`<extra_id_0>` ... `<extra_id_99>`) are natively present in T5/Flan-T5.
4. **Supported Model Variants**:
   * **`joint`**: Joint entity recognition and relation extraction. All entity mentions get their own block (`<extra_id_i> head [<e_type> type]`). Entities without outgoing relations emit no relation tokens. Tail entities are referenced by surface text (no tail types).
   * **`boundary_joint`**: Joint entity span boundary extraction (no entity types) and relation extraction. All entity mentions get their own block (`<extra_id_i> head`). Entities without outgoing relations emit no relation tokens.
   * **`re`**: Relation extraction with the entity *type* schema supplied in the prompt (entity spans are still predicted, not given). **Only entities that act as a head in at least one relation get their own block** (`<extra_id_i> head <e_type> head_type <r_type> rel <tail> tail <e_type> tail_type`). Non-participating entities and tail-only entities are omitted as head blocks.
   * **`boundary_re`**: Relation extraction between entity spans without entity types. **Only entities that act as a head in at least one relation get their own block** (`<extra_id_i> head <r_type> rel <tail> tail`). Non-participating entities and tail-only entities are omitted as head blocks.
5. **Rejection & Null Blocks**: Optional negative schema type markers (`<null> type`) included in Graph outputs to force explicit model rejection of absent entity or relation types.

---

### Running Example & Variant Specifications

The following running example demonstrates the exact encoder input prompts and decoder nested graph outputs across all supported variants:

* **Text**: `Barack Obama was born in Honolulu and served as the president of the United States`
* **Entities**:
  * `Barack Obama` (`person`) $\rightarrow$ Head of 2 relations (`place of birth` $\to$ `Honolulu`, `president of` $\to$ `United States`)
  * `Honolulu` (`city`) $\rightarrow$ Head of 1 relation (`located in` $\to$ `United States`), Tail of 1 relation
  * `United States` (`country`) $\rightarrow$ Tail only (no outgoing relations)
* **Relations**: `(Barack Obama, place of birth, Honolulu)`, `(Barack Obama, president of, United States)`, `(Honolulu, located in, United States)`
* **Schema**:
  * **Entity Types**: `person`, `city`, `country`, `organization`, `artifact`
  * **Relation Types**: `place of birth`, `president of`, `located in`, `founded`, `killed`

#### 1. `joint`
* **Task**: Joint entity span and type extraction + relation extraction across all entities.
* **Encoder Input (Natural Prompt)**:
  ```text
  Extract all entities of type [artifact, city, country, organization, person] and find relations of type [founded, killed, located in, place of birth, president of] among the extracted entities. Text: Barack Obama was born in Honolulu and served as the president of the United States
  ```
* **Decoder Output (Nested Graph)**:
  ```text
  <extra_id_0> Barack Obama <e_type> person <r_type> place of birth <tail> Honolulu <nr_type> president of <tail> United States <extra_id_1> Honolulu <e_type> city <r_type> located in <tail> United States <extra_id_2> United States <e_type> country
  ```

#### 2. `boundary_joint`
* **Task**: Joint entity span boundary extraction (without entity types) + relation extraction across all entities.
* **Encoder Input (Natural Prompt)**:
  ```text
  Extract all entities and find relations of type [founded, killed, located in, place of birth, president of] among the extracted entities. Text: Barack Obama was born in Honolulu and served as the president of the United States
  ```
* **Decoder Output (Nested Graph)**:
  ```text
  <extra_id_0> Barack Obama <r_type> place of birth <tail> Honolulu <nr_type> president of <tail> United States <extra_id_1> Honolulu <r_type> located in <tail> United States <extra_id_2> United States
  ```

#### 3. `re`
* **Task**: Relation extraction with entity types provided for head and tail entities. Non-head entities (e.g. `United States`) are omitted as head blocks.
* **Encoder Input (Natural Prompt)**:
  ```text
  Extract all relations of type [founded, killed, located in, place of birth, president of] among the entities of type [artifact, city, country, organization, person] in the given text. Text: Barack Obama was born in Honolulu and served as the president of the United States
  ```
* **Decoder Output (Nested Graph)**:
  ```text
  <extra_id_0> Barack Obama <e_type> person <r_type> place of birth <tail> Honolulu <e_type> city <nr_type> president of <tail> United States <e_type> country <extra_id_1> Honolulu <e_type> city <r_type> located in <tail> United States <e_type> country
  ```

#### 4. `boundary_re`
* **Task**: Relation extraction between entity mentions without entity types. Non-head entities (e.g. `United States`) are omitted as head blocks.
* **Encoder Input (Natural Prompt)**:
  ```text
  Extract all relations of type [founded, killed, located in, place of birth, president of] among the entities in the given text. Text: Barack Obama was born in Honolulu and served as the president of the United States
  ```
* **Decoder Output (Nested Graph)**:
  ```text
  <extra_id_0> Barack Obama <r_type> place of birth <tail> Honolulu <nr_type> president of <tail> United States <extra_id_1> Honolulu <r_type> located in <tail> United States
  ```

---

## 2. Module: `s2g.linearisation`

The `linearisation` package defines the special token mapping, graph linearisation into Graph strings, Graph parsing logic back to structured graph blocks, and encoder prompt building.

---

### 2.1. `s2g/linearisation/special_tokens.py`

#### Purpose
Defines the special token registry, vocabulary expansion, sentinel token string generation, and embedding initialization ("warm starting") for special tokens.

#### Constants & Token Map
* `ALL_TOKEN_NAMES`: `['e_type', 'r_type', 'nr_type', 'tail', 'null']`
* `VALID_VARIANTS`: `{'re', 'boundary_re', 'boundary_joint', 'joint'}`

#### Special Token Mapping (`S2GTokens.token_strs`)
| Token Key | Vocabulary Special Token String | Semantic Role |
|---|---|---|
| `'e_type'` | `<e_type>` | Entity type token in Graph |
| `'r_type'` | `<r_type>` | Primary relation type token in Graph |
| `'nr_type'` | `<nr_type>` | Nested relation type token (for same head entity) in Graph |
| `'tail'` | `<tail>` | Static tail token preceding tail entity text |
| `'null'` | `<null>` | Standalone negative rejection marker token in Graph |

#### Key Classes & Functions

##### `S2GTokens(variant: str, use_rejection: bool = False)`
* **`base_tok_map`**: Maps active tokens per variant (`re`, `boundary_re`, `boundary_joint`, `joint`).
* **`self.active_tokens`**: Active token set for configured variant. Adds `'null'` if `use_rejection=True`.
* **`self.all_tokens`**: List of vocabulary special token strings.
* **`self.sentinel_token(idx)`**: Returns `<extra_id_{idx}>` for rolling entity demarcation.

##### `add_special_tokens_to_tokenizer(tokenizer, tokens: S2GTokens, model=None, warm_start: bool = True) -> int`
* Adds `tokens.all_tokens` to HuggingFace tokenizer via `add_special_tokens({'additional_special_tokens': ...})`.
* If `model` is provided and tokens were added:
  * Sets `model.config.tie_word_embeddings = False`.
  * Resizes model token embeddings via `model.resize_token_embeddings(len(tokenizer))`.
* If `warm_start=True`: Initializes new token embeddings by averaging input/output embeddings of natural language phrases:
  * `'e_type'` $\rightarrow$ `"entity type: "`
  * `'r_type'` $\rightarrow$ `"relation: "`
  * `'nr_type'` $\rightarrow$ `"next relation: "`
  * `'tail'` $\rightarrow$ `"object: "`
  * `'null'` $\rightarrow$ `"not found: "`

---

### 2.2. `s2g/linearisation/graph.py`

#### Purpose
Handles nested graph building (each entity mention co-located with its outgoing relations, omitting relation tokens for relation-less entities in joint variants, and non-head entity omission for RE variants) and unified state machine parsing.

#### Data Structures & Types
* `EntityBlock`: `Dict[str, Any]` containing `'text'`, `'type'` (optional), and `'relations'` (`List[Dict[str, Any]]` where each relation is `{'type': rel_type, 'tail_text': tail_text, 'tail_type': tail_type}`).
* `Triplet`: `Tuple[str, str, str]` $\rightarrow$ `(head_text, rel_type, tail_text)`.
* `RejectedItem`: `str` label representing a rejected (null) schema type.

#### Key Functions

##### `build_graph(ent_blocks, variant, tokens, use_nesting=True, random_graph=False, use_rejection=False, rejected_ent_types=None, rejected_rel_types=None) -> str`
* Constructs linearised nested Graph target string:
  * **Variant `joint` / `boundary_joint`**:
    - Emits all entities: `<extra_id_i> head [<e_type> type] [<r_type> rel1 <tail> tail1 <nr_type> rel2 <tail> tail2 ...]`
    - If an entity has no outgoing relations: no relation tokens are appended.
  * **Variant `re` / `boundary_re`**:
    - **Skips non-head entities** (only entities with at least one outgoing relation get a block).
    - Emits: `<extra_id_i> head [<e_type> head_type] <r_type> rel1 <tail> tail1 [<e_type> tail1_type] [<nr_type> ...]`
  * **Sentinel numbering** is contiguous from `<extra_id_0>` in every variant, and is capped at `MAX_SENTINELS = 100` (T5 provides `<extra_id_0>` .. `<extra_id_99>`); blocks beyond that are dropped with a warning.
  * **Rejection** (`use_rejection=True`) appends `<null> type` for every sampled negative, including when the graph is otherwise empty — an instance with no extractable content still yields an explicit rejection target rather than an empty string.

##### `parse_graph(text: str, tok: S2GTokens) -> Tuple[List[EntityBlock], List[RejectedItem]]`
* State-machine parser:
  1. Tracks the active head entity via `<extra_id_i>`, parses its mention text and optional `<e_type>`.
  2. Reads relations introduced by `<r_type>` / `<nr_type>`, and tail text/type after `<tail>`.
  3. Reconstructs structured `EntityBlock` list and extracts evaluation triplets.
* Tolerant of malformed generations: a sentinel index that has already been filled starts a **new** block rather than overwriting the existing one, and tail mentions that never appear as a head block are appended as entities so they still count towards NER recall.
* Types are resolved leniently — a block with no `<e_type>` inherits a type from any relation that names it as a tail.

---

## 3. Module: `s2g.data`

The `data` package handles dataset loading, memory-mapped JSONL indexing, dynamic schema sampling, batch collation, and raw dataset preprocessing.

---

### 3.0. Instance Schema (the on-disk JSONL contract)

Every preprocessor emits one JSON object per line in this shape, and `S2GCollator` consumes exactly these keys:

| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | Whitespace-joined source sentence (`" ".join(tokens)`). |
| `tokens` | `List[str]` | Token list; `offset` values index into it. |
| `entities` | `List[Dict]` | Each `{'text': str, 'offset': [start, end), 'type': str}`. |
| `relations` | `List[Dict]` | Each `{'head': <entity dict>, 'tail': <entity dict>, 'type': str}`. |
| `entity_types` | `List[str]` | Sorted unique entity types **present in this instance** (the positive entity schema). |
| `rel_types` | `List[str]` | Sorted unique relation types **present in this instance** (the positive relation schema). |

`entity_types` / `rel_types` are what `sample_types` treats as positives; everything else in the corpus schema is a negative candidate. Instances with no entities or no relations are retained (mirroring REBEL), and yield an empty target graph.

Alongside the splits, each preprocessor writes `entity.schema` and `relation.schema` — one type per line, sorted, derived from the **train** split only, and read back by `load_schema` / `load_ent_schema`.

---

### 3.1. `s2g/data/dataset.py`

#### Purpose
Random-access reader for JSONL splits that avoids holding the corpus in RAM. Backed by `mmap` plus a precomputed byte-offset table.

#### Key Classes & Functions

##### `S2GDataset(filepath, subset_fraction: Optional[float] = None, seed: Optional[int] = 0)`
* Builds an offset index at construction, then memory-maps the file.
* `__len__` $\rightarrow$ number of indexed lines; `__getitem__(i)` $\rightarrow$ `json.loads` of the byte slice `[start, end)`.
* `subset_fraction` (when in `(0, 1)`) keeps a deterministic random subsample of `max(1, n * fraction)` lines, drawn with `np.random.default_rng(seed)` and re-sorted to preserve file order.
* `__getstate__` / `__setstate__` strip and lazily reopen `_mmap` / `_file`, so the dataset can be pickled into DataLoader worker processes; `__del__` closes both.

##### `_build_offset_index(filepath) -> np.ndarray`
* Scans the file in 64 MB chunks (`_SCAN_CHUNK_BYTES`), locating newlines vectorially with `np.where(arr == 10)`.
* Returns an `(N, 2)` `int64` array of `[start, end)` byte offsets. A trailing line without a newline is included; empty lines are dropped via `keep = ends > starts`.

> **Note.** `train.py` and `evaluate.py` never pass `subset_fraction`; subsetting for validation is done with `torch.utils.data.Subset` instead (see `validation.percent_check` / `validation.train_percent_check`).

---

### 3.2. `s2g/data/collator.py`

#### Purpose
Turns raw instances into encoder/decoder token tensors: samples the schema shown in the prompt, filters the gold graph to that sampled schema, linearises it, and tokenises both sides.

#### `S2GCollator(tokenizer, ent_schema, rel_schema, config)`

Configuration is passed as a plain dict (assembled in `train.py` / `evaluate.py` / `measure_lengths.py`):

| Config key | Effect |
|---|---|
| `variant` | Selects the `prepare_{variant}` method; must be in `VALID_VARIANTS`. |
| `mode` | `'budget'` or `'bernoulli'` schema sampling (see below). |
| `max_source_length` / `max_target_length` | Truncation budgets for encoder and decoder. |
| `max_ent_types` / `max_rel_types` | Cap on the number of types shown in the prompt; `None` = full schema. |
| `prompt_type` | Passed through to the prompt builders (`'natural'`, or `'false'` for raw text). |
| `random_prompt` | Shuffle schema type order in the prompt instead of sorting. |
| `random_graph` | Shuffle entity and relation order in the target. |
| `use_rejection` | Append `<null> type` markers for sampled negatives. |
| `use_nesting` | Use `<nr_type>` for a head's 2nd+ relations; when `False`, every relation uses `<r_type>`. |
| `max_steps`, `pos_rate_*`, `neg_rate_*`, `pos_max_*`, `neg_max_*` | Bernoulli curriculum endpoints. |
| `seed` | Seeds the collator's private `random.Random`. |

##### `__call__(batch) -> Dict[str, Tensor]`
Dispatches each instance through `getattr(self, f"prepare_{self.variant}")`, collecting `(encoder_input, decoder_target)` string pairs, then tokenises the batch.

##### `prepare_re` / `prepare_boundary_re` / `prepare_joint` / `prepare_boundary_joint`
Each follows the same three steps, differing only in which schemas are sampled and whether types are used:

1. `sample_types(...)` $\rightarrow$ `(positives, negatives)` for the entity and/or relation schema.
2. `build_*_encoder_input(positives + negatives, ...)` $\rightarrow$ the prompt.
3. `organise_filter_and_block(..., allowed_types=set(positives), use_types=...)` $\rightarrow$ `build_graph(...)` $\rightarrow$ the target.

| Method | Entity schema sampled | `use_types` | Rejection types passed |
|---|---|---|---|
| `prepare_joint` | yes | `True` | entity + relation negatives |
| `prepare_re` | yes | `True` | entity + relation negatives |
| `prepare_boundary_joint` | no | `False` | relation negatives only |
| `prepare_boundary_re` | no | `False` | relation negatives only |

Because the gold graph is filtered to `set(positives)`, a type dropped from the prompt is also dropped from the target — prompt and target always agree.

##### `sample_types(instance_types, schema, max_types) -> (positives, negatives)`
* **`budget`** (used for all evaluation, and all fine-tuning configs): keeps **every** positive, then draws negatives from the pool up to `max_types - len(instance_types)`. With `max_types=None` the entire negative pool is used, i.e. the prompt lists the full schema.
* **`bernoulli`** (curriculum, intended for pre-training): truncates positives to `pos_k` and negatives to `neg_k`, applies the `max_types` budget, then independently keeps each surviving type with probability `pos_rate` / `neg_rate`. Positives may therefore be dropped, which is the point — the model learns to handle partial schemas.

##### `schedule_values() -> (pos_rate, neg_rate, pos_k, neg_k)`
Linearly interpolates each `*_start` $\rightarrow$ `*_end` pair over `frac = min(current_step, max_steps) / max_steps`.

##### `current_step` (property)
Backed by a **shared-memory** `torch` tensor rather than a plain attribute. `collate_fn` executes inside DataLoader worker processes that hold pickled copies of the collator, so a plain attribute written by `StepTrackingCallback` in the main process would never reach them, and the curriculum would stay frozen at step 0 for the whole run.

##### `tokenize(encoder_inputs, decoder_targets) -> Dict[str, Tensor]`
Tokenises both sides with `truncation=True, padding='longest'`, then replaces pad ids in the labels with `-100` so they are ignored by the loss. Returns `input_ids`, `attention_mask`, `labels`.

##### `to_eval_mode() -> S2GCollator`
Returns a **new** collator with `mode` forced to `'budget'`, leaving every other setting intact. Used by `S2GTrainer.get_eval_dataloader`, `GenerateTextSamplesCallback`, and both evaluation scripts, so that scoring never depends on curriculum state.

---

### 3.3. Preprocessing Scripts

All three share the same CLI (`--input_dir`, `--output_dir`, `--config_map`), write `train.jsonl` / `val.jsonl` / `test.jsonl` plus the two schema files, and apply label prettification from a YAML map (`configs/data/*.yaml`) whose `entities:` and `relations:` blocks rename raw corpus labels (e.g. `Peop` $\rightarrow$ `person`, `OrgBased_In` $\rightarrow$ `is based in`). Unmapped labels pass through unchanged.

| Script | Input format | Notes |
|---|---|---|
| `preprocess_conll04.py` | SpERT-style JSON array with `tokens`, `entities` (`start` / `end` / `type`), `relations` (`head` / `tail` indices) | Reads `conll04_{train,dev,test}.json`. Out-of-range relation indices are skipped. |
| `preprocess_nyt.py` | JointRE JSON array with `spo_list` / `spo_details` | Relations sorted by head start offset (matching REBEL's `nyt_typed.py`); entities deduplicated by `(start, end)` in a registry. |
| `preprocess_scierc.py` | JSONL, one document per line with `sentences`, `ner`, `relations` | Sentence-level by default; `--document_level` concatenates all sentences into one instance. Converts document-absolute indices to the target frame and drops spans that fall outside it. |

---

## 4. Module: `s2g.evaluation`

The `evaluation` package provides corpus-level and per-type macro metric calculation functions as well as the streaming `S2GEvaluator`.

---

### 4.1. `s2g/evaluation/metrics.py`

#### Purpose
Scores predicted against gold `EntityBlock` lists. Matching is **text-and-type based** — spans are compared by surface string, not by offset, since the decoder emits text rather than indices.

#### Types
* `Triplet`: `(head_text, rel_type, tail_text)` — relation *boundary* match.
* `Quintuple`: `(head_text, head_type, rel_type, tail_text, tail_type)` — relation *strict* match.
* `EntityMention`: `(head_text, entity_type)` — entity *strict* match.

#### Key Functions

##### `extract_from_blocks(blocks) -> (triplets, quintuples, entities, mentions)`
Single pass over one sentence's blocks. Builds a text $\rightarrow$ block map so a relation's tail type can fall back to the tail entity's own declared type (`rel['tail_type'] or ent_map[tail].get('type')`) — this is what makes strict scoring work for `joint`, where tail types are never emitted inline. A quintuple is only produced when both head and tail types are known.

##### `corpus_prf(all_predicted, all_gold, prefix) -> Dict[str, float]`
Micro (corpus-level) PRF: accumulates `|pred ∩ gold|`, `|pred|` and `|gold|` as **sets per sentence** across the corpus, so duplicates within a sentence collapse. Returns `{prefix}_precision`, `{prefix}_recall`, `{prefix}_f1`.

##### `per_type_macro(all_predicted, all_gold, type_fn, schema, prefix) -> Dict[str, float]`
REBEL-style macro: computes a global micro PRF **per type in `schema`**, then averages unweighted across types. Types absent from both predictions and gold contribute 0.0, matching `re_score()` in REBEL's `score.py`. Returns `macro_{prefix}_precision|recall|f1`.

##### `compute_metrics_for_variant(variant, all_pred_blocks, all_gold_blocks, rel_schema=None, ent_schema=None)`
Extracts everything in one pass, then **discards out-of-schema predictions** — predicted relation types not in `rel_schema` and entity types not in `ent_schema` are dropped before scoring, again matching REBEL, which only iterates over known types. Gold is never filtered.

Metrics emitted per variant:

| Metric key prefix | Matched tuple | `joint` | `re` | `boundary_joint` | `boundary_re` |
|---|---|:---:|:---:|:---:|:---:|
| `ner_boundary_*` | `head_text` | ✓ | ✓ | ✓ | ✓ |
| `ner_*`, `macro_ner_*` | `(head_text, head_type)` | ✓ | ✓ | — | — |
| `boundary_*`, `macro_boundary_*` | `(head_text, rel_type, tail_text)` | ✓ | ✓ | ✓ | ✓ |
| `strict_*`, `macro_strict_*` | `(head_text, head_type, rel_type, tail_text, tail_type)` | ✓ | ✓ | — | — |

Each prefix yields `_precision`, `_recall` and `_f1`. `ner_boundary` is micro-only — there is no type to group a macro average by. Because the boundary variants emit no `strict_f1`, their configs must set `validation.early_stopping_metric: boundary_f1`.

---

### 4.2. `s2g/evaluation/evaluator.py`

#### Purpose
Decodes model output straight from batch tensors, parses it into blocks, and drives a constant-memory evaluation loop that streams per-instance records to disk.

#### `S2GEvaluator(tokenizer, tokens, variant, rel_schema, ent_schema)`

##### `clean_text(text) -> str`
Strips pad / EOS / BOS strings and collapses whitespace. The S2G special tokens and sentinels are deliberately **kept** — the parser needs them.

##### `parse_text(text) -> (blocks, rejected)`
`clean_text` followed by `parse_graph`.

##### `process_batch_outputs(input_ids, generated_ids, labels) -> (pred_blocks, gold_blocks, records)`
Replaces `-100` with the pad id, then batch-decodes predictions and labels with `skip_special_tokens=False` (inputs are decoded with `skip_special_tokens=True`). Parses both sides and builds a per-instance record:

```json
{"text": "...", "prediction_raw": "...", "gold_raw": "...",
 "parsed_pred_blocks": [], "parsed_gold_blocks": [], "rejected": []}
```

Gold blocks come from the **collated labels**, not from the raw dataset — so gold is reconstructed through exactly the same linearise $\rightarrow$ parse round trip as the prediction, and any schema sampling applied by the collator applies to both sides.

##### `compute_final_metrics(all_pred_blocks, all_gold_blocks)`
Thin wrapper over `compute_metrics_for_variant` with the configured variant and schemas.

##### `run_evaluation(dataset, split, dataloader, out_dir, model, max_target_length, num_beams, device, constraint_decoding=False, length_penalty=None, early_stopping=None, no_repeat_ngram_size=None)`
Streaming loop under `torch.inference_mode()`:
1. Moves the batch to `device`, entering `torch.autocast` when the model dtype is `bfloat16` / `float16` on CUDA.
2. Calls `model.generate`; beam-only arguments (`length_penalty`, `early_stopping`, `no_repeat_ngram_size`) are attached only when `num_beams > 1`. When `constraint_decoding=True`, an FSM `LogitsProcessor` from `s2g.model` is added.
3. Parses the batch and **flushes records to `{split}_results.jsonl` immediately**, keeping only the parsed blocks in memory.
4. Writes aggregate metrics to `{split}_metrics.json` and returns them.

> **Note.** Only the parsed block lists are accumulated across the corpus; raw text never is. This is what keeps memory flat on large test sets.

---

## 5. Summary Matrix of Module Responsibilities

| Package / File | Primary Responsibility |
|---|---|
| `s2g.linearisation.special_tokens` | Token registry for `<e_type>`, `<r_type>`, `<nr_type>`, `<tail>`, `<null>`, and sentinel helper `S2GTokens.sentinel_token(idx)`. |
| `s2g.linearisation.graph` | Graph construction with `<tail>` token (`build_graph`), state-machine parsing (`parse_graph`), block filtering (`organise_filter_and_block`). |
| `s2g.linearisation.prompt` | Encoder input prompt construction for natural language schema instructions. |
| `s2g.data.dataset` | Memory-mapped JSONL reader (`S2GDataset`) with a vectorised byte-offset index, picklable into DataLoader workers. |
| `s2g.data.collator` | Schema sampling (`budget` / `bernoulli`), prompt and target construction, tokenisation and label masking (`S2GCollator`). |
| `s2g.data.preprocess_*` | Corpus-specific converters to the shared instance schema, plus `entity.schema` / `relation.schema` generation. |
| `s2g.evaluation.metrics` | Micro and per-type macro PRF metric calculation using text-and-type tuple matching. |
| `s2g.evaluation.evaluator` | Tensor-direct batch decoding, graph parsing, and the streaming evaluation loop that writes `{split}_results.jsonl` / `{split}_metrics.json`. |
