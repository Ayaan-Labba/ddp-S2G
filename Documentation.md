# S2G (Sentence-to-Graph) Codebase Documentation

This document provides a comprehensive, exhaustive technical reference for the **S2G (Sentence-to-Graph)** codebase (`ablations` branch). It is designed to furnish complete context for developers working on or extending this project.

---

## 1. High-Level Architecture Overview

**S2G** frames joint entity and relation extraction (IE) as a sequence-to-sequence (Text-to-Text) translation problem built upon **Flan-T5** (Base/Large).

By default, S2G uses natural language instruction prompts for the encoder input and generates a linearised token graph. **Every linearisation token is a reserved T5 sentinel**, so nothing is ever added to the vocabulary and no embedding resize is needed. The roles take the top of the range — `<extra_id_95>` (entity type), `<extra_id_96>` (relation), `<extra_id_97>` (nested relation), `<extra_id_98>` (tail), `<extra_id_99>` (null) — and **block markers count up from `<extra_id_0>`**, one per block, through the 95 left below them.

This branch exists to run the CoNLL04 ablation study (`ABLATION_PLAN.md`). Nesting mode and inline tail types are **config settings**, not branches. Axis 1 chose rolling markers, and that format is now the only one the code emits — the fixed-marker arm that `main` carried is gone. See §9.

> Examples throughout write the roles by their symbolic names (`<e_type>`, `<r_type>`, `<nr_type>`, `<tail>`, `<null>`) for readability; the emitted strings are the sentinels above. Block markers are shown as they are actually emitted, `<extra_id_0>` upward.

```
                            ┌──────────────────────────────────────────────────────────┐
                            │                  Encoder Input (Prompt)                  │
                            │  Extract all entities from [...] and relations from      │
                            │  [...] in the given text.                                │
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

1. **Prompt**: The encoder prompt formats the input text with task instructions and target entity/relation schema types, e.g. `"Extract all entities from [...] and relations from [...] in the given text. Text: ..."`. The boundary variants drop the entity clause. Setting `prompt.type: false` disables the instruction and feeds the raw source text instead (ablation only).
2. **Graph (Nested Scheme)**: Linearised target representation where each entity mention is co-located with its outgoing relations.
   - Every block is **opened** by a marker, the first included, and nothing closes the sequence: an *n*-block graph carries exactly *n* markers, and an empty graph is the empty string.
   - The first relation is introduced by `<r_type> rel_type <tail> tail_text` (in `re`, also `<e_type> tail_type`).
   - Subsequent relations for the same head entity are introduced by `<nr_type> rel_type <tail> tail_text`.
   - Entities with **no outgoing relations** simply omit relation tokens (ending directly after entity mention/type).
3. **Block markers**: rolling sentinels, counting upward — `<extra_id_0>` opens the *first* block and block *i* is opened by `<extra_id_i>`. Not configurable: Axis 1 measured this against a single reused marker and against a closed sequence, and this form won both.
4. **Vocabulary Special Tokens**: There are none to add. Every role and every marker is a reserved sentinel already in the T5/Flan-T5 vocabulary, so the tokenizer is left exactly as it shipped (32100 rows), no resize occurs, and there is nothing to initialise. `verify_token_integrity` is the only tokenizer step, and it asserts rather than modifies.
5. **Supported Model Variants**:
   * **`joint`**: Joint entity recognition and relation extraction. All entity mentions get their own block (`head [<e_type> type]`). Entities without outgoing relations emit no relation tokens. Tail types are emitted inline **iff `graph.joint_tail_type`**; otherwise they are recovered from the tail's own block.
   * **`boundary_joint`**: Joint entity span boundary extraction (no entity types) and relation extraction. All entity mentions get their own block (`head`). Entities without outgoing relations emit no relation tokens.
   * **`re`**: Relation extraction with the entity *type* schema supplied in the prompt (entity spans are still predicted, not given). **Only entities that act as a head in at least one relation get their own block** (`head <e_type> head_type <r_type> rel <tail> tail <e_type> tail_type`). Non-participating and tail-only entities are omitted as head blocks. `re` **always** emits inline tail types — not configurable.
   * **`boundary_re`**: Relation extraction between entity spans without entity types. **Only entities that act as a head in at least one relation get their own block** (`head <r_type> rel <tail> tail`). Non-participating and tail-only entities are omitted as head blocks.
6. **Nesting (`graph.nesting`)**: How a head's 2nd+ relations are emitted — `nr_type` (one block per head, subsequent relations on `<nr_type>`), `r_type` (one block per head, every relation on `<r_type>`), or `none` (one relation per block, mention and type repeated). See §2.2.
7. **Rejection & Null Blocks**: Optional negative schema type markers (`<null> type`) included in Graph outputs to force explicit model rejection of absent entity or relation types.
8. **Deduplication (`graph.dedup`)**: Controls whether repeated mentions collapse when the *target* is built. Deduplication keys on `(text, type)`, so homographs are never merged. Parsing never deduplicates. Held constant at `True` across the ablation.
9. **Dual scoring**: Every evaluation reports text-based metrics and offset-based metrics (`offset_` prefix) side by side. Gold comes from the preprocessed annotations, never from parsing the model's own target format.

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
  Extract all entities from [artifact, city, country, organization, person] and relations from [founded, killed, located in, place of birth, president of] in the given text. Text: Barack Obama was born in Honolulu and served as the president of the United States
  ```
* **Decoder Output** (`joint_tail_type: false`):
  ```text
  <extra_id_0> Barack Obama <e_type> person <r_type> place of birth <tail> Honolulu <nr_type> president of <tail> United States <extra_id_1> Honolulu <e_type> city <r_type> located in <tail> United States <extra_id_2> United States <e_type> country
  ```
* **Decoder Output** (`joint_tail_type: true`):
  ```text
  <extra_id_0> Barack Obama <e_type> person <r_type> place of birth <tail> Honolulu <e_type> city <nr_type> president of <tail> United States <e_type> country <extra_id_1> Honolulu <e_type> city <r_type> located in <tail> United States <e_type> country <extra_id_2> United States <e_type> country
  ```

#### 2. `boundary_joint`
* **Task**: Joint entity span boundary extraction (without entity types) + relation extraction across all entities.
* **Encoder Input (Natural Prompt)**:
  ```text
  Extract all relations from [founded, killed, located in, place of birth, president of] in the given text. Text: Barack Obama was born in Honolulu and served as the president of the United States
  ```
* **Decoder Output (Nested Graph)**:
  ```text
  <extra_id_0> Barack Obama <r_type> place of birth <tail> Honolulu <nr_type> president of <tail> United States <extra_id_1> Honolulu <r_type> located in <tail> United States <extra_id_2> United States
  ```

#### 3. `re`
* **Task**: Relation extraction with entity types provided for head and tail entities. Non-head entities (e.g. `United States`) are omitted as head blocks.
* **Encoder Input (Natural Prompt)**:
  ```text
  Extract all entities from [artifact, city, country, organization, person] and relations from [founded, killed, located in, place of birth, president of] in the given text. Text: Barack Obama was born in Honolulu and served as the president of the United States
  ```
* **Decoder Output (Nested Graph)** — the ablation baseline, carried out of Axis 1:
  ```text
  <extra_id_0> Barack Obama <e_type> person <r_type> place of birth <tail> Honolulu <e_type> city <nr_type> president of <tail> United States <e_type> country <extra_id_1> Honolulu <e_type> city <r_type> located in <tail> United States <e_type> country
  ```

#### 4. `boundary_re`
* **Task**: Relation extraction between entity mentions without entity types. Non-head entities (e.g. `United States`) are omitted as head blocks.
* **Encoder Input (Natural Prompt)**:
  ```text
  Extract all relations from [founded, killed, located in, place of birth, president of] in the given text. Text: Barack Obama was born in Honolulu and served as the president of the United States
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
Defines the special token registry and tokenizer integrity verification. Nothing here modifies the tokenizer or the model.

#### Constants & Token Map
* `ALL_TOKEN_NAMES`: `['e_type', 'r_type', 'nr_type', 'tail', 'null']`
* `VALID_VARIANTS`: `{'re', 'boundary_re', 'boundary_joint', 'joint'}`
* `NUM_SENTINELS`: `100` — T5 ships exactly `<extra_id_0>` .. `<extra_id_99>`.
* `MAX_MARKER_SENTINELS`: `NUM_SENTINELS - len(ALL_TOKEN_NAMES)` = `95` — the indices below the roles, all of them available to block markers.

#### Special Token Mapping (`S2GTokens.token_strs`)
| Token Key | Sentinel | Semantic Role |
|---|---|---|
| `'e_type'` | `<extra_id_95>` | Entity type token in Graph |
| `'r_type'` | `<extra_id_96>` | Primary relation type token in Graph |
| `'nr_type'` | `<extra_id_97>` | Nested relation type token (for same head entity) in Graph |
| `'tail'` | `<extra_id_98>` | Static tail token preceding tail entity text |
| `'null'` | `<extra_id_99>` | Rejection marker; **active only with `use_rejection`** |

> The map is *derived* from `ALL_TOKEN_NAMES` and `MAX_MARKER_SENTINELS` rather than written out, so the marker ceiling and the role block cannot drift into each other. A checkpoint records its map in `s2g_format.json`; scoring one under a different map would mis-parse every target rather than fail, so `evaluate.py` refuses outright.

#### Key Classes & Functions

##### `S2GTokens(variant: str, use_rejection: bool = False)`
* **`base_tok_map`**: Maps active role tokens per variant.
* **`self.active_tokens`**: Adds `'null'` when `use_rejection`. This gate matters more than it looks: an inactive role is not in `role_token_strs`, so the parser would read its sentinel as a **block marker** rather than as a role — silently, producing wrong blocks instead of an error.
* **`self.role_token_strs`**: The active role tokens. `parse_graph` tests these by exact identity *before* treating anything as a marker, so a role sentinel can never be read as a separator.
* **`S2GTokens.sentinel_token(idx)`**: `<extra_id_{idx}>`, for block markers.

##### `verify_token_integrity(tokenizer) -> None`
Raises `RuntimeError` unless every one of the 100 sentinels (a) encodes to exactly **one id** and (b) decodes back **verbatim** under `skip_special_tokens=False`. Markers and roles are all sentinels, so the range covers everything the format can emit. Those are the two properties it depends on: a marker split across pieces breaks the block numbering, and a role split across pieces is never matched by the parser.

> It asserts behaviour rather than membership of a registry, deliberately. `additional_special_tokens` was removed in transformers 5 and `all_special_tokens` can drop the sentinels, while `added_tokens_decoder` keeps them flagged and both properties above continue to hold. Checking behaviour survives both quirks; checking a list did not.

> **There is no `add_special_tokens_to_tokenizer`.** Registration, `resize_token_embeddings`, `tie_word_embeddings = False` and warm-start initialisation were all removed with the dedicated-token arm: nothing the format emits is new vocabulary, so there is nothing to add, resize or initialise. `train.warm_start` no longer exists either, and a config still carrying it fails at load.

---

### 2.2. `s2g/linearisation/graph.py`

#### Purpose
Handles nested graph building (each entity mention co-located with its outgoing relations, omitting relation tokens for relation-less entities in joint variants, and non-head entity omission for RE variants) and unified state machine parsing. Nesting mode and inline tail types are **emission-time** settings; parsing is a single code path shared by every arm.

#### Data Structures & Types
* `EntityBlock`: `Dict[str, Any]` containing `'text'`, `'type'` (optional), and `'relations'` (`List[Dict[str, Any]]` where each relation is `{'type': rel_type, 'tail_text': tail_text, 'tail_type': tail_type}`).
* `Triplet`: `Tuple[str, str, str]` $\rightarrow$ `(head_text, rel_type, tail_text)`.
* `RejectedItem`: `str` label representing a rejected (null) schema type.

#### Key Functions

##### `organise_filter_and_block(entities, relations, allowed_ent_types, allowed_rel_types, variant='joint', use_types=True, dedup=True) -> List[EntityBlock]`
Turns raw instance annotations into the block list that `build_graph` linearises.

1. Filters entities to `allowed_ent_types` (skipped when `use_types=False`) and relations to `allowed_rel_types` whose head **and** tail survived; sorts both by offset.
2. Selects which entities are entitled to a block — `joint` / `boundary_joint` emit every entity, `re` / `boundary_re` only those heading at least one relation.
3. Builds the blocks, governed by `dedup`.
4. Attaches each relation to its head block.

**`dedup`** (config key `graph.dedup`) controls collapsing:

| `dedup` | Entities | Relations |
|---|---|---|
| `True` (default) | Mentions merge on **`(text, type)`** | Merge on `(head_text, head_type, rel_type, tail_text, tail_type)` |
| `False` | One block per mention | Every relation kept |

Keying on `(text, type)` rather than text alone is what keeps **homographs** — `Washington` the person versus `Washington` the location — as separate blocks. Boundary variants carry `type=None`, so their key degenerates to text, as intended.

##### `build_graph(ent_blocks, variant, tokens, nesting='nr_type', joint_tail_type=False, random_graph=False, use_rejection=False, rejected_ent_types=None, rejected_rel_types=None) -> str`
Constructs the linearised nested Graph target string.

* **Which blocks are emitted**: `joint` / `boundary_joint` emit every entity; `re` / `boundary_re` emit only entities heading at least one relation. Selection happens *before* the cap, so a target is never under-filled by relation-less entities that were going to be skipped anyway.
* **Every block is marked, the first included.** `sentinel_token(i)` opens block *i*, so `<extra_id_0>` opens the **first** block.
* **Nothing closes the sequence.** A terminal marker was tried and measured worse, with the roles on sentinels and with them on dedicated tokens, so none is emitted: an *n*-block graph ends on the last block's content, and an empty graph is the empty string.
* **`nesting`** (`'nr_type'` | `'r_type'` | `'none'`):
  | Value | Blocks per head | Relation token |
  |---|---|---|
  | `'nr_type'` (default) | one | `<r_type>` first, `<nr_type>` thereafter |
  | `'r_type'` | one | `<r_type>` for every relation |
  | `'none'` | one **per relation**, mention and type repeated | `<r_type>` |

  > **Naming trap.** The retired `use_nesting=False` maps to `'r_type'`, **not** to `'none'` — the old flag only swapped the relation token, it never split the block. `'none'` is new behaviour, implemented by expanding blocks at emission time. Block *grouping* is untouched: `organise_filter_and_block` keeps merging mentions on `(text, type)` exactly as in the other arms, so `'none'` is not `dedup=False` and must not be implemented as such.

* **Tail types**: `re` always emits `<e_type> tail_type`; `joint` emits it iff `joint_tail_type=True`; the boundary variants never do.
* **Cap**: markers spend one sentinel per block, the first included, so the 95 indices below the roles allow **95** blocks; rejection reserves one further index for its own marker (Stage 3), leaving **94**. Excess blocks are truncated with a warning. The cap is what keeps a marker from ever reaching `<extra_id_95>` and colliding with a role.
* **Rejection** (`use_rejection=True`) appends `<null> type` for every sampled negative, including when the graph is otherwise empty. Stage 3 of the port replaces this with the single-marker CoT rejection tail.

##### `max_emitted_blocks(use_rejection) -> int`
The block ceiling, `MAX_MARKER_SENTINELS` less one when rejection reserves its index.

##### `parse_graph(text: str, tok: S2GTokens) -> Tuple[List[EntityBlock], List[RejectedItem]]`
* State-machine parser:
  1. Splits on `<extra_id_\d+>` — every linearisation token is a sentinel, so one pattern isolates them all.
  2. **Identity before pattern.** Role tokens are matched by exact string equality against `tok.role_token_strs` *first*; any remaining sentinel is a block marker. This ordering is the whole reason roles and markers can share one range.
  3. **Seeds a first block**, so that content preceding any marker still lands somewhere — a malformed generation, or a target in the earlier format where the first block was unmarked. The seed is dropped if it never receives text, which is the normal case now that every block is marked.
  4. Reads relations introduced by `<r_type>` / `<nr_type>`, and tail text/type after `<tail>`.
* **Append, never index.** Any marker appends a new block; its index is read and then **discarded**, so a repeated or out-of-order index in a malformed generation is harmless rather than corrupting.
* **Parsing never deduplicates, for any variant.** Every emitted block is retained, so repeated mentions and repeated relations survive into scoring exactly as generated. Deduplication is a *target construction* concern only (`graph.dedup`), never a parsing one.

##### `resolve_tail_entities(entities: List[EntityBlock]) -> List[EntityBlock]`
Reconciles relation tails against the entity blocks, in place. Shared by `parse_graph` and by gold construction (`s2g.evaluation.gold`), so both sides of a comparison are reconciled identically.

* A tail mention resolves to the **first** block carrying that text. Without inline tail types the type has to be recovered from the entity's own block, and duplicated mentions must resolve deterministically.
* Tails with no block of their own are appended as entities, so they still count towards NER recall — this is what makes the RE variants scorable on entities at all.
* Type resolution runs both ways: an untyped relation inherits `tail_type` from its matched block (the joint case), and an untyped block inherits a type from an inline `<e_type>` on a relation naming it as tail (the RE case).

> **Known limitation (strict scoring ceiling).** First-occurrence matching cannot be *correct* for a homograph tail in the joint variants: if `Washington [person]` precedes `Washington [location]`, a relation pointing at the latter resolves to the former. This is inherent to the joint format, where tail types are never emitted inline and surface text is the only handle.
>
> Since gold is now read from the annotation rather than round-tripped through the target format (§4.3), gold carries the *true* tail type while the prediction carries the first-occurrence one. The model is therefore genuinely penalised on these triples, and a flawless generation cannot reach `strict_f1 = 1.0` on a sentence containing a homograph tail. This is honest measurement, not a bug — but it is a ceiling worth knowing about when reading strict numbers.
>
> Affected volume is small: 0 cases in CoNLL04 and NYT, 3 in SciERC train, and 28 / 1 / 4 in `scierc_doc` train / val / test. Emitting inline tail types for `joint` is the only real fix — which is exactly what `graph.joint_tail_type: true` does, and why the B1 vs B2 comparison partly measures a scoring artefact rather than a pure format effect (§9).

---

### 2.3. `s2g/linearisation/prompt.py`

#### Purpose
Encoder input construction. One builder serves every variant; the boundary variants simply drop the entity clause.

```text
Extract all entities from [{e_types}] and relations from [{r_types}] in the given text. Text: {text}
Extract all relations from [{r_types}] in the given text. Text: {text}
```

##### `build_instruction(rel_types, ent_types=None, use_ent_types=True, random_order=False) -> str`
The instruction alone, without the source text — kept separate so Stage 3's CoT prompt can wrap the identical wording in a different frame.

##### `build_encoder_input(text, rel_types, ent_types=None, use_ent_types=True, random_order=False, prompt='natural') -> str`
Instruction + `" Text: {text}"`. `prompt.type: false` returns the raw text instead.

Type lists are sorted unless `random_prompt`. The four per-variant builders (`build_re_encoder_input`, `build_joint_encoder_input`, `build_boundary_re_encoder_input`, `build_boundary_joint_encoder_input`) are retained as thin wrappers with unchanged signatures, so `collator.py` is untouched by the consolidation.

> **The leading verb is not a config key.** The `Extract` / `Mark` arm (Axis 3, C1) is a one-word edit made by hand in this file.

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

#### Annotation Identity: the Offset Rule

**An entity annotation is identified by its offset.** Every preprocessor keys entities on `(start, end)`, which fixes what counts as a duplicate:

* The **same span** recorded more than once is one annotation. NYT has no standalone entity list — mentions are derived from relation participants, so a span heading three relations is annotated three times — and `entities_registry` collapses those to one record. SciERC's `span_idx` / `span_to_ent` do the same. CoNLL04 needs no collapsing; its entity list is already offset-unique.
* The **same surface text at different offsets** is two annotations, and both are kept. This is the case that matters for scoring: a sentence mentioning `Moscow` twice, each in its own relation, carries two gold entities and two gold relations.
* **Homographs** follow automatically. Type never participates in the key, so a differing type can neither merge nor drop a mention.

Duplicate counts under this rule, per split (train / val / test):

| Dataset | Entities | Relations | Surface forms at multiple offsets | Extra entity records | Homographs |
|---|---|---|---|---|---|
| conll04 | 3377 / 893 / 1079 | 1283 / 343 / 422 | 59 / 18 / 18 | 62 / 18 / 20 | 0 / 0 / 0 |
| nyt | 121450 / 10848 / 10836 | 94222 / 8489 / 8616 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| scierc | 5598 / 811 / 1685 | 3219 / 455 / 974 | 64 / 12 / 11 | 67 / 13 / 12 | 3 / 0 / 0 |
| scierc_doc | 5598 / 811 / 1685 | 3219 / 455 / 974 | 496 / 69 / 115 | 639 / 88 / 164 | 28 / 1 / 4 |

> **Note.** NYT has no repeated-text entities at all: each surface form gets one canonical span per sentence. Its duplication shows up only as repeated relation triples (5856 / 504 / 496), which share head *and* tail offsets and are therefore the same annotation stated twice — correctly collapsed. NYT's raw `spo_details` contain **zero** spans with conflicting types in any split.

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
| `nesting` | `'nr_type'` / `'r_type'` / `'none'` — see `build_graph`. |
| `joint_tail_type` | Emit inline tail types for `joint`. `re` always emits them regardless. |
| `prompt_style` | `'direct'` or `'cot'`. Reserved for Stage 3; unused by the current builders. |
| `dedup` | Collapse entities on `(text, type)` and relations on the full quintuple during block building. See `organise_filter_and_block`. |
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
Scores predictions against gold on **two parallel tracks**:

* **Text-based** — spans compared by surface string, the historical behaviour.
* **Offset-based** (prefix `offset_`) — every predicted mention is located in the source token sequence and compared by span index, matching how the corpora actually annotate.

Both tracks are reported from every evaluation. Text metrics keep their original keys and values.

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

##### `score_bundles(variant, pred_bundles, gold_bundles, rel_schema, ent_schema, prefix='')`
Scores per-instance `(triplets, quintuples, entities, mentions)` bundles. **Discards out-of-schema predictions** — predicted relation types not in `rel_schema` and entity types not in `ent_schema` are dropped before scoring, matching REBEL, which only iterates over known types. Gold is never filtered.

Text tuples and offset tuples share the same positional layout — the type sits at index 1 of a mention or triplet and index 2 of a quintuple — so one scoring path serves both. Only `prefix` differs (`''` vs `'offset_'`).

##### `compute_metrics_for_variant(variant, all_pred_blocks, all_gold_blocks, rel_schema=None, ent_schema=None, all_pred_offsets=None, all_gold_offsets=None)`
Extracts text tuples from the blocks in one pass and calls `score_bundles` with `prefix=''`; when offset bundles are supplied it calls it a second time with `prefix='offset_'` and merges the result.

Metrics emitted per variant:

| Metric key prefix | Matched tuple | `joint` | `re` | `boundary_joint` | `boundary_re` |
|---|---|:---:|:---:|:---:|:---:|
| `ner_boundary_*` | `head_text` | ✓ | ✓ | ✓ | ✓ |
| `ner_*`, `macro_ner_*` | `(head_text, head_type)` | ✓ | ✓ | — | — |
| `boundary_*`, `macro_boundary_*` | `(head_text, rel_type, tail_text)` | ✓ | ✓ | ✓ | ✓ |
| `strict_*`, `macro_strict_*` | `(head_text, head_type, rel_type, tail_text, tail_type)` | ✓ | ✓ | — | — |

Each prefix yields `_precision`, `_recall` and `_f1`. `ner_boundary` is micro-only — there is no type to group a macro average by. Because the boundary variants emit no `strict_f1`, their configs must set `validation.early_stopping_metric: boundary_f1`.

Every row above is emitted a second time with an `offset_` prefix (`offset_ner_boundary_f1`, `macro_offset_strict_f1`, ...), matched on offsets instead of text:

| Metric key prefix | Matched tuple |
|---|---|
| `offset_ner_boundary_*` | `(start, end)` |
| `offset_ner_*`, `macro_offset_ner_*` | `((start, end), head_type)` |
| `offset_boundary_*`, `macro_offset_boundary_*` | `(head_span, rel_type, tail_span)` |
| `offset_strict_*`, `macro_offset_strict_*` | `(head_span, head_type, rel_type, tail_span, tail_type)` |

---

### 4.2. `s2g/evaluation/offsets.py`

#### Purpose
Projects predicted graphs onto source offsets. The decoder emits surface text, so a prediction has to be located in the token sequence before it can be scored against offset annotations.

##### `OffsetResolver(tokens)`
`resolve(text)` returns **every** offset where the mention occurs, by sliding a window of `len(text.split())` tokens over the source. Results are memoised per instance, so a mention resolves to the same offsets whether met as an entity block, a relation head, or a relation tail — otherwise a triplet's endpoints could disagree with the entity predictions drawn from the same text.

Text occurring nowhere in the source is a **hallucination**. It receives a unique negative sentinel offset `(-1, -n)` rather than being dropped: dropping would quietly inflate precision, whereas a sentinel can never match gold yet still counts towards the prediction total. Sentinels appear in the per-instance records, so hallucinated mentions can be inspected directly.

##### `project_blocks(blocks, tokens) -> (bundle, resolution_map)`
Emits **one predicted entity per match**, and expands each relation over the **cross product** of its head and tail matches: a head matching *k* times and a tail matching *m* times yields *k x m* predicted relation instances. Also returns the text -> offsets map that produced the bundle, for inclusion in the evaluation records.

This is what repairs duplicate-mention recall. Given a sentence where `Moscow` occurs twice and both occurrences are gold tails of the same head, a model emitting the relation **once** is credited with both gold annotations, because the single emission projects onto both offsets.

Conversely, offset scoring is naturally set-like on spans: emitting the same mention twice still projects onto the same offsets, so duplicate predictions cannot be rewarded twice.

> **Known limitation (offset precision ceiling).** Projection is blind to *which* occurrence the model meant. A mention emitted once projects onto **every** occurrence of its text, so when a surface form is ambiguous the extra projections are wrong by construction: recall is unaffected, precision pays. On the `Washington` homograph sentence a flawless generation scores `offset_boundary_recall = 1.0` but `offset_boundary_precision = 0.5`, and both `Washington` spans inherit both predicted types, so `offset_ner_f1 < 1.0` even though `offset_ner_boundary_f1 = 1.0`.
>
> This follows directly from the "each matched offset is a unique prediction" rule and applies to *unambiguous* repeated mentions too — there it is exactly the desired behaviour, since every occurrence really is gold. Only genuinely ambiguous forms pay.

---

### 4.3. `s2g/evaluation/gold.py`

#### Purpose
Builds gold **directly from the preprocessed instance**. Gold was previously obtained by parsing the collated labels, which made it a round trip through `build_graph` -> `parse_graph` and silently inherited every lossy step of that path: targets truncated at `max_target_length` lost their tail, and joint tail types were recovered by surface matching instead of being read off the annotation.

##### `build_gold_blocks(instance, variant, dedup=True) -> List[EntityBlock]`
Runs `organise_filter_and_block` on the raw instance, passing the instance's own `entity_types` / `rel_types` as the allowed schema. This reproduces `budget` sampling exactly — budget keeps every positive and only pads the prompt with negatives, so nothing in the gold graph is ever filtered out, and evaluation always runs in budget mode via `to_eval_mode()`. The result is passed through `resolve_tail_entities`, since predictions are reconciled the same way; without it the RE variants would score every tail mention as a precision error.

##### `build_gold_offsets(instance, variant) -> bundle`
Reads offsets straight off the annotation, **independent of `dedup`** — a deduplicated block keeps only its first offset, so offset gold cannot be derived from blocks without losing exactly the repeated mentions it exists to measure. For `re` / `boundary_re` it is restricted to relation participants, since those variants never ask the model for non-participating entities and scoring against them would cap recall at an unreachable value.

---

### 4.4. `s2g/evaluation/evaluator.py`

#### Purpose
Decodes model output straight from batch tensors, parses it into blocks, pairs it with gold from the dataset, and drives a constant-memory evaluation loop that streams per-instance records to disk.

#### `S2GEvaluator(tokenizer, tokens, variant, rel_schema, ent_schema, dedup=True)`

##### `clean_text(text) -> str`
Strips pad / EOS / BOS strings and collapses whitespace. The S2G linearisation tokens are deliberately **kept** — the parser needs them.

##### `parse_text(text) -> (blocks, rejected)`
`clean_text` followed by `parse_graph`.

##### `build_gold(instance) -> (blocks, offset_bundle)`
Thin wrapper over `build_gold_blocks` / `build_gold_offsets` using the configured variant and `dedup`.

##### `process_batch_outputs(input_ids, generated_ids, labels, instances) -> (pred_blocks, gold_blocks, pred_offsets, gold_offsets, records)`
Replaces `-100` with the pad id, then batch-decodes predictions with `skip_special_tokens=False` (inputs with `skip_special_tokens=True`). Parses the prediction, projects it onto offsets, and pairs it with gold built from `instances`. The decoded labels are retained **only** as a debugging field.

Per-instance record:

```json
{"text": "...", "encoder_input": "...", "prediction_raw": "...", "gold_raw": "...",
 "parsed_pred_blocks": [], "gold_blocks": [],
 "pred_offsets": {"mention text": [[start, end]]}, "rejected": []}
```

Negative values in `pred_offsets` mark hallucinated mentions.

> **Breaking change.** `parsed_gold_blocks` is now `gold_blocks` and is dataset-derived; `text` is the source sentence rather than the decoded prompt, which moved to `encoder_input`; `pred_offsets` is new. Anything consuming `{split}_results.jsonl` needs updating.

##### `compute_final_metrics(all_pred_blocks, all_gold_blocks, all_pred_offsets=None, all_gold_offsets=None)`
Thin wrapper over `compute_metrics_for_variant` with the configured variant and schemas.

##### `run_evaluation(dataset, split, dataloader, out_dir, model, max_target_length, num_beams, device, constraint_decoding=False, length_penalty=None, early_stopping=None, no_repeat_ngram_size=None)`
Streaming loop under `torch.inference_mode()`:
1. Moves the batch to `device`, entering `torch.autocast` when the model dtype is `bfloat16` / `float16` on CUDA.
2. Calls `model.generate`; beam-only arguments (`length_penalty`, `early_stopping`, `no_repeat_ngram_size`) are attached only when `num_beams > 1`. When `constraint_decoding=True`, an FSM `LogitsProcessor` from `s2g.model` is added.
3. Slices the corresponding instances out of `dataset` and parses the batch, then **flushes records to `{split}_results.jsonl` immediately**, keeping only the parsed blocks and offset bundles in memory.
4. Writes aggregate metrics to `{split}_metrics.json` and returns them.

> **Gold alignment.** Instances are taken from `dataset` by position via a running cursor. This is sound because both callers build the loader with `shuffle=False` and no `drop_last`, so batch *n* is a contiguous slice. A mismatch between the cursor and `len(dataset)` at the end of the loop is logged as a warning.

> **Note.** Only the parsed block lists and offset bundles are accumulated across the corpus; raw text never is. This is what keeps memory flat on large test sets.

---

### 4.5. Worked Evaluation Example

A real instance from `data/conll04/test.jsonl`, chosen because it repeats a mention. Tokens (indexed):

```text
0:Dancers 1:of 2:Moscow 3:'s 4:Bolshoi 5:Ballet 6:in 7:Moscow 8:and 9:Leningrad 10:'s 11:Kirov 12:Ballet ...
```

`Moscow` is annotated **twice**, at `[2,3]` and `[7,8]`, and `Bolshoi Ballet` is based in *both*:

```text
(Bolshoi Ballet [4,6], organization based in, Moscow [2,3])
(Bolshoi Ballet [4,6], organization based in, Moscow [7,8])
(Kirov Ballet  [11,13], organization based in, Leningrad [9,10])
```

#### Step 1 — gold, both tracks

| | Entities | Relations |
|---|---|---|
| Text gold | 10 | **4** |
| Offset gold | 11 | **5** |

The text track loses one relation and one entity outright: `(Bolshoi Ballet, organization based in, Moscow)` is one string tuple no matter how many times it is annotated. This is failure mode 1 — the ceiling sits below the true annotation count, so a model cannot reach 100% recall even in principle.

#### Step 2 — a prediction

Suppose the model emits one correct relation, the entity it points at, and one hallucination:

```text
<extra_id_0> Bolshoi Ballet <e_type> organization <r_type> organization based in <tail> Moscow
<extra_id_1> Moscow <e_type> location
<extra_id_2> Atlantis <e_type> location
```

(Line-broken for readability; the target is one line.)

#### Step 3 — projection onto offsets

```json
{"Bolshoi Ballet": [[4, 6]], "Moscow": [[2, 3], [7, 8]], "Atlantis": [[-1, -1]]}
```

`Moscow` matches twice, so the single emitted relation expands into **two** predicted triplets — `(4,6) -> (2,3)` and `(4,6) -> (7,8)` — both of which are gold. `Atlantis` occurs nowhere in the source and takes the sentinel `(-1,-1)`.

#### Step 4 — scores

| Metric | Text | Offset |
|---|---|---|
| `ner_boundary_precision` | 0.667 | 0.750 |
| `ner_boundary_recall` | 0.200 | 0.273 |
| `boundary_precision` | 1.000 | 1.000 |
| `boundary_recall` | **0.250** | **0.400** |

Relation recall rises from 1/4 to 2/5: the one emitted relation is credited against **both** gold annotations, because the mention it names genuinely occurs at both offsets. Precision is unharmed — both projections are correct — while the hallucinated `Atlantis` still counts against entity precision on both tracks rather than vanishing.

> Contrast with the ambiguous case in §2.2 and §4.2: here both occurrences of `Moscow` carry the same type, so every projection is right. Had they been a homograph, the extra projection would have been wrong and precision would have fallen instead.

---

## 5. Module: `s2g.training`

### 5.1. `s2g/training/trainer.py`

#### `S2GTrainer(Seq2SeqTrainer)`
Extra constructor keywords beyond `Seq2SeqTrainer`: `variant`, `tokens`, `ent_schema`, `rel_schema`, `eval_train_dataset`, `scheduler_type`, `dedup`.

##### `create_scheduler(...)`
Adds an **inverse square root** schedule (linear warmup, then `sqrt(warmup / step)`) when `scheduler.type: inverse_sqrt`. Because HF has no such type, `train.py` passes `lr_scheduler_type='constant'` in that case and this override takes over.

##### `get_eval_dataloader(eval_dataset=None)`
Swaps in a budget-mode collator, then defers to `Trainer` so distributed sharding, pinned memory and prefetching are preserved. Also records the dataset for gold construction, and drops HF's cached `"eval"` dataloader whenever the requested dataset changes — `Trainer` caches every non-string eval dataset under that single key when `dataloader_persistent_workers` is set, so alternating between the validation set and the train subset would otherwise reuse whichever loader was built first.

##### `evaluate(eval_dataset=None, **gen_kwargs)`
Runs the normal validation pass, then a second pass over `eval_train_dataset` under the `eval_train` prefix for train/val gap monitoring. Early-stopping callbacks are detached for that second pass and restored in a `finally`, so the train subset can never trigger early stopping.

##### `compute_metrics_hf(eval_preds)`
Decodes predictions, parses them, then takes gold from the dataset recorded by `get_eval_dataloader` and projects predictions onto offsets.

##### `gold_instances(num_preds)`
Returns the backing instances in prediction order, or `None` when that order cannot be trusted — in which case `compute_metrics_hf` falls back to the old label round trip and reports **text metrics only**, with a warning. Falls back when:

* no eval dataset was recorded;
* `world_size > 1` — HF's distributed eval sampler shards strided, so gathered predictions no longer follow dataset order and positional pairing would score every prediction against the wrong gold;
* `len(dataset) != num_preds`.

Single-process runs are unaffected, which includes all of `evaluate.py` and `train.py`'s post-training evaluation (both guarded by `is_world_process_zero`).

---

### 5.2. `s2g/training/callbacks.py`

| Callback | Responsibility |
|---|---|
| `StepTrackingCallback` | Writes `state.global_step` into the collator's shared-memory counter each step, driving the bernoulli curriculum. |
| `GenerateTextSamplesCallback` | Every `callbacks.sample_generation_interval` steps (and once at train begin), generates from a fixed 8-instance sample and logs a W&B table: source, encoder input, predicted/gold entities, triplets and raw graphs. Rank-0 only; exceptions are logged, never raised. |
| `PeriodicCheckpointCallback` | Forces a save every `checkpoint.every_n_steps` and writes `run_metadata.json` (`wandb_run_id`, `last_step`) for resumable W&B runs. |
| `S2GEarlyStoppingCallback` | `EarlyStoppingCallback` that refuses to increment its patience counter while the best metric is still `<= 0.0`, so a model that has not yet produced a parseable graph is not killed early. |
| `load_run_metadata(output_dir)` | Reads `run_metadata.json` back for W&B resume; returns `None` with a warning if absent. |

---

## 6. Module: `s2g.scripts`

### 6.1. `config_utils.py`
OmegaConf **structured** config: the `S2GConfig` dataclass tree is the schema, so unknown keys and wrong types are rejected at load time.

`load_config()` extracts `--config <path>` (or `--config=<path>`), merges the YAML over the dataclass defaults, then merges dotlist overrides. `validate_dotlist` rejects anything starting with `-` or missing `=`, so a mistyped flag fails loudly instead of being ignored.

```bash
python -m s2g.scripts.train --config configs/variants/joint/conll04.yaml \
    optimizer.lr=1e-4 graph.dedup=false
```

Config groups: `data`, `model`, `tokenizer`, `prompt`, `graph`, `optimizer`, `scheduler`, `train`, `validation`, `generation`, `evaluation`, `checkpoint`, `callbacks`, `wandb`, `hardware`.

### 6.2. `train.py`
Fine-tuning and pre-training share this entry point.

1. Sets `CUDA_VISIBLE_DEVICES` from `hardware.gpu_ids` (ignored under `WORLD_SIZE > 1`), seeds everything from `train.seed`.
2. Initialises W&B, resuming the run id from `run_metadata.json` when `checkpoint.resume_from` is set; writes the resolved config to `{output_dir}/config.yaml`.
3. Loads train/val datasets; `validation.percent_check` and `validation.train_percent_check` select deterministic `Subset`s.
4. Loads the model and calls `verify_token_integrity`; the tokenizer and embedding matrix are left untouched.
5. Builds the collator, callbacks, `Seq2SeqTrainingArguments` and `S2GTrainer`, then trains.
6. On rank 0: saves `best_model/` with `variant.txt` and `s2g_format.json`, then runs streaming evaluation on val and test and logs `final_val/*` / `final_test/*` to W&B.

**`s2g_format.json`** persists every setting that changes how targets are linearised — `variant`, `prompt_type`, `style`, `use_rejection`, `nesting`, `joint_tail_type`, `dedup`, `max_ent_types`, `max_rel_types`, `token_strs` — so standalone evaluation cannot silently score against a different format. This matters more on this branch than it ever did: with eight arms in flight, scoring an arm's checkpoint under another arm's format is a live risk, not a hypothetical one.

### 6.3. `evaluate.py`
Standalone evaluation of a saved checkpoint. Reads `s2g_format.json` from the checkpoint directory and prefers it over the evaluation config for every format-critical setting — `nesting`, `joint_tail_type`, `style` and the rest — warning loudly when the sidecar is missing and **raising** when its `token_strs` disagree with the current map, since no setting could make those metrics meaningful. The variant is resolved from the sidecar, then `variant.txt`, then the config. Collation is forced to budget mode via `to_eval_mode()`.

### 6.4. `measure_lengths.py`
Scans every split and reports p50/p75/p90/p95/p99/max encoder and decoder token lengths, then suggests `max_source_length` / `max_target_length` as p99 rounded up to a multiple of 32. **Re-run once per ablation arm**: the prompt wording and `nesting: none` both change target length, so a budget measured for one arm may truncate another. The measured values are recorded per arm rather than held constant. Sets the collator's step to `max_steps` first, so bernoulli schedules are measured at their worst-case negative-sampling endpoint.

### 6.5. `measure_vram.py`
Binary-searches the largest train batch size (forward + backward) and eval batch size (`model.generate` at full target length) that fit in VRAM.

> **Not in use:** `inference.py`, `train_rebel.py`, and `s2g/model/constraint_decoder.py`.

---

## 7. Configuration Files

```text
configs/
├── pretrain.yaml            # REBEL pre-training (flan-t5-large, bernoulli curriculum)
├── finetune.yaml            # Generic fine-tuning defaults + per-dataset suggestions
├── evaluate.yaml            # Standalone evaluation defaults
├── data/                    # Label prettification maps, consumed by preprocess_*
│   └── conll04 | nyt | scierc | scierc_doc .yaml
└── variants/                # Ready-to-run per (variant, dataset) configs
    ├── ablation/            # CoNLL04 ablation arms (see §9)
    │   └── baseline.yaml
    └── joint | boundary_joint | re | boundary_re
        └── conll04.yaml | nyt.yaml
```

`configs/data/*.yaml` hold `entities:` and `relations:` maps that rename raw corpus labels (`Peop` -> `person`, `OrgBased_In` -> `is based in`); unmapped labels pass through unchanged.

Points worth knowing when writing a new variant config:

* `validation.early_stopping_metric` must be `boundary_f1` for the boundary variants — they emit no `strict_f1`. The offset metrics (`offset_strict_f1`, ...) are also valid choices.
* `scheduler.type: inverse_sqrt` is handled by `S2GTrainer`, not HF.
* `graph.dedup`, `graph.nesting`, `graph.joint_tail_type`, `graph.use_rejection`, `prompt.type` and `prompt.style` must match between training and evaluation; the `s2g_format.json` sidecar enforces this automatically.
* `graph.use_nesting`, `graph.markers` and `train.warm_start` no longer exist. Because the config is a *structured* OmegaConf schema, a stale key fails at load time rather than being ignored — including as a CLI override, so `graph.markers=rolling` is now an error rather than a no-op.
* CoNLL04 runs use flan-t5-base, ~2882 steps (~100 epochs over 922 sentences at effective batch 32), `constant_with_warmup`, and validate once per epoch (`check_interval: 29`).
* NYT values are placeholders pending benchmarking.

---

## 8. Summary Matrix of Module Responsibilities

| Package / File | Primary Responsibility |
|---|---|
| `s2g.linearisation.special_tokens` | Sentinel role registry (`<extra_id_95>` .. `<extra_id_99>`), block-marker allocation, tokenizer integrity verification. |
| `s2g.linearisation.graph` | Block building with `dedup` (`organise_filter_and_block`), graph construction with nesting / tail-type settings (`build_graph`), state-machine parsing (`parse_graph`), tail reconciliation (`resolve_tail_entities`). |
| `s2g.linearisation.prompt` | Single encoder-input builder (`build_instruction` / `build_encoder_input`) plus per-variant wrappers. |
| `s2g.data.dataset` | Memory-mapped JSONL reader (`S2GDataset`) with a vectorised byte-offset index, picklable into DataLoader workers. |
| `s2g.data.collator` | Schema sampling (`budget` / `bernoulli`), prompt and target construction, tokenisation and label masking (`S2GCollator`). |
| `s2g.data.preprocess_*` | Corpus-specific converters to the shared instance schema, plus `entity.schema` / `relation.schema` generation. |
| `s2g.evaluation.metrics` | Micro and per-type macro PRF over text and offset tuples (`score_bundles`). |
| `s2g.evaluation.offsets` | Projection of predicted mentions onto source offsets, with hallucination sentinels. |
| `s2g.evaluation.gold` | Gold blocks and gold offset tuples built directly from the preprocessed instance. |
| `s2g.evaluation.evaluator` | Tensor-direct batch decoding, graph parsing, and the streaming evaluation loop that writes `{split}_results.jsonl` / `{split}_metrics.json`. |
| `s2g.training.trainer` | Inverse-sqrt scheduling, budget-mode eval loaders, train-subset evaluation, dataset-sourced gold with a DDP guard. |
| `s2g.training.callbacks` | Curriculum step tracking, W&B sample tables, periodic checkpoints, guarded early stopping. |
| `s2g.scripts.*` | Structured-config entry points: training, standalone evaluation, length and VRAM budgeting. |


---

## 9. The CoNLL04 Ablation

This branch supersedes `main` (fixed markers) and `sentinel` (rolling markers): both retire. Axis 1 settled what distinguished them in favour of `sentinel`'s rolling markers, so the fixed arm has been removed rather than kept as a config key. `ABLATION_PLAN.md` holds the study design, `PORTING_PLAN.md` the staged port.

> **Format note.** The emitted format differs from `ABLATION_PLAN.md` in exactly one respect: §2 rule 1 specifies that the opening marker of the first block is omitted, and its §5 examples show that form, but **every block is marked, the first included** — a leading marker trained better than omitting it.
>
> Two further changes were tried and reverted, so the plan's form stands on both: closing the sequence with a terminal `sentinel_token(n)`, and giving the roles dedicated vocabulary (`<ent>`, `<e_type>`, ...) instead of sentinels. Each measured worse, and the second was tested both with and without the terminal marker. The roles are back on `<extra_id_95>` .. `<extra_id_99>` per §1, `<ent>` no longer exists, and nothing is added to the vocabulary.
>
> Checkpoints written while the dedicated tokens were in effect carry a `token_strs` sidecar naming them and an embedding matrix of 32104–32105 rows. They cannot be scored by this revision; `evaluate.py` raises rather than reporting meaningless numbers.

**24 runs = 8 arms × 3 seeds**, greedy-sequential over three axes, decided on `strict_f1` (mean ± std over seeds; when two arms' intervals overlap, the incumbent is carried rather than the nominal winner). Each axis's winner is carried into the next, so no later result is unconditional — they hold only under the carried-over winners, and cross-axis interactions are not measured.

| Axis | Arm | Setting |
|---|---|---|
| 1 — Markers *(settled)* | A1 | one reused marker — **removed from the code** |
| | A2 *(winner, now the only form)* | rolling `<extra_id_i>`, one per block |
| 2a — Variant | B1 | `model.variant: joint`, `graph.joint_tail_type: true` |
| | B2 | `model.variant: joint`, `graph.joint_tail_type: false` |
| 2b — Nesting | B3 | `graph.nesting: r_type` |
| | B4 | `graph.nesting: none` |
| 3 — Prompts | C1 | "Mark" wording (hand edit in `prompt.py`) |
| | C2 | `prompt.style: cot`, `graph.use_rejection: true` *(Stage 3, not yet implemented)* |

Stage 2a resolves as `winner(B1, B2)` vs the `re` baseline; Stage 2b runs B3 and B4 on that winner, with the carried `nr_type` run as the third reference point. Boundary variants are **not** ablated — they are complementary to the typed variants rather than comparable, so the boundary counterpart of the winning variant is adopted without a run.

Held constant across all 24 runs: flan-t5-base, `max_steps=2882`, batch 32 × grad-acc 1, `lr=3e-4`, `constant_with_warmup`, `warmup_steps=346`, `check_interval=29`, `bf16`, `dedup: true`, `random_prompt: false`, `random_graph: false`, `num_beams=3` at test, `early_stopping_patience=10`, `early_stopping_metric: strict_f1`. Only `train.seed` varies within an arm; only the setting under test varies between arms. `max_source_length` / `max_target_length` are re-measured per arm with `measure_lengths.py` and recorded alongside its results.

### Caveats to carry into the writeup

* **NER metrics are not comparable across the Stage-2a `re` vs `joint` comparison.** Gold differs between them by construction: `build_gold_offsets` restricts `re` / `boundary_re` gold to relation participants, while `joint` scores against every annotated entity. The relation tuples are identical, so `strict_f1` — the deciding metric — *is* comparable; but every `ner_*` and `offset_ner_*` figure moves for reasons that have nothing to do with the format under test. Report the NER numbers within a variant, never across that boundary.
* **B1 vs B2 moves the homograph ceiling.** With inline tail types, `resolve_tail_entities` no longer has to guess a tail's type by first-occurrence surface match (§2.2). Part of any `strict_f1` shift is therefore a measurement artefact rather than a format effect. CoNLL04 has 0 affected cases, which bounds the size of this — but state it.
* **C2 confounds CoT with rejection.** It is the only arm carrying `use_rejection`, so a CoT delta cannot be attributed to step-by-step framing alone. `prompt.style` and `graph.use_rejection` are orthogonal keys, so a *direct + rejection* control run is a single config flip if budget allows.
