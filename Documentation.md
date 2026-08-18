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
                            │  <extra_id_1> e2 <e_type> type <r_type> none             │
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

1. **Prompt**: The encoder prompt formats the input text with task instructions and target entity/relation schema types. By default, natural language prompts are used (e.g. `"Extract all entities of type [...] and find relations of type [...] among the extracted entities. Text: ..."`). Alternatively, a special-token schema format (`prompt: ssi`, using `<extra_id_0>` for entity types, `<extra_id_1>` for relation types, and `<extra_id_2>` before source text) is also supported.
2. **Graph (Nested Sentinel Scheme)**: Linearised target representation where each entity mention is co-located with its outgoing relations.
   - Introduced by rolling sentinels (`<extra_id_0>`, `<extra_id_1>`, ...).
   - The first relation is introduced by `<r_type> rel_type <tail> tail_text` (in `re`, also `<e_type> tail_type`).
   - Subsequent relations for the same head entity are introduced by `<nr_type> rel_type <tail> tail_text`.
   - Entities with **no outgoing relations** emit `<r_type> none`.
3. **Vocabulary Special Tokens**: Explicit special tokens `<e_type>`, `<r_type>`, `<nr_type>`, `<tail>`, and `<null>` are added to the tokenizer vocabulary. Sentinel tokens (`<extra_id_0>` ... `<extra_id_99>`) are natively present in T5/Flan-T5.
4. **Supported Model Variants**:
   * **`joint`**: Joint entity recognition and relation extraction. All entity mentions get their own block (`<extra_id_i> head [<e_type> type]`). Entities without outgoing relations emit `<r_type> none`. Tail entities are referenced by surface text (no tail types).
   * **`boundary_joint`**: Joint entity span boundary extraction (no entity types) and relation extraction. All entity mentions get their own block (`<extra_id_i> head`). Entities without outgoing relations emit `<r_type> none`.
   * **`re`**: Relation extraction given typed head and tail entities. **Only entities that act as a head in at least one relation get their own block** (`<extra_id_i> head <e_type> head_type <r_type> rel <tail> tail <e_type> tail_type`). Non-participating entities and tail-only entities are omitted as head blocks.
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
  <extra_id_0> Barack Obama <e_type> person <r_type> place of birth <tail> Honolulu <nr_type> president of <tail> United States <extra_id_1> Honolulu <e_type> city <r_type> located in <tail> United States <extra_id_2> United States <e_type> country <r_type> none
  ```

#### 2. `boundary_joint`
* **Task**: Joint entity span boundary extraction (without entity types) + relation extraction across all entities.
* **Encoder Input (Natural Prompt)**:
  ```text
  Extract all entities and find relations of type [founded, killed, located in, place of birth, president of] among the extracted entities. Text: Barack Obama was born in Honolulu and served as the president of the United States
  ```
* **Decoder Output (Nested Graph)**:
  ```text
  <extra_id_0> Barack Obama <r_type> place of birth <tail> Honolulu <nr_type> president of <tail> United States <extra_id_1> Honolulu <r_type> located in <tail> United States <extra_id_2> United States <r_type> none
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

##### `S2GTokens(variant: str, use_rejection: bool = False, prompt: str = 'natural')`
* **`base_tok_map`**: Maps active tokens per variant (`re`, `boundary_re`, `boundary_joint`, `joint`).
* **`self.active_tokens`**: Active token set for configured variant. Adds `'null'` if `use_rejection=True`.
* **`self.all_tokens`**: List of vocabulary special token strings (`['<e_type>', '<nr_type>', '<tail>', '<null>']`).
* **`self.sentinel_token(idx)`**: Returns `<extra_id_{idx}>` for rolling entity demarcation.

##### `add_special_tokens_to_tokenizer(tokenizer, tokens: S2GTokens, model=None, warm_start: bool = True) -> int`
* Adds `tokens.all_tokens` to HuggingFace tokenizer via `add_special_tokens({'additional_special_tokens': ...})`.
* If `model` is provided and tokens were added:
  * Sets `model.config.tie_word_embeddings = False`.
  * Resizes model token embeddings via `model.resize_token_embeddings(len(tokenizer))`.
* If `warm_start=True`: Initializes new token embeddings by averaging input/output embeddings of natural language phrases:
  * `'e_type'` $\rightarrow$ `"entity type: "`
  * `'nr_type'` $\rightarrow$ `"next relation: "`
  * `'tail'` $\rightarrow$ `"object: "`
  * `'null'` $\rightarrow$ `"not found: "`

---

### 2.2. `s2g/linearisation/graph.py`

#### Purpose
Handles nested graph building (each entity mention co-located with its outgoing relations, `<r_type> none` for relation-less entities in joint variants, and non-head entity omission for RE variants) and unified state machine parsing.

#### Data Structures & Types
* `EntityBlock`: `Dict[str, Any]` containing `'text'`, `'type'` (optional), and `'relations'` (`List[Dict[str, Any]]` where each relation is `{'type': rel_type, 'tail_text': tail_text, 'tail_type': tail_type}`).
* `Triplet`: `Tuple[str, str, str]` $\rightarrow$ `(head_text, rel_type, tail_text)`.
* `RejectedItem`: `str` label representing a rejected (null) schema type.

#### Key Functions

##### `build_graph(ent_blocks, variant, tokens, use_nesting=True, random_graph=False, use_rejection=False, rejected_ent_types=None, rejected_rel_types=None) -> str`
* Constructs linearised nested Graph target string:
  * **Variant `joint` / `boundary_joint`**:
    - Emits all entities: `<extra_id_i> head [<e_type> type] <r_type> rel1 <tail> tail1 [<nr_type> rel2 <tail> tail2 ...]`
    - If an entity has no outgoing relations: emits `<r_type> none`.
  * **Variant `re` / `boundary_re`**:
    - **Skips non-head entities** (only entities with at least one outgoing relation get a block).
    - Emits: `<extra_id_i> head [<e_type> head_type] <r_type> rel1 <tail> tail1 [<e_type> tail1_type] [<nr_type> ...]`

##### `parse_graph(text: str, tok: S2GTokens, use_nesting: bool = True) -> Tuple[List[EntityBlock], List[RejectedItem]]`
* State-machine parser:
  1. Tracks the active head entity via `<extra_id_i>`, parses its mention text and optional `<e_type>`.
  2. Reads relations introduced by `<r_type>` / `<nr_type>`, tail text/type after `<tail>`, and handles `<r_type> none`.
  3. Reconstructs structured `EntityBlock` list and extracts evaluation triplets.

---

## 3. Module: `s2g.data`

The `data` package handles dataset loading, memory-mapped JSONL indexing, dynamic schema sampling, batch collation, and raw dataset preprocessing.

---

## 4. Module: `s2g.evaluation`

The `evaluation` package provides corpus-level and per-type macro metric calculation functions as well as the streaming `S2GEvaluator`.

---

## 5. Summary Matrix of Module Responsibilities

| Package / File | Primary Responsibility |
|---|---|
| `s2g.linearisation.special_tokens` | Token registry for `<e_type>`, `<r_type>`, `<nr_type>`, `<tail>`, `<null>`, and sentinel helper `S2GTokens.sentinel_token(idx)`. |
| `s2g.linearisation.graph` | Graph construction with `<tail>` token (`build_graph`), state-machine parsing (`parse_graph`), block filtering (`organise_filter_and_block`). |
| `s2g.linearisation.prompt` | Encoder input prompt construction for natural language instructions and SSI schema. |
| `s2g.evaluation.metrics` | Micro and per-type macro PRF metric calculation using text-and-type tuple matching. |
| `s2g.evaluation.evaluator` | Tensor-direct batch decoding, streaming evaluation harness, and DataLoader decoder. |
