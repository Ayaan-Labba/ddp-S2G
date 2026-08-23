# Sentence-to-Graph (S2G): Automatic Knowledge Graph Generation from Unstructured Text

A seq2seq approach to joint entity and relation extraction framed as a text-to-text problem. The encoder receives a source sentence prefixed by a natural language instruction carrying the entity and relation type schema. The decoder generates a linearised **Sentence-to-Graph** representation using rolling sentinel tokens (`<extra_id_0>`, `<extra_id_1>`, ...) and dedicated vocabulary special tokens (`<e_type>`, `<r_type>`, `<nr_type>`, `<tail>`, `<null>`).

Built on **Flan-T5 Base** (~250M parameters), pre-trained on [REBEL](https://huggingface.co/datasets/Babelscape/rebel-dataset), and fine-tuned on CoNLL04, NYT-multi, and SciERC.

*For deep architectural specifications, linearisation grammars, and evaluation details, please refer to `Documentation.md`.*

---

## Project Structure

```text
configs/
├── pretrain.yaml              # Pre-training hyperparameters
├── finetune.yaml              # Benchmark fine-tuning defaults
├── evaluate.yaml              # Evaluation decoding configurations
├── data/                      # Label prettification maps per corpus
└── variants/                  # Ready-to-run configs per (variant, dataset)
    └── joint | boundary_joint | re | boundary_re
s2g/
├── linearisation/             # Prompt builder, nested graph builder & state-machine parser
│   ├── special_tokens.py      # Vocabulary special tokens (<e_type>, <r_type>, <nr_type>, <tail>, <null>)
│   ├── graph.py               # Block building, nested graph builder, FSM parser
│   └── prompt.py              # Natural language instruction prompt builders
├── data/                      # Memory-mapped datasets, collators, corpus preprocessors
├── evaluation/                # Text and offset metrics, gold construction, streaming evaluator
│   ├── metrics.py             # Micro + REBEL-style per-type macro PRF
│   ├── offsets.py             # Projects predicted mentions onto source offsets
│   ├── gold.py                # Gold built directly from the preprocessed annotations
│   └── evaluator.py           # Tensor-direct decoding and streaming evaluation loop
├── training/                  # Custom Seq2SeqTrainer and callbacks
└── scripts/                   # Entry-point scripts
    ├── train.py               # Unified script for fine-tuning and pre-training
    ├── evaluate.py            # Standalone evaluation
    ├── measure_lengths.py     # Calculates 99th-percentile buffer sizes
    ├── measure_vram.py        # GPU VRAM batch size estimator
    └── config_utils.py        # OmegaConf config loader
data/                          # Preprocessed corpora (conll04, nyt, scierc, scierc_doc)
requirements.txt
Documentation.md
README.md
```

---

## Linearisation & Graph Formats

Each entity block is introduced by a rolling sentinel token (`<extra_id_0>`, `<extra_id_1>`, ...). Relation targets feature the static `<tail>` special token directly followed by tail entity text. Entities without outgoing relations in joint variants simply omit relation tokens. RE variants omit non-head entities.

### Example
* **Text**: *"Barack Obama was born in Honolulu and served as the president of the United States"*
* **Entities**: `Barack Obama` (person), `Honolulu` (city), `United States` (country)
* **Relations**: `(Barack Obama, place of birth, Honolulu)`, `(Barack Obama, president of, United States)`, `(Honolulu, located in, United States)`

#### 1. `joint` (Nested)
```text
<extra_id_0> Barack Obama <e_type> person <r_type> place of birth <tail> Honolulu <nr_type> president of <tail> United States <extra_id_1> Honolulu <e_type> city <r_type> located in <tail> United States <extra_id_2> United States <e_type> country
```

#### 2. `boundary_joint` (Nested)
```text
<extra_id_0> Barack Obama <r_type> place of birth <tail> Honolulu <nr_type> president of <tail> United States <extra_id_1> Honolulu <r_type> located in <tail> United States <extra_id_2> United States
```

#### 3. `re`
```text
<extra_id_0> Barack Obama <e_type> person <r_type> place of birth <tail> Honolulu <e_type> city <nr_type> president of <tail> United States <e_type> country <extra_id_1> Honolulu <e_type> city <r_type> located in <tail> United States <e_type> country
```

#### 4. `boundary_re`
```text
<extra_id_0> Barack Obama <r_type> place of birth <tail> Honolulu <nr_type> president of <tail> United States <extra_id_1> Honolulu <r_type> located in <tail> United States
```

---

## Setup

```bash
pip install -r requirements.txt
```

NLTK data is only needed by the (currently unused) `scripts/inference.py`:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Quickstart: Training & Evaluation

### Step 1 — Preprocess Benchmark Dataset

```bash
python -m s2g.data.preprocess_conll04 \
    --input_dir  data/raw/conll04 \
    --output_dir data/conll04 \
    --config_map configs/data/conll04.yaml
```

Writes `train.jsonl` / `val.jsonl` / `test.jsonl` plus `entity.schema` and `relation.schema` (derived from the train split). Entity annotations are keyed by **offset**: the same span recorded twice is one entity, while the same text at two different offsets is two entities and both are kept.

### Step 2 — Train Model

Either use a ready-made variant config:

```bash
python -m s2g.scripts.train --config configs/variants/joint/conll04.yaml
```

Or start from the generic defaults and override on the command line:

```bash
torchrun --nproc_per_node=4 -m s2g.scripts.train \
    --config configs/finetune.yaml \
    model.variant=joint \
    data.data_dir=data/conll04 \
    data.rel_schema=data/conll04/relation.schema \
    data.ent_schema=data/conll04/entity.schema \
    data.output_dir=outputs/finetune/conll04_joint
```

Training ends by saving `best_model/` and running evaluation on val and test.

### Step 3 — Evaluate

```bash
python -m s2g.scripts.evaluate \
    --config configs/evaluate.yaml \
    model.pretrained_checkpoint=outputs/finetune/conll04_joint/best_model \
    data.data_dir=data/conll04 \
    data.rel_schema=data/conll04/relation.schema \
    data.ent_schema=data/conll04/entity.schema \
    data.output_dir=outputs/finetune/conll04_joint/eval \
    evaluation.split=test
```

Format-critical settings (`variant`, `graph.dedup`, `graph.use_nesting`, `graph.use_rejection`, `prompt.type`, schema caps) are read from `best_model/s2g_format.json` and override the evaluation config, so a checkpoint can never be scored against a different linearisation than it was trained on.

Outputs `{split}_metrics.json` and `{split}_results.jsonl` (one record per instance: source text, encoder input, raw prediction, parsed blocks, gold blocks, and the predicted offset map).

---

## Tests

```bash
python -m unittest discover -s tests -t .
```

48 tests over the linearisation round trip, offset projection, gold construction and scoring, across all four variants and both `dedup` settings. Stdlib `unittest` only — no extra dependencies; pytest can collect them too if you add it.

---

## Deduplication (`graph.dedup`)

Benchmarks annotate entities by offset, but the decoder emits text, so repeated mentions need a policy. `graph.dedup` sets it for **target construction only** — parsing never deduplicates.

| Setting | Effect |
|---|---|
| `true` (default) | Mentions collapse on `(text, type)`; relations collapse on the full quintuple |
| `false` | Every mention gets its own block; every relation is kept |

The key is `(text, type)`, not text alone, so **homographs** — `Washington` the person versus `Washington` the location — are never merged.

---

## Metrics Computed

Every evaluation reports two parallel tracks. Gold is taken directly from the preprocessed annotations, not by parsing the model's target format.

**Text-based** — matched on surface strings:

* **NER Boundary F1:** `head_text` (micro only).
* **NER Strict F1:** `(head_text, head_type)`.
* **Relation Boundary F1:** `(head_text, rel_type, tail_text)`.
* **Relation Strict F1:** `(head_text, head_type, rel_type, tail_text, tail_type)`.

**Offset-based** (`offset_` prefix) — each predicted mention is located in the source tokens, every match counts as a distinct prediction, and relations expand over the head x tail cross product:

* `offset_ner_boundary_f1`, `offset_ner_f1`, `offset_boundary_f1`, `offset_strict_f1`.

This is what lets a single emission earn credit for two gold annotations when a mention legitimately repeats in the sentence. Predicted text occurring nowhere in the source is given a negative sentinel offset — it can never match gold but still counts against precision, and stays visible in `{split}_results.jsonl` for inspection.

See `Documentation.md` §4.5 for a worked example on a real CoNLL04 sentence, and §2.2 / §4.2 for two ceilings that homographs impose on strict and offset scoring.

Each metric also has a REBEL-style per-type macro variant (`macro_*`, `macro_offset_*`) wherever there is a type to group by; boundary NER is micro-only. The boundary variants emit no `strict_f1`, so their configs must set `validation.early_stopping_metric: boundary_f1`.