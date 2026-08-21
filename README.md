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
└── evaluate.yaml              # Evaluation decoding configurations
s2g/
├── linearisation/             # Prompt builder, nested graph builder & state-machine parser
│   ├── special_tokens.py      # Vocabulary special tokens (<e_type>, <r_type>, <nr_type>, <tail>, <null>)
│   ├── graph.py               # Nested graph builder and FSM parser
│   └── prompt.py              # Natural language instruction prompt builders
├── data/                      # Memory-mapped datasets and collators
├── evaluation/                # Text-validated index-bound metrics and tensor-direct evaluator
├── training/                  # Custom Seq2SeqTrainer
└── scripts/                   # Entry-point scripts
    ├── train.py               # Unified script for fine-tuning and pre-training
    ├── evaluate.py            # Standalone evaluation
    ├── measure_lengths.py     # Calculates 99th-percentile buffer sizes
    ├── measure_vram.py        # GPU VRAM batch size estimator
    └── config_utils.py        # OmegaConf config loader
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
# Install dependencies.
pip install -r requirements.txt

# Download NLTK tokeniser data (required for entity span alignment).
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Quickstart: Training & Evaluation

### Step 1 — Preprocess Benchmark Dataset

```bash
python -m s2g.data.preprocess_conll04 \
    --input_dir  data/raw/conll04 \
    --output_dir data/conll04
```

### Step 2 — Train Model

```bash
torchrun --nproc_per_node=4 -m s2g.scripts.train \
    --config configs/finetune.yaml \
    model.variant=joint \
    data.data_dir=data/conll04 \
    data.rel_schema=data/conll04/relation.schema \
    data.ent_schema=data/conll04/entity.schema \
    data.output_dir=outputs/finetune/conll04_joint
```

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

---

## Metrics Computed

`evaluate.py` computes corpus-level (micro) PRF for all four metrics below, plus
REBEL-style per-type macro PRF for every metric that has a type to group by
(NER Strict, Relation Boundary, Relation Strict — boundary NER has no type to
group by, so it is micro-only):
* **NER Boundary F1:** `head_text` entity span match (micro only).
* **NER Strict F1:** `(head_text, head_type)` entity span and type match.
* **Relation Boundary F1:** `(head_text, rel_type, tail_text)` triplet match.
* **Relation Strict F1:** `(head_text, head_type, rel_type, tail_text, tail_type)` quintuple match.