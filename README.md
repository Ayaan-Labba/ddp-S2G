# Sentence-to-Graph (S2G): Automatic Knowledge Graph Generation from Unstructured Text

A seq2seq approach to joint entity and relation extraction framed as a text-to-text problem. The encoder receives a source sentence prefixed by a **Schema-Structured Input (SSI)** that enumerates the entity and relation types in scope. The decoder generates a flat **Structured Extraction Language (SEL)** string encoding all entities, their types, pairwise relations, and explicit rejections of absent schema types. At test time, a task-specific finite-state machine (FSM) constrains decoding to produce only valid SEL expressions.

Built on **Flan-T5 Base** (~250M parameters), pre-trained on [REBEL](https://huggingface.co/datasets/Babelscape/rebel-dataset), and fine-tuned on CoNLL04, NYT-multi, and SciERC.

*For deep architectural specifications, SEL grammars, and ablation study details, please refer to the `S2G Documentation.md` and `Research Plan.md`.*

---

## Project Structure

```text
configs/
├── pretrain.yaml              # Pre-training hyperparameters
├── finetune.yaml              # Benchmark fine-tuning defaults
└── evaluate.yaml              # Evaluation decoding configurations
s2g/
├── linearisation/             # Core SEL/SSI module
├── data/                      # Memory-mapped datasets and budget-mode collators
├── model/                     # Flan-T5 wrappers and constraint FSM decoder
├── evaluation/                # Metrics and W&B generation callbacks
├── training/                  # Custom Seq2SeqTrainer for multi-task loss
└── scripts/                   # Entry-point scripts
    ├── train.py               # Unified script for fine-tuning and pre-training
    ├── evaluate.py            # Standalone evaluation with optional FSM decoding
    ├── inference.py           # Interactive and batch extraction
    ├── measure_lengths.py     # Calculates 99th-percentile buffer sizes
    └── config_utils.py        # OmegaConf config loader
requirements.txt
README.md
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

## Quickstart: Fine-Tuning

### Step 1 — Preprocess a Benchmark Dataset

```bash
# Example: CoNLL04
python -m s2g.data.preprocess_conll04 \
    --input_dir  data/raw/conll04 \
    --output_dir data/conll04
```

### Step 2 — Train the Model

Use the unified `train.py` script. The model variant is determined by the config overrides.

Available `model.model_variant` options:
* **`joint`**: Jointly predicts entity spans, entity types, and typed relations.
* **`boundary_joint`**: Jointly predicts entity spans and relations between them (no entity types).
* **`re`**: Relation extraction with typed head/tail entities.
* **`boundary_re`**: Relation extraction between entity spans (no entity types).

Per-variant × per-dataset configs live under `configs/tasks/<variant>/<dataset>.yaml`.

```bash
# Multi-GPU training using Torchrun
torchrun --nproc_per_node=4 -m s2g.scripts.train \
    --config configs/finetune.yaml \
    model.model_variant=joint \
    data.data_dir=data/conll04 \
    data.schema_file=data/conll04/relation.schema \
    data.entity_schema_file=data/conll04/entity.schema \
    data.output_dir=outputs/finetune/conll04_joint \
    train.max_steps=10000 \
    optimizer.lr=1e-4 \
    scheduler.warmup_steps=500
```

### Step 3 — Evaluate with FSM Constraint Decoding

```bash
python -m s2g.scripts.evaluate \
    --config configs/evaluate.yaml \
    model.pretrained_checkpoint=outputs/finetune/conll04_joint/best_model \
    data.data_dir=data/conll04 \
    data.schema_file=data/conll04/relation.schema \
    data.entity_schema_file=data/conll04/entity.schema \
    data.output_dir=outputs/finetune/conll04_joint/eval \
    evaluation.split=test \
    generation.constraint_decoding=true
```

---

## Inference Mode

### Interactive Inference

```bash
python -m s2g.scripts.inference \
    --checkpoint outputs/finetune/conll04_joint/best_model \
    --schema_file data/conll04/relation.schema \
    --entity_schema_file data/conll04/entity.schema

>>> Barack Obama was born in Honolulu, Hawaii.

  Boundary spans: ['Barack Obama', 'Honolulu', 'Hawaii']
  NER entities:   [{'text': 'Barack Obama', 'type': 'Peop'}, ...]
  Triplets:
    (Barack Obama, Live_In, Honolulu)
```

### Batch Extraction

```bash
python -m s2g.scripts.inference \
    --checkpoint outputs/finetune/conll04_joint/best_model \
    --schema_file data/conll04/relation.schema \
    --entity_schema_file data/conll04/entity.schema \
    --input_file sentences.txt \
    --output_file predictions.jsonl \
    --constraint_decoding true
```

---

## Dataset-Specific Hyperparameters

Starting-point suggestions; tune via W&B sweeps.

| Dataset | max\_steps | lr | warmup | max\_src | max\_tgt |
|---|---|---|---|---|---|
| CoNLL04 | 10 000 | 1e-4 | 500 | 256 | 128 |
| NYT-multi | 60 000 | 5e-5 | 2 000 | 384 | 200 |
| SciERC | 20 000 | 5e-5 | 1 000 | 512 | 256 |

*All experiments: AdamW (`β₁=0.9`, `β₂=0.999`, `ε=1e-8`, `wd=0`), cosine schedule, bf16, seed 0, effective batch size 32.*

---

## Metrics Computed

`evaluate.py` automatically computes micro (corpus-level) and macro (instance-average) variants of:
* **NER Boundary F1:** Entity text span match (no type). Calculated for `joint`.
* **NER Strict F1:** Entity text + type match. Calculated for `joint`.
* **Relation Boundary F1:** `(head, rel_type, tail)` triplet match. Calculated for `re`, `boundary_re`, `boundary_joint`, and `joint`.
* **Relation Strict F1:** `(head, head_type, rel_type, tail, tail_type)` quintuple match. Calculated for `re` and `joint`.