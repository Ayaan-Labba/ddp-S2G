# Sentence-to-Graph (S2G): Automatic Knowledge Graph Generation from Unstructured Text

A seq2seq approach to joint entity and relation extraction framed as a text-to-text problem. The encoder receives a source sentence prefixed by a **Schema-Structured Input (SSI)** that enumerates the entity and relation types in scope. The decoder generates a flat **Structured Extraction Language (SEL)** string encoding all entities, their types, pairwise relations, and explicit rejections of absent schema types. At test time a task-specific finite-state machine (FSM) constrains decoding to produce only valid SEL expressions.

Built on **Flan-T5 Base** (~250M parameters), pre-trained on [REBEL](https://huggingface.co/datasets/Babelscape/rebel-dataset) (Experiment 3), and fine-tuned on CoNLL04, NYT-multi, and SciERC (Experiment 1).

---

## Architecture

### Task Settings

Five task settings are organised across two independently trained models:

| Model | Tasks | Encoder input | Decoder output |
|---|---|---|---|
| **Pipeline** | Boundary | `<bound> raw text` | Entity spans |
| | NER | `<type> T₁ … <ner> boundary-augmented text` | Typed entity spans |
| | RE | `<rel> R₁ … <re> entity+type-augmented text` | Entity–relation chains |
| **Joint** | Joint | `<rel> R₁ … <joint> raw text` | Entity–relation chains |
| | Joint+ | `<type> T₁ … <rel> R₁ … <joint+> raw text` | Typed entity–relation chains |

The Pipeline model is evaluated sequentially — Boundary predictions augment the NER encoder input; NER predictions augment the RE encoder input. The Joint model's two tasks are independent.

### SEL Grammars

Each task has a distinct SEL grammar. For example:

```
# Boundary
<ent> Barack Obama </ent> <ent> Honolulu </ent>

# NER  (+ null block for absent types)
<ent> Barack Obama <type> person </ent> <ent> Honolulu <type> city </ent>
<null> <type> organization <type> artifact

# RE
<ent> Barack Obama <rel> place of birth <tail> Honolulu
                   <rel> president of   <tail> United States </ent>
<null> <rel> founded <rel> killed

# Joint+
<ent> Barack Obama <type> person <rel> place of birth <tail> Honolulu </ent>
<ent> Honolulu     <type> city                                         </ent>
<null> <type> organization <rel> founded
```

### SSI — Budget Mode

All gold-positive types for an instance are always included in the SSI. Remaining budget (`max_ent_types_in_prompt`, `max_rel_types_in_prompt`) is filled with uniformly sampled negatives from the schema. This ensures the decoder consistently sees both relevant and irrelevant type labels, making the rejection-block generation well-calibrated.

---

## Project Structure

```
configs/
├── pretrain.yaml              # REBEL pre-training hyperparameters
├── finetune.yaml              # Benchmark fine-tuning defaults
└── sweep_pretrain.yaml        # W&B sweep definition
vanilla_s2g/
├── linearisation/             # Core SEL/SSI module
│   ├── special_tokens.py      #   Two token registries (Pipeline, Joint)
│   ├── ssi.py                 #   Task-specific encoder input builders + text augmentation
│   └── sel.py                 #   Task-specific SEL construction and parsing
├── data/                      # Data loading and preprocessing
│   ├── dataset.py             #   Memory-mapped JSONL dataset
│   ├── collator.py            #   Multi-task budget-mode collator
│   ├── preprocess_rebel.py    #   REBEL → S2G JSONL
│   ├── preprocess_conll04.py  #   CoNLL04 → S2G JSONL
│   ├── preprocess_nyt_multi.py#   NYT-multi → S2G JSONL
│   └── preprocess_scierc.py   #   SciERC → S2G JSONL
├── model/                     # Model wrapper and constraint decoder
│   ├── model.py               #   S2GModel (variant-aware Flan-T5 wrapper)
│   └── constraint_decoder.py  #   Task-aware FSM logits processor
├── training/                  # Shared training infrastructure
│   └── trainer.py             #   S2GTrainer (multi-task loss + pipeline evaluate)
├── evaluation/                # Metrics and callbacks
│   ├── metrics.py             #   Micro + macro P/R/F1 at four granularity levels
│   └── callbacks.py           #   W&B sample table, step tracking, periodic checkpoints
└── scripts/                   # Entry-point scripts
    ├── config_utils.py        #   OmegaConf config loader
    ├── pretrain.py            #   REBEL pre-training (Experiment 3)
    ├── finetune.py            #   Benchmark fine-tuning (Experiment 1)
    ├── evaluate.py            #   Standalone evaluation with optional FSM decoding
    └── inference.py           #   Interactive and batch extraction
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

## Experiment 1 — Benchmark Fine-tuning

Experiment 1 fine-tunes Flan-T5 Base directly on CoNLL04, NYT-multi, and SciERC using the ablation-optimal configuration: Boundary + NER + RE (Pipeline model) and Joint + Joint+ (Joint model).

### Step 1 — Preprocess a Benchmark Dataset

```bash
# CoNLL04
python -m vanilla_s2g.data.preprocess_conll04 \
    --input_dir  data/raw/conll04 \
    --output_dir data/conll04

# NYT-multi
python -m vanilla_s2g.data.preprocess_nyt_multi \
    --input_dir  data/raw/nyt_multi \
    --output_dir data/nyt_multi

# SciERC
python -m vanilla_s2g.data.preprocess_scierc \
    --input_dir  data/raw/scierc \
    --output_dir data/scierc
```

Each script expects `train.json`, `dev.json`, `test.json` in the input directory and writes `train.jsonl`, `val.jsonl`, `test.jsonl`, `entity.schema`, and `relation.schema` to the output directory.

### Step 2 — Fine-tune the Pipeline Model

```bash
python -m vanilla_s2g.scripts.finetune \
    --config configs/finetune.yaml \
    model.model_variant=pipeline \
    data.data_dir=data/conll04 \
    data.schema_file=data/conll04/relation.schema \
    data.entity_schema_file=data/conll04/entity.schema \
    data.output_dir=outputs/finetune/conll04_pipeline \
    train.max_steps=10000 \
    optimizer.lr=1e-4 \
    scheduler.warmup_steps=500
```

### Step 3 — Fine-tune the Joint Model

```bash
python -m vanilla_s2g.scripts.finetune \
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

For multi-GPU training (recommended for NYT-multi and SciERC):

```bash
torchrun --nproc_per_node=4 -m vanilla_s2g.scripts.finetune \
    --config configs/finetune.yaml \
    model.model_variant=pipeline \
    data.data_dir=data/nyt_multi \
    ...
```

### Step 4 — Evaluate on the Test Set

```bash
python -m vanilla_s2g.scripts.evaluate \
    --config configs/evaluate.yaml \
    model.pretrained_checkpoint=outputs/finetune/conll04_pipeline/best_model \
    data.data_dir=data/conll04 \
    data.schema_file=data/conll04/relation.schema \
    data.entity_schema_file=data/conll04/entity.schema \
    data.output_dir=outputs/finetune/conll04_pipeline/eval \
    evaluation.split=test \
    generation.constraint_decoding=true
```

Output files written to `data.output_dir`:

| File | Content |
|---|---|
| `test_results.jsonl` | Per-instance structured predictions (all tasks) |
| `test_metrics.json` | All micro and macro F1 metrics |

### Step 5 — Resume After Interruption

```bash
python -m vanilla_s2g.scripts.finetune \
    --config configs/finetune.yaml \
    ... \
    checkpoint.resume_from=outputs/finetune/conll04_pipeline/checkpoint-last
```

---

## Experiment 3 — REBEL Pre-training

Pre-training on REBEL initialises the model on large-scale open-domain relation extraction before benchmark fine-tuning.

### Step 1 — Preprocess REBEL

```bash
python -m vanilla_s2g.data.preprocess_rebel \
    --output_dir data/rebel \
    --top_k 220
```

### Step 2 — Pre-train

```bash
torchrun --nproc_per_node=4 -m vanilla_s2g.scripts.pretrain \
    --config configs/pretrain.yaml \
    model.model_variant=pipeline \
    data.data_dir=data/rebel \
    data.schema_file=data/rebel/relation.schema \
    data.output_dir=outputs/pretrain/pipeline
```

### Step 3 — Fine-tune from the Pre-trained Checkpoint

```bash
python -m vanilla_s2g.scripts.finetune \
    --config configs/finetune.yaml \
    model.pretrained_checkpoint=outputs/pretrain/pipeline/best_model \
    model.model_variant=pipeline \
    data.data_dir=data/conll04 \
    ...
```

---

## Inference

### Interactive Mode

```bash
python -m vanilla_s2g.scripts.inference \
    --checkpoint outputs/finetune/conll04_pipeline/best_model \
    --schema_file data/conll04/relation.schema \
    --entity_schema_file data/conll04/entity.schema

>>> Barack Obama was born in Honolulu, Hawaii.

  Boundary spans: ['Barack Obama', 'Honolulu', 'Hawaii']
  NER entities:   [{'text': 'Barack Obama', 'type': 'Peop'}, ...]
  Triplets:
    (Barack Obama, Live_In, Honolulu)
```

### Batch Mode

```bash
python -m vanilla_s2g.scripts.inference \
    --checkpoint outputs/finetune/conll04_pipeline/best_model \
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

All experiments: AdamW (`β₁=0.9`, `β₂=0.999`, `ε=1e-8`, `wd=0`), cosine schedule, bf16, seed 0, effective batch size 32.

---

## Metrics

`S2GTrainer.evaluate()` and `evaluate.py` compute micro (corpus-level) and macro (instance-average) variants of:

| Metric | Measures |
|---|---|
| **NER Boundary F1** | Entity text span match (no type) |
| **NER Strict F1** | Entity text + type match |
| **Relation Boundary F1** | `(head, rel_type, tail)` triplet match |
| **Relation Strict F1** | `(head, head_type, rel_type, tail, tail_type)` quintuple match |

Metrics are prefixed by task in the returned dict. For the Pipeline model: `boundary_*`, `ner_*`, `re_*`. For the Joint model: `joint_*`, `joint_plus_*`. Early stopping uses `re_rel_boundary_f1` (Pipeline) or `joint_rel_boundary_f1` (Joint) by default.

---

## W&B Sweeps

```bash
# Create a sweep from the pre-training definition.
wandb sweep configs/sweep_pretrain.yaml

# Launch an agent (repeat on multiple machines for parallel search).
wandb agent <sweep_id>
```

---

## Monitoring

| Signal | Frequency |
|---|---|
| Training loss (per-task + total) | Every 50 steps |
| Learning rate | Every 50 steps |
| Validation metrics (all tasks) | Every `check_interval` steps |
| Train-eval subsample metrics | Every `check_interval` steps |
| Sample prediction table (W&B) | Every `sample_generation_interval` steps |
| Safety-net checkpoint | Every `every_n_steps` steps |