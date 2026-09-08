#!/bin/bash
# Runs one ablation arm at 5 seeds, one after the other.
#
#   scripts/ablations.sh axis-3/cot c3-cot prompt.style=cot graph.use_rejection=true
#
# Arg 1 is the arm's path under outputs/ablations/, arg 2 the W&B run-name prefix.
# Anything after them is passed straight through as config overrides, so the arm's
# setting can be flipped here rather than by editing configs/ablation.yaml.

set -e

ARM="$1"
NAME="$2"
shift 2

for SEED in 0 1 2 3 4; do
    echo "=== $NAME seed $SEED ==="
    python -m s2g.scripts.train \
        --config configs/ablation.yaml \
        train.seed="$SEED" \
        data.output_dir="outputs/ablations/$ARM/seed-$SEED" \
        wandb.run_name="$NAME-seed-$SEED" \
        "$@"
done
