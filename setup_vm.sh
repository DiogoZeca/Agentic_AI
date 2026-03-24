#!/usr/bin/env bash
# setup_vm.sh — Prepare a Linux VM for a 24h training run
#
# Usage:
#   1. Copy this script and your data to the VM:
#        rsync -avz --partial --inplace AIModel/ user@vm-host:~/spike/AIModel/
#        rsync -avz --partial --inplace AIModel/data/cluster_cpu_data.csv user@vm-host:~/spike/AIModel/data/
#
#   2. SSH into the VM and run this script:
#        ssh user@vm-host
#        cd ~/spike && bash setup_vm.sh
#
#   3. The script creates a tmux session named "train".
#      Attach any time with:  tmux attach -t train
#      Detach safely with:    Ctrl-B  then  D
#
# The training log is written to AIModel/data/rerun_log.txt (unbuffered).
# ------------------------------------------------------------------

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/AIModel"
VENV="$WORKDIR/.venv"
LOG="$WORKDIR/data/rerun_log.txt"

echo "=== Spike Prediction VM Setup ==="
echo "Working directory: $WORKDIR"

# ── 1. System dependencies ────────────────────────────────────────────────────
echo ""
echo "[1/4] Installing system dependencies..."
sudo apt-get update -q
sudo apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-pip tmux

# ── 2. Python virtual environment ────────────────────────────────────────────
echo ""
echo "[2/4] Creating Python virtual environment..."
cd "$WORKDIR"
python3.12 -m venv .venv
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r requirements-train.txt
echo "     Dependencies installed."

# ── 3. Validate data file ─────────────────────────────────────────────────────
echo ""
echo "[3/4] Validating input data..."
if [ ! -f "data/cluster_cpu_data.csv" ]; then
    echo "ERROR: data/cluster_cpu_data.csv not found."
    echo "       Transfer it first:"
    echo "         rsync -avz --partial --inplace data/cluster_cpu_data.csv user@vm-host:~/spike/AIModel/data/"
    exit 1
fi
ROW_COUNT=$("$VENV/bin/python3" -c "import pandas as pd; df = pd.read_csv('data/cluster_cpu_data.csv', nrows=5); print(len(df.columns), 'columns OK')")
echo "     $ROW_COUNT"

# ── 4. Launch training in tmux ────────────────────────────────────────────────
echo ""
echo "[4/4] Starting training session in tmux (session: train)..."
mkdir -p data/full_run

tmux new-session -d -s train -x 220 -y 50 \
    "cd $WORKDIR && $VENV/bin/python3 -u train_spike_classifier.py \
        --data-path data/cluster_cpu_data.csv \
        --artifacts-dir data/full_run \
        --from-step 1 --force \
        --tune-hyperparams --optuna-trials 30 \
        2>&1 | tee $LOG; echo '=== TRAINING COMPLETE ==='; read"

echo ""
echo "=== Done ==="
echo ""
echo "Training is running in tmux session 'train'."
echo "  Attach:  tmux attach -t train"
echo "  Detach:  Ctrl-B then D  (session keeps running)"
echo "  Log:     tail -f $LOG"
echo ""
