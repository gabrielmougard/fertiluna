#!/usr/bin/env bash
# Remote pipeline run on a vast.ai instance: clone, install, render datasets,
# train, export. Driven by vast_run.py; not meant to be invoked by hand.
#
# Required env (set by the orchestrator via `ssh ... env A=1 B=2 bash`):
#   REPO_URL          git@github.com:owner/repo.git or https://...
#   REPO_REF          branch / tag / commit SHA to check out
#   TRAIN_N           train dataset size (e.g. 200000)
#   VAL_N             val dataset size   (e.g. 20000)
#   TRAIN_SEED        seed for train set
#   VAL_SEED          seed for val set
#   STYLE             generic | premom | blend
#   WIDTH             model width (e.g. 3.0)
#   EPOCHS
#   VERSION           artifact suffix (e.g. v1)
#   WORKERS           dataset render workers; default = nproc (matplotlib-bound,
#                     scales linearly)
#
# Optional overrides — leave unset to let the training script auto-pick from
# the detected hardware (recommended):
#   BATCH_SIZE        training batch size (auto-picked from VRAM × width).
#   TRAIN_WORKERS     DataLoader workers (auto-picked, capped at 16).
#   LR                learning rate     (auto-scaled linearly from batch_size).
set -euo pipefail

log() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }

: "${REPO_URL:?REPO_URL is required}"
: "${REPO_REF:?REPO_REF is required}"
: "${TRAIN_N:?TRAIN_N is required}"
: "${VAL_N:?VAL_N is required}"
: "${TRAIN_SEED:=1}"
: "${VAL_SEED:=99}"
: "${STYLE:=blend}"
: "${WIDTH:=3.0}"
: "${EPOCHS:=40}"
: "${VERSION:=v1}"
: "${WORKERS:=$(nproc)}"
# BATCH_SIZE / TRAIN_WORKERS / LR are intentionally NOT defaulted here — the
# training script's auto-tuner picks them from detected hardware. Set them
# in the env (e.g. via vast_run.py --batch-size / --train-workers / --lr)
# to override.
: "${WORKDIR:=/workspace}"

log "host info"
nvidia-smi || true
echo "CPU cores: $(nproc)   render workers: $WORKERS   train: auto-tuned"
echo "Free disk:"; df -h "$WORKDIR" || df -h /

log "install OS deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y >/dev/null
apt-get install -y --no-install-recommends git rsync ca-certificates openssh-client >/dev/null

# SSH agent forwarding from the orchestrator makes private GitHub clones work
# without ever shipping a key onto the box. github.com host key is pre-trusted.
mkdir -p ~/.ssh
ssh-keyscan -T 10 github.com >> ~/.ssh/known_hosts 2>/dev/null || true
chmod 700 ~/.ssh

log "clone $REPO_URL @ $REPO_REF"
cd "$WORKDIR"
rm -rf repo
git clone --filter=blob:none "$REPO_URL" repo
cd repo
git checkout "$REPO_REF"
git log -1 --oneline

log "install python deps (vision extras)"
cd model
# The PyTorch base image already ships torch+CUDA; reuse it. Install the rest
# of the project in editable mode so `python -m scripts.*` resolves.
python -m pip install --upgrade pip >/dev/null
# Install everything *except* torch (already present) to avoid clobbering the
# CUDA build with a CPU wheel.
python -m pip install \
  "numpy>=1.26" "pandas>=2.1" "scikit-learn>=1.4" "skl2onnx>=1.16" \
  "onnx>=1.15" "onnxruntime>=1.17" "joblib>=1.3" "tqdm>=4.66" \
  "pillow>=10.0" "matplotlib>=3.8" "onnxscript>=0.1" >/dev/null
python -m pip install --no-deps -e . >/dev/null
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

log "render train dataset (n=$TRAIN_N seed=$TRAIN_SEED style=$STYLE workers=$WORKERS)"
python -m scripts.build_vision_dataset \
  --out data --n "$TRAIN_N" --seed "$TRAIN_SEED" \
  --style "$STYLE" --workers "$WORKERS"

log "render val dataset (n=$VAL_N seed=$VAL_SEED)"
python -m scripts.build_vision_dataset \
  --out data --n "$VAL_N" --seed "$VAL_SEED" \
  --style "$STYLE" --workers "$WORKERS"

TRAIN_DIR="data/charts-${STYLE}-${TRAIN_N}-seed${TRAIN_SEED}"
VAL_DIR="data/charts-${STYLE}-${VAL_N}-seed${VAL_SEED}"

log "train + export (width=$WIDTH epochs=$EPOCHS, auto-tuning batch/workers/lr)"
# The training script auto-tunes batch_size, num_workers, and lr to the
# detected hardware and sets PYTORCH_CUDA_ALLOC_CONF / OMP_NUM_THREADS itself.
# We deliberately pass NEITHER --batch-size NOR --num-workers — auto picks
# right for this box. To override, set BATCH_SIZE / TRAIN_WORKERS in the env.
EXTRA_ARGS=()
if [ -n "${BATCH_SIZE:-}" ];     then EXTRA_ARGS+=(--batch-size   "$BATCH_SIZE"); fi
if [ -n "${TRAIN_WORKERS:-}" ];  then EXTRA_ARGS+=(--num-workers  "$TRAIN_WORKERS"); fi
if [ -n "${LR:-}" ];             then EXTRA_ARGS+=(--lr           "$LR"); fi

python -m scripts.train_and_export_vision \
  --train-npz "$TRAIN_DIR" \
  --val-npz   "$VAL_DIR" \
  --width "$WIDTH" --epochs "$EPOCHS" \
  --out artifacts --version "$VERSION" \
  "${EXTRA_ARGS[@]}"

log "artifacts produced"
ls -lh artifacts/
echo "REMOTE_DONE_OK"
