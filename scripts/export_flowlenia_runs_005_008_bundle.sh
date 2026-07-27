#!/bin/sh
# Export everything needed to replay Flow-Lenia runs 005-008 after retiring a remote machine.
# Usage:
#   sh scripts/export_flowlenia_runs_005_008_bundle.sh /absolute/path/to/repo

set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: sh $0 /absolute/path/to/repo" >&2
    exit 2
fi

REPO=$1
OPT_REL=experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt/optimization

if [ ! -d "$REPO" ]; then
    echo "Repository directory does not exist: $REPO" >&2
    exit 2
fi

cd "$REPO"

for RUN in 005 006 007 008; do
    if [ ! -d "$OPT_REL/run_$RUN" ]; then
        echo "Missing optimization checkpoint directory: $OPT_REL/run_$RUN" >&2
        exit 2
    fi
done

STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT=$REPO/flowlenia_runs_005_008_bundle_$STAMP.tar.gz
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/flowlenia-runs-005-008.XXXXXX")

cleanup() {
    rm -rf "$STAGE"
}
trap cleanup EXIT HUP INT TERM

mkdir -p "$STAGE/meta" "$STAGE/files"

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$STAGE/meta/git_head.txt"
    git status --short > "$STAGE/meta/git_status.txt"
    git diff --binary > "$STAGE/meta/git_diff.patch" || true
    git archive --format=tar --prefix=repo_head/ HEAD > "$STAGE/files/repo_head.tar"
else
    echo "git provenance unavailable" > "$STAGE/meta/git_head.txt"
fi

if command -v conda >/dev/null 2>&1; then
    conda env export -n torchjax --no-builds > "$STAGE/meta/conda_torchjax.yml" 2>&1 || true
    conda list -n torchjax --explicit > "$STAGE/meta/conda_torchjax_explicit.txt" 2>&1 || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$STAGE/meta/nvidia_smi.txt" 2>&1 || true
fi

if command -v conda >/dev/null 2>&1; then
    conda run --no-capture-output -n torchjax python - <<'PY' > "$STAGE/meta/python_jax.txt" 2>&1 || true
import platform
import jax
import jaxlib
import numpy

print("python:", platform.python_version())
print("jax:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("numpy:", numpy.__version__)
print("devices:", jax.devices())
PY
fi

: > "$STAGE/files/artifacts.list"
for RUN in 005 006 007 008; do
    find "$OPT_REL/run_$RUN" -type f -print >> "$STAGE/files/artifacts.list"
done

for FILE in \
    experiments/paper_check_flow_lenia/config_lockheed_1_openai_es_fixed_init_9opt.yaml \
    experiments/paper_check_flow_lenia/optimization/config_longrun_check_fix.yaml \
    experiments/paper_check_flow_lenia/frustration_simulation/config.yaml \
    experiments/paper_suite/flowlenia_arun_apf_300k_train50_grid128.yaml \
    scripts/main_opt_msc.py \
    scripts/util.py \
    scripts/run_paper_check_optimization.py \
    scripts/paper_suite_flowlenia_arun_apf.py \
    scripts/flowlenia_minibang_simulate.py
do
    if [ -f "$FILE" ]; then
        printf '%s\n' "$FILE" >> "$STAGE/files/artifacts.list"
    fi
done

LC_ALL=C sort -u "$STAGE/files/artifacts.list" > "$STAGE/files/artifacts_sorted.list"
tar -cf "$STAGE/files/runs_005_008_artifacts.tar" -T "$STAGE/files/artifacts_sorted.list"

(
    cd "$REPO"
    sha256sum $(cat "$STAGE/files/artifacts_sorted.list")
) > "$STAGE/meta/artifacts_sha256.txt"

cat > "$STAGE/README.txt" <<'EOF'
This bundle contains the remote source artifacts required to replay Flow-Lenia
optimization runs 005, 006, 007, and 008.

Included:
  - complete optimization checkpoint directories for runs 005-008;
  - resolved optimization configs and relevant simulation code;
  - a git HEAD source snapshot plus uncommitted diff;
  - conda/JAX/GPU provenance; and SHA-256 checksums.

Not included:
  - random parameter groups 005-008, which already exist on the local machine;
  - analysis/results;
  - APF/C1 simulation output;
  - videos; and unrelated checkpoints.
EOF

tar -czf "$OUT" -C "$STAGE" .
sha256sum "$OUT"
printf 'ARCHIVE=%s\n' "$OUT"
