#!/bin/sh
# Usage:
#   sh scripts/dump_remote_flowlenia_sigma_provenance.sh /absolute/path/to/repo
# Optional environment override:
#   OPT_REL=experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt/optimization

set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: sh $0 /absolute/path/to/repo" >&2
    exit 2
fi

REPO=$1
OPT_REL=${OPT_REL:-experiments/paper_check_flow_lenia/checkpoints_lockheed_1_openai_es_fixed_init_9opt/optimization}
OPT_ROOT=$REPO/$OPT_REL

if [ ! -d "$REPO" ]; then
    echo "Repository directory does not exist: $REPO" >&2
    exit 2
fi

if [ ! -d "$OPT_ROOT" ]; then
    echo "Optimization directory does not exist: $OPT_ROOT" >&2
    exit 2
fi

STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT=$REPO/flowlenia_remote_sigma_provenance_$STAMP.tar.gz
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/flowlenia-sigma-provenance.XXXXXX")

cleanup() {
    rm -rf "$STAGE"
}
trap cleanup EXIT HUP INT TERM

mkdir -p "$STAGE/meta" "$STAGE/files"
cd "$REPO"

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$STAGE/meta/git_head.txt"
    git status --short > "$STAGE/meta/git_status.txt"
    git diff -- scripts/main_opt_msc.py scripts/util.py \
        scripts/run_paper_check_optimization.py scripts/run_main_opt_from_yaml.py \
        > "$STAGE/meta/relevant_git_diff.patch" || true
else
    echo "git provenance unavailable" > "$STAGE/meta/git_head.txt"
fi

: > "$STAGE/meta/source_sha256.txt"
: > "$STAGE/files/source_files.list"
for FILE in \
    scripts/main_opt_msc.py \
    scripts/util.py \
    scripts/run_paper_check_optimization.py \
    scripts/run_main_opt_from_yaml.py \
    experiments/paper_check_flow_lenia/config_lockheed_1_openai_es_fixed_init_9opt.yaml
do
    if [ -f "$FILE" ]; then
        sha256sum "$FILE" >> "$STAGE/meta/source_sha256.txt"
        printf '%s\n' "$FILE" >> "$STAGE/files/source_files.list"
    fi
done

tar -cf "$STAGE/files/source_code.tar" -T "$STAGE/files/source_files.list"

: > "$STAGE/files/remote_files.list"
find "$OPT_REL" -type f \( \
    -name optimization_config.yaml -o \
    -iname '*.log' -o \
    -iname '*.out' -o \
    -iname '*.err' -o \
    -iname '*command*' \
\) -print >> "$STAGE/files/remote_files.list"

tar -cf "$STAGE/files/remote_run_configs_and_logs.tar" -T "$STAGE/files/remote_files.list"

tar -czf "$OUT" -C "$STAGE" .
sha256sum "$OUT"
printf 'ARCHIVE=%s\n' "$OUT"
