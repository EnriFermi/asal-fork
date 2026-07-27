#!/usr/bin/env bash
set -eo pipefail

# Bootstrap JupyterLab + Jupyter AI + Codex + jai-acp-autoapprove
# into an EXISTING conda env.
#
# Usage:
#   bash bootstrap_jai_lab_existing_env_v2.sh ENV_NAME
#
# Example:
#   JAI_REPO_SPEC="git+https://github.com/<USER>/jai-acp-autoapprove.git" \
#   bash bootstrap_jai_lab_existing_env_v2.sh torchjax
#
# This script DOES NOT create a conda env. It requires an existing one.
#
# Important:
# - No `set -u`: conda compiler hooks may reference unset CONDA_BACKUP_* vars.
# - Conda package cache is forced to ~/.conda/pkgs to avoid /opt/conda permission errors.
# - If conda cannot install nodejs, Node/npm is installed locally into ~/.local/node-*.

ENV_NAME="${1:-}"
JAI_REPO_SPEC="${JAI_REPO_SPEC:-}"
CODEX_ACP_PACKAGE="${CODEX_ACP_PACKAGE:-@agentclientprotocol/codex-acp}"
NODE_VERSION="${NODE_VERSION:-22.11.0}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

install_local_node() {
  echo "==> Installing local Node.js/npm into ~/.local"

  local os arch node_arch tarball url prefix
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"

  case "$os:$arch" in
    linux:x86_64) node_arch="linux-x64" ;;
    linux:aarch64) node_arch="linux-arm64" ;;
    darwin:arm64) node_arch="darwin-arm64" ;;
    darwin:x86_64) node_arch="darwin-x64" ;;
    *) die "unsupported platform for local Node install: $os $arch" ;;
  esac

  tarball="node-v${NODE_VERSION}-${node_arch}.tar.xz"
  url="https://nodejs.org/dist/v${NODE_VERSION}/${tarball}"
  prefix="$HOME/.local/node-v${NODE_VERSION}-${node_arch}"

  mkdir -p "$HOME/.local"
  if [[ ! -x "$prefix/bin/node" ]]; then
    curl -fL "$url" -o "/tmp/${tarball}"
    rm -rf "$prefix"
    tar -xJf "/tmp/${tarball}" -C "$HOME/.local"
  fi

  export PATH="$prefix/bin:$PATH"

  if ! grep -q "$prefix/bin" "$HOME/.bashrc" 2>/dev/null; then
    echo "export PATH=\"$prefix/bin:\$PATH\"" >> "$HOME/.bashrc"
  fi
  if [[ "${SHELL:-}" == *zsh* ]] && ! grep -q "$prefix/bin" "$HOME/.zshrc" 2>/dev/null; then
    echo "export PATH=\"$prefix/bin:\$PATH\"" >> "$HOME/.zshrc"
  fi

  node -v
  npm -v
}

ensure_npm() {
  if need_cmd npm; then
    echo "==> npm already available"
    node -v
    npm -v
    return
  fi

  echo "==> npm not found; trying conda nodejs with user-writable package cache"
  mkdir -p "$HOME/.conda/pkgs"

  set +e
  CONDA_PKGS_DIRS="$HOME/.conda/pkgs" \
    conda install -y -n "$ENV_NAME" -c conda-forge nodejs
  local conda_status=$?
  set -e

  conda activate "$ENV_NAME"

  if need_cmd npm; then
    echo "==> npm installed via conda"
    node -v
    npm -v
    return
  fi

  echo "WARNING: conda nodejs install failed or npm still missing. Falling back to local Node.js."
  install_local_node

  need_cmd npm || die "npm is still missing after local Node install"
}

[[ -n "$ENV_NAME" ]] || die "Pass existing conda env name: bash $0 ENV_NAME"

echo "==> Checking conda"
need_cmd conda || die "conda not found."

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  die "conda env '$ENV_NAME' does not exist. Create it yourself first."
fi

echo "==> Activating env: $ENV_NAME"
conda activate "$ENV_NAME"

echo "==> Python in env"
which python
python --version

ensure_npm

echo "==> Updating pip tooling"
python -m pip install -U pip wheel setuptools

echo "==> Installing JupyterLab, Jupyter AI, ACP client, and useful Lab extensions"
python -m pip install -U \
  jupyterlab \
  jupyter-ai \
  jupyter-ai-acp-client \
  aiohttp \
  ipykernel \
  ipywidgets \
  jupyterlab-lsp \
  python-lsp-server \
  ruff-lsp \
  jupyterlab-git \
  jupyterlab_execute_time \
  jupytext \
  jupyterlab_code_formatter \
  black \
  isort

echo "==> Registering ipykernel"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo "==> Installing Codex CLI via npm"
npm install -g @openai/codex

need_cmd codex || die "codex not found after npm install. Check npm global bin path."
codex --version || true

echo "==> Installing Codex ACP adapter: $CODEX_ACP_PACKAGE"
npm install -g "$CODEX_ACP_PACKAGE"

if ! need_cmd codex-acp; then
  echo "WARNING: codex-acp not found after installing $CODEX_ACP_PACKAGE"
  echo "Trying legacy package @zed-industries/codex-acp"
  npm install -g @zed-industries/codex-acp
fi

need_cmd codex-acp || die "codex-acp not found. Check npm global bin path."
which codex
which codex-acp

echo "==> Writing Codex config"
mkdir -p "$HOME/.codex"
cat > "$HOME/.codex/config.toml" <<'TOML'
approval_policy = "never"
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = true
TOML

echo "==> Installing jai-acp-autoapprove package"
if [[ -z "$JAI_REPO_SPEC" ]]; then
  if [[ -f "pyproject.toml" ]] && grep -q 'jai-acp-autoapprove' pyproject.toml; then
    JAI_REPO_SPEC="-e ."
  else
    cat >&2 <<'MSG'
ERROR: JAI_REPO_SPEC is not set and current directory does not look like jai-acp-autoapprove repo.

Run one of:
  export JAI_REPO_SPEC="git+https://github.com/<USER>/jai-acp-autoapprove.git"
  bash bootstrap_jai_lab_existing_env_v2.sh ENV_NAME

or run this script from the jai-acp-autoapprove repo root.
MSG
    exit 1
  fi
fi

python -m pip install -U $JAI_REPO_SPEC

echo "==> Installing autoapprove .pth loader into this env"
jai-acp-autoapprove install

echo "==> Checking autoapprove patch"
JUPYTER_AI_AUTO_APPROVE_ACP=1 jai-acp-autoapprove check

echo "==> Optional: disable noisy realtime collaboration plugins if present"
jupyter labextension disable "@jupyter/collaboration-extension:rtcGlobalAwareness" >/dev/null 2>&1 || true
jupyter labextension disable "@jupyter/docprovider-extension" >/dev/null 2>&1 || true

cat <<EOF

DONE.

Use:
  conda activate $ENV_NAME
  export PATH="\$HOME/.local/bin:\$PATH"

Check:
  JUPYTER_AI_AUTO_APPROVE_ACP=1 jai-acp-autoapprove check
  which codex
  which codex-acp
  node -v
  npm -v

Local:
  JUPYTER_AI_AUTO_APPROVE_ACP=1 jupyter lab

Constructor/code-server remote:
  jai-remote-lab

If Codex is not logged in yet:
  codex
EOF
