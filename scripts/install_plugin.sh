#!/usr/bin/env bash
# Install KnowWhere Hermes plugin from this repo via symlink.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGIN_SRC="${REPO_ROOT}/hermes-plugin/knowwhere"
PLUGIN_DST="${HOME}/.hermes/plugins/knowwhere"
BRANCH_EXPECTED="feat/subconscious-outcome-loop"

echo "==> KnowWhere plugin install"
echo "    Repo: ${REPO_ROOT}"

current_branch="$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || true)"
if [[ "${current_branch}" != "${BRANCH_EXPECTED}" ]]; then
  echo "WARN: expected branch ${BRANCH_EXPECTED}, got '${current_branch}'"
fi

for f in plugin.yaml __init__.py; do
  if [[ ! -f "${PLUGIN_SRC}/${f}" ]]; then
    echo "ERROR: missing ${PLUGIN_SRC}/${f}"
    exit 1
  fi
done

kind="$(grep -E '^kind:' "${PLUGIN_SRC}/plugin.yaml" | awk '{print $2}')"
if [[ "${kind}" != "standalone" ]]; then
  echo "ERROR: plugin.yaml must set kind: standalone (got '${kind}')"
  exit 1
fi

mkdir -p "${HOME}/.hermes/plugins"
if [[ -e "${PLUGIN_DST}" && ! -L "${PLUGIN_DST}" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  backup="${HOME}/.hermes/plugins/knowwhere.backup.${ts}"
  echo "==> Backing up existing plugin dir to ${backup}"
  mv "${PLUGIN_DST}" "${backup}"
elif [[ -L "${PLUGIN_DST}" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  backup="${HOME}/.hermes/plugins/knowwhere.symlink.${ts}"
  echo "==> Saving old symlink reference to ${backup}"
  readlink "${PLUGIN_DST}" > "${backup}" || true
  rm "${PLUGIN_DST}"
fi

ln -sfn "${PLUGIN_SRC}" "${PLUGIN_DST}"
echo "==> Symlink: ${PLUGIN_DST} -> ${PLUGIN_SRC}"

if command -v hermes >/dev/null 2>&1; then
  if hermes plugins list 2>/dev/null | grep -q "knowwhere.*enabled"; then
    echo "==> Plugin already enabled"
  else
    yes "" 2>/dev/null | hermes plugins enable knowwhere 2>/dev/null \
      || hermes plugins enable knowwhere 2>/dev/null \
      || true
    echo "==> Plugin enable attempted"
  fi
else
  echo "WARN: hermes CLI not in PATH — enable manually: hermes plugins enable knowwhere"
fi

echo ""
echo "Done. Restart Hermes gateway / start a NEW chat session for hooks to load."
echo "memory.provider is NOT changed — Hindsight remains compatible."
