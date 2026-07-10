#!/usr/bin/env bash
# Install KnowWhere Hermes plugin from this repo via symlink.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLUGIN_SRC="${REPO_ROOT}/hermes-plugin/knowwhere"
PLUGIN_DST="${HOME}/.hermes/plugins/knowwhere"
BACKUP_ROOT="${HOME}/.hermes/plugin-backups/knowwhere"
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

# Migrate stale backups left inside plugin discovery root (loader collision).
mkdir -p "${BACKUP_ROOT}"
shopt -s nullglob
for old in "${HOME}/.hermes/plugins/knowwhere.backup."*; do
  if [[ -d "${old}" ]]; then
    ts="$(basename "${old}" | sed 's/^knowwhere\.backup\.//')"
    dest="${BACKUP_ROOT}/${ts}"
    echo "==> Migrating stale backup ${old} -> ${dest}"
    mkdir -p "${dest}"
    mv "${old}"/* "${dest}/" 2>/dev/null || true
    rmdir "${old}" 2>/dev/null || mv "${old}" "${dest}/_dir"
  fi
done
shopt -u nullglob

mkdir -p "${HOME}/.hermes/plugins"
if [[ -e "${PLUGIN_DST}" && ! -L "${PLUGIN_DST}" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  backup_dir="${BACKUP_ROOT}/${ts}"
  echo "==> Backing up existing plugin dir to ${backup_dir}"
  mkdir -p "${backup_dir}"
  mv "${PLUGIN_DST}" "${backup_dir}/"
elif [[ -L "${PLUGIN_DST}" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  ref_file="${BACKUP_ROOT}/${ts}/symlink_target.txt"
  echo "==> Saving old symlink reference to ${ref_file}"
  mkdir -p "${BACKUP_ROOT}/${ts}"
  readlink "${PLUGIN_DST}" > "${ref_file}" || true
  rm "${PLUGIN_DST}"
fi

ln -sfn "${PLUGIN_SRC}" "${PLUGIN_DST}"
echo "==> Symlink: ${PLUGIN_DST} -> ${PLUGIN_SRC}"

if command -v hermes >/dev/null 2>&1; then
  if hermes plugins list 2>/dev/null | grep -Eiq "knowwhere.*(enabled|✓|active)"; then
    echo "==> Plugin already enabled (skipping enable prompt)"
  else
    echo "==> Plugin enable skipped — run: hermes plugins enable knowwhere"
  fi
else
  echo "WARN: hermes CLI not in PATH — enable manually: hermes plugins enable knowwhere"
fi

echo ""
echo "Done. Restart Hermes gateway / start a NEW chat session for hooks to load."
echo "memory.provider is NOT changed — Hindsight remains compatible."
