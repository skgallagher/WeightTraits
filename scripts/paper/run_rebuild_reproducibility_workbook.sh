#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${WEIGHTTRAITS_CODEX_RUNTIME:-${XDG_CACHE_HOME:-${HOME}/.cache}/codex-runtimes/codex-primary-runtime}"
runtime_manifest="${runtime_root}/runtime.json"
node_bin="${runtime_root}/dependencies/node/bin/node"
node_modules_dir="${runtime_root}/dependencies/node/node_modules"
artifact_manifest="${node_modules_dir}/@oai/artifact-tool/package.json"
module_link="${script_dir}/node_modules"

for required_path in \
  "${runtime_manifest}" \
  "${node_bin}" \
  "${node_modules_dir}" \
  "${artifact_manifest}"
do
  if [[ ! -e "${required_path}" ]]; then
    printf 'missing Codex workbook runtime path: %s\n' "${required_path}" >&2
    printf 'set WEIGHTTRAITS_CODEX_RUNTIME to the loader-provided runtime root\n' >&2
    exit 2
  fi
done

runtime_contract="$("${node_bin}" -e '
  const fs = require("node:fs");
  const runtime = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  const artifact = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
  process.stdout.write([
    runtime.bundleVersion,
    runtime.nodeVersion,
    runtime.artifactToolVersion,
    artifact.version,
    artifact.private,
  ].join("\t"));
' "${runtime_manifest}" "${artifact_manifest}")"

IFS=$'\t' read -r bundle_version node_version runtime_artifact_version package_artifact_version package_private \
  <<< "${runtime_contract}"

if [[ "${bundle_version}" != "26.805.11740" \
  || "${node_version}" != "v24.14.0" \
  || "${runtime_artifact_version}" != "2.8.39" \
  || "${package_artifact_version}" != "2.8.39" \
  || "${package_private}" != "true" ]]
then
  printf 'Codex workbook runtime contract mismatch: %s\n' "${runtime_contract}" >&2
  exit 2
fi

if [[ -e "${module_link}" || -L "${module_link}" ]]; then
  printf 'refusing to replace existing module path: %s\n' "${module_link}" >&2
  exit 2
fi

cleanup_module_link() {
  if [[ -L "${module_link}" ]]; then
    unlink "${module_link}"
  fi
}
trap cleanup_module_link EXIT

ln -s "${node_modules_dir}" "${module_link}"
"${node_bin}" "${script_dir}/rebuild_reproducibility_workbook.mjs" "$@"
