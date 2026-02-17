#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".jules" ]]; then
  echo ".jules policy check passed (directory not present)."
  exit 0
fi

violations=()
while IFS= read -r -d '' path; do
  rel="${path#./}"
  case "$rel" in
    .jules/bolt.md|.jules/examples_tutorials_tracking.md)
      ;;
    *)
      violations+=("$rel")
      ;;
  esac
done < <(find .jules -mindepth 1 -maxdepth 1 -type f -print0 | sort -z)

if (( ${#violations[@]} > 0 )); then
  echo "Found disallowed .jules artifacts:" >&2
  printf '  - %s\n' "${violations[@]}" >&2
  echo "Allowed: .jules/bolt.md, .jules/examples_tutorials_tracking.md" >&2
  exit 1
fi

echo ".jules policy check passed."
