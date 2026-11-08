#!/usr/bin/env bash
# Publish the workspace crates to crates.io, one at a time, dependencies first.
#
# Used by the crates job of .github/workflows/release.yml instead of
# `cargo publish --workspace`, which stops at the first error, cannot be re-run
# once some crates are up (it refuses a version that already exists), and does
# not wait out the crates.io rate limit on new crates (a burst of about 5, then
# about one every 10 minutes; the first release creates 10). This script:
#   * orders the publishable crates by their normal, build and versioned dev
#     dependencies (crates.io requires all three to exist already);
#   * skips a crate whose version is already in the crates.io index, so a
#     failed job can simply be re-run;
#   * on HTTP 429 waits until the time crates.io gives (or
#     PUBLISH_RETRY_SECONDS, default 600) and tries the same crate again.
# Any other error stops the script.
#
# Usage: scripts/publish-crates.sh          (CARGO_REGISTRY_TOKEN set)
#        scripts/publish-crates.sh --plan   (print the order and what is
#                                            already published; publish nothing)

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

plan=0
[[ "${1:-}" == "--plan" ]] && plan=1
max_attempts=15

# "name version" per line, dependencies before dependents.
order="$(cargo metadata --no-deps --format-version 1 --locked | python3 -c '
import json, sys
pkgs = {p["name"]: p for p in json.load(sys.stdin)["packages"] if p["publish"] != []}
deps = {
    n: {d["name"] for d in p["dependencies"]
        if d["name"] in pkgs and (d["kind"] != "dev" or d["req"] != "*")}
    for n, p in pkgs.items()
}
done, seen = [], set()
def visit(n, path):
    if n in seen:
        return
    if n in path:
        sys.exit("dependency cycle: " + " -> ".join(path + [n]))
    for d in sorted(deps[n]):
        visit(d, path + [n])
    seen.add(n)
    done.append(n)
for n in sorted(pkgs):
    visit(n, [])
for n in done:
    print(n, pkgs[n]["version"])
')"

# True if NAME VERSION is in the crates.io sparse index.
published() {
  local name="$1" version="$2" path
  case "${#name}" in
    1) path="1/$name" ;;
    2) path="2/$name" ;;
    3) path="3/${name:0:1}/$name" ;;
    *) path="${name:0:2}/${name:2:2}/$name" ;;
  esac
  curl -sf "https://index.crates.io/$path" | grep -q "\"vers\":\"$version\""
}

while read -r name version; do
  if published "$name" "$version"; then
    echo "$name $version: already on crates.io, skipped"
    continue
  fi
  if (( plan )); then
    echo "$name $version: to publish"
    continue
  fi
  for (( attempt = 1; ; attempt++ )); do
    echo "$name $version: publishing (attempt $attempt)"
    if out="$(cargo publish -p "$name" --locked 2>&1 | tee /dev/stderr)"; then
      break
    fi
    if ! grep -qiE '429|too many' <<<"$out" || (( attempt == max_attempts )); then
      echo "$name $version: publish failed" >&2
      exit 1
    fi
    wait="${PUBLISH_RETRY_SECONDS:-600}"
    when="$(sed -n 's/.*try again after \([^.]*GMT\).*/\1/p' <<<"$out" | head -n 1)"
    if [[ -n "$when" ]] && until="$(date -d "$when" +%s 2>/dev/null)"; then
      wait=$(( until - $(date +%s) + 15 ))
      (( wait < 30 )) && wait=30
    fi
    echo "$name $version: crates.io rate limit, retrying in ${wait}s"
    sleep "$wait"
  done
done <<<"$order"
