#!/usr/bin/env bash
# Determinism lint (spec/VDS-1.md sections 2 and 4).
#
# Fails if Rust code outside crates/vikshep-detmath calls a platform-dependent
# or rounding-changing float operation:
#   * transcendental and power functions of f32/f64 (exp, ln, log*, sin, cos,
#     tan, powf, powi, hypot, cbrt, ...): they lower to the platform libm and
#     differ between operating systems; use vikshep-detmath instead;
#   * mul_add: fused multiply-add rounds once instead of twice and is
#     forbidden in Tier-1 code (and needs a spec decision anywhere else);
#   * fast-math / algebraic float intrinsics, which permit reassociation;
#   * direct use of the libm crate, which must go through vikshep-detmath.
# sqrt is allowed (IEEE-754 requires it to be correctly rounded). Constants
# from std::f32::consts / std::f64::consts are allowed: they are exact
# binary literals, not computations.
#
# Exceptions live in scripts/determinism-lint.allow, one per line:
#   <path> :: <substring of the offending line> :: <justification>
# Every exception must carry a justification and must still match something;
# stale entries fail the lint.
#
# Usage: scripts/determinism-lint.sh   (from any directory inside the repo)

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
allow_file="scripts/determinism-lint.allow"

funcs='exp|exp2|exp_m1|ln|ln_1p|log|log2|log10|sin|cos|tan|sin_cos|asin|acos|atan|atan2|sinh|cosh|tanh|asinh|acosh|atanh|powf|powi|mul_add|hypot|cbrt|gamma|ln_gamma'
pattern="\\.(${funcs})[[:space:]]*\\(|\\bf(32|64)::(${funcs})\\b"
pattern+="|\\b(fadd|fsub|fmul|fdiv|frem)_(fast|algebraic)\\b|\\balgebraic_(add|sub|mul|div|rem)\\b"
pattern+="|\\b(core|std)::intrinsics\\b|\\blibm::"

mapfile -t files < <(find . -name '*.rs' \
    -not -path './target/*' \
    -not -path './crates/vikshep-detmath/*' \
    -not -path './.git/*' | sed 's|^\./||' | LC_ALL=C sort)

# Collect violations as "path:line:text", ignoring the part of each line
# after a `//` comment marker.
violations=()
for f in "${files[@]}"; do
    while IFS= read -r hit; do
        [[ -n "$hit" ]] && violations+=("$f:$hit")
    done < <(awk -v pat="$pattern" '{
        code = $0
        i = index(code, "//")
        if (i > 0) code = substr(code, 1, i - 1)
        if (code ~ pat) print FNR ":" $0
    }' "$f")
done

# Parse the allowlist.
allow_paths=()
allow_subs=()
allow_used=()
if [[ -f "$allow_file" ]]; then
    lineno=0
    while IFS= read -r entry || [[ -n "$entry" ]]; do
        lineno=$((lineno + 1))
        [[ -z "${entry// /}" || "$entry" =~ ^[[:space:]]*# ]] && continue
        path="${entry%% :: *}"
        rest="${entry#* :: }"
        sub="${rest%% :: *}"
        why="${rest#* :: }"
        if [[ "$path" == "$entry" || "$sub" == "$rest" || -z "${why// /}" ]]; then
            echo "determinism-lint: $allow_file:$lineno: malformed entry (need 'path :: substring :: justification')" >&2
            exit 2
        fi
        allow_paths+=("$path")
        allow_subs+=("$sub")
        allow_used+=(0)
    done < "$allow_file"
fi

status=0
for v in "${violations[@]}"; do
    path="${v%%:*}"
    text="${v#*:}"
    text="${text#*:}"
    allowed=0
    for i in "${!allow_paths[@]}"; do
        if [[ "$path" == "${allow_paths[$i]}" && "$text" == *"${allow_subs[$i]}"* ]]; then
            allowed=1
            allow_used[$i]=1
        fi
    done
    if [[ $allowed -eq 0 ]]; then
        echo "determinism-lint: forbidden float operation: $v" >&2
        status=1
    fi
done

for i in "${!allow_paths[@]}"; do
    if [[ "${allow_used[$i]}" -eq 0 ]]; then
        echo "determinism-lint: stale allowlist entry: ${allow_paths[$i]} :: ${allow_subs[$i]}" >&2
        status=1
    fi
done

if [[ $status -eq 0 ]]; then
    echo "determinism-lint: OK (${#files[@]} files checked, ${#allow_paths[@]} allowlisted exceptions)"
fi
exit $status
