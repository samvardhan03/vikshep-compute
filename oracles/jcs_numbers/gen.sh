#!/usr/bin/env bash
# Regenerate crates/vikshep-numerics/tests/fixtures/ecmascript_numbers.txt:
# binary64 bit patterns and the string JavaScript's String(x) produces for
# them (ECMAScript Number::toString, which RFC 8785 adopts). Developer-run;
# CI only reads the committed fixture.
#
#   bash oracles/jcs_numbers/gen.sh        (needs python3 and node)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/../../crates/vikshep-numerics/tests/fixtures/ecmascript_numbers.txt"
mkdir -p "$(dirname "$out")"
python3 - <<'PY' > "$here/bits.tmp"
import random, struct
r = random.Random(8785)
def b(x): return struct.unpack('>Q', struct.pack('>d', x))[0]
out = []
while len(out) < 3000:
    v = r.getrandbits(64)
    if (v >> 52) & 0x7ff != 0x7ff:
        out.append(v)
for _ in range(1000):
    out.append(b(r.randint(1, 10 ** r.randint(1, 17)) * 10.0 ** r.randint(-30, 30)))
    out.append(b(float(f"{r.randint(1, 999999)}.{r.randint(0, 99)}5e{r.randint(-25, 25)}")))
for k in range(-1074, 1024, 2):
    out.append(b(2.0 ** k))
for e in range(-30, 30):
    out.append(b(10.0 ** e)); out.append(b(-10.0 ** e))
out.append(0x43143ff3c1cb0959)  # exact tie, resolved to even
print("\n".join(f"{v:016x}" for v in out))
PY
node -e '
const fs = require("fs");
const lines = fs.readFileSync(process.argv[1], "utf8").trim().split("\n");
const v = new DataView(new ArrayBuffer(8));
const out = ["# binary64 bits (hex) and ECMAScript String(x), node " + process.version];
for (const h of lines) { v.setBigUint64(0, BigInt("0x" + h)); out.push(h + " " + String(v.getFloat64(0))); }
fs.writeFileSync(process.argv[2], out.join("\n") + "\n");' "$here/bits.tmp" "$out"
rm "$here/bits.tmp"
echo "wrote $(($(wc -l < "$out") - 1)) values to $out"
