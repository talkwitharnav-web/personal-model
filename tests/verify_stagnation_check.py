"""Re-score the four completed smoke runs with the new stagnation check.

The check must FAIL stress_v1 (post-norm at 3x LR, which stopped learning without ever spiking)
and PASS the other three. If it can't separate those, it is worthless.
"""
import json
from pathlib import Path

from training.smoke_test import assess

EXPECTED = {
    "race_v1": "PASS",
    "race_v2": "PASS",
    "stress_v1": "FAIL",
    "stress_v2": "PASS",
}

failures = 0
for name, expected in EXPECTED.items():
    path = Path("tests/smoke_results") / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    result = assess(data["history"])
    ok = result["verdict"] == expected
    failures += 0 if ok else 1
    print(f"{'PASS' if ok else 'MISMATCH'}  {name:<12} verdict={result['verdict']:<4} "
          f"expected={expected:<4} second_half_gain={result['second_half_gain']:>7} "
          f"vs_unigram={result['vs_unigram']:>8}  "
          f"(spikes={data['spikes']}, crash-based checks say fine)")
    for reason in result["reasons"]:
        print(f"      ! {reason}")

print()
print(f"{len(EXPECTED) - failures}/{len(EXPECTED)} correct")
raise SystemExit(1 if failures else 0)
