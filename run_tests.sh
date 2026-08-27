#!/usr/bin/env bash
set -e
for t in tests/test_synthetic.py tests/test_vectorised.py tests/test_sectors_regime.py tests/test_plan.py tests/test_invariants.py; do
  echo "=== $t"
  python3 "$t"
  echo
done
