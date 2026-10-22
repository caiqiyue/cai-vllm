#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== CAI-vLLM harness verification ==="

echo "[L0] Environment"
python --version
python - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("[FAIL] Python 3.10-3.12 is required")
print(f"[PASS] Python {sys.version_info.major}.{sys.version_info.minor}")
PY

echo "[L1] Harness state"
python - <<'PY'
import json
from pathlib import Path

root = json.loads(Path("feature_list.json").read_text(encoding="utf-8"))
mirror = json.loads(Path(".ai/state/feature-list.json").read_text(encoding="utf-8"))
if root != mirror:
    raise SystemExit("[FAIL] feature_list.json and .ai/state/feature-list.json differ")
active = [feature for feature in root["features"] if feature["status"] == "in_progress"]
if len(active) != 1:
    raise SystemExit(f"[FAIL] expected exactly one in_progress feature, found {len(active)}")
for feature in root["features"]:
    if feature["status"] == "passing" and not feature.get("evidence"):
        raise SystemExit(f"[FAIL] {feature['id']} is passing without evidence")
print(f"[PASS] one active feature: {active[0]['id']}")
PY

echo "[L2] Python syntax and package metadata"
python -m compileall -q nanovllm bench.py example.py
python - <<'PY'
import re
from pathlib import Path

metadata = Path("pyproject.toml").read_text(encoding="utf-8")
assert re.search(r'^name\s*=\s*"cai-vllm"\s*$', metadata, re.MULTILINE)
assert re.search(r'^include\s*=\s*\["nanovllm\*"\]\s*$', metadata, re.MULTILINE)
print("[PASS] syntax and package metadata")
PY

echo "[L3] Repository hygiene"
git -c core.whitespace=trailing-space,-blank-at-eof diff --check
git -c core.whitespace=trailing-space,-blank-at-eof diff --cached --check
echo "[PASS] no whitespace errors"

echo "[L4] Runtime dependency and import smoke"
set +e
runtime_output="$(python -c 'import torch; import nanovllm; print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")' 2>&1)"
runtime_status=$?
set -e
if [[ $runtime_status -eq 0 ]]; then
    echo "[PASS] $runtime_output"
elif [[ "${CAI_VLLM_REQUIRE_GPU:-0}" == "1" ]]; then
    echo "[FAIL] GPU/runtime smoke is required but unavailable"
    echo "$runtime_output"
    exit "$runtime_status"
else
    echo "[SKIP] GPU/runtime smoke unavailable in this environment"
    echo "$runtime_output"
fi

echo "=== Verification complete ==="
echo "Next: read feature_list.json, work on its single in_progress feature, and record evidence."

