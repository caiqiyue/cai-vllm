param(
    [switch]$RequireGpu
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

Write-Host '=== CAI-vLLM harness verification ==='
Write-Host '[L0] Environment'
python --version
if ($LASTEXITCODE -ne 0) { throw '[FAIL] Python is unavailable' }
python -c "import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), 'Python 3.10-3.12 is required'; print(f'[PASS] Python {sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) { throw '[FAIL] unsupported Python version' }

Write-Host '[L1] Harness state'
python -c "import json, pathlib; a=json.loads(pathlib.Path('feature_list.json').read_text(encoding='utf-8')); b=json.loads(pathlib.Path('.ai/state/feature-list.json').read_text(encoding='utf-8')); assert a == b, 'feature list copies differ'; active=[f for f in a['features'] if f['status']=='in_progress']; assert len(active)==1, f'expected one active feature, found {len(active)}'; assert all(f['status']!='passing' or f.get('evidence') for f in a['features']), 'passing feature lacks evidence'; print('[PASS] one active feature: ' + active[0]['id'])"
if ($LASTEXITCODE -ne 0) { throw '[FAIL] invalid harness state' }

Write-Host '[L2] Python syntax and package metadata'
python -m compileall -q nanovllm bench.py example.py
if ($LASTEXITCODE -ne 0) { throw '[FAIL] Python compilation failed' }
$metadataCheck = @'
from pathlib import Path

metadata = Path("pyproject.toml").read_text(encoding="utf-8")
assert 'name = "cai-vllm"' in metadata
assert 'include = ["nanovllm*"]' in metadata
print("[PASS] syntax and package metadata")
'@
$metadataCheck | python -
if ($LASTEXITCODE -ne 0) { throw '[FAIL] invalid package metadata' }

Write-Host '[L3] Repository hygiene'
git -c core.whitespace=trailing-space,-blank-at-eof diff --check
if ($LASTEXITCODE -ne 0) { throw '[FAIL] repository whitespace errors found' }
git -c core.whitespace=trailing-space,-blank-at-eof diff --cached --check
if ($LASTEXITCODE -ne 0) { throw '[FAIL] staged repository whitespace errors found' }
Write-Host '[PASS] no whitespace errors'

Write-Host '[L4] Runtime dependency and import smoke'
& python -c "import torch; import nanovllm; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"
if ($LASTEXITCODE -ne 0) {
    if ($RequireGpu) { throw '[FAIL] GPU/runtime smoke is required but unavailable' }
    Write-Host '[SKIP] GPU/runtime smoke unavailable in this environment'
}

Write-Host '=== Verification complete ==='
Write-Host 'Next: read feature_list.json, work on its single in_progress feature, and record evidence.'

