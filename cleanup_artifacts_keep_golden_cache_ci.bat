@echo off
setlocal
set "ROOT=%~dp0"
set "ARTIFACTS=%ROOT%artifacts"

if not exist "%ARTIFACTS%" (
  echo artifacts folder not found: %ARTIFACTS%
  exit /b 0
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root = $env:ARTIFACTS; " ^
  "if (-not (Test-Path -LiteralPath $root)) { Write-Host 'artifacts not found'; exit 0 }; " ^
  "Get-ChildItem -Force -LiteralPath $root | " ^
  "Where-Object { $_.Name -ne 'golden' -and $_.Name -ne 'llm_cache.json' -and $_.Name -notlike 'ci_bundle_*' } | " ^
  "Remove-Item -Recurse -Force"

echo Cleanup complete. Kept: golden, llm_cache.json, ci_bundle_*.zip
