param([string]$Python = 'python', [string]$DataZip = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$RunArguments = @('-u', 'run_cv.py', '--dataset', 'all', '--epochs', '150', '--lr', '0.001', '--patience', '25', '--min-epochs', '30')
if ($DataZip) { $RunArguments += @('--data-zip', $DataZip) }
# Use a fresh output directory to preserve the delivered results.
$RunArguments += @('--output-dir', ('results_rerun_' + (Get-Date -Format 'yyyyMMdd_HHmmss')))
& $Python @RunArguments
if ($LASTEXITCODE -ne 0) { throw 'Training failed; see output above.' }
