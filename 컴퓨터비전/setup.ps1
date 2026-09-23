$ErrorActionPreference = 'Stop'
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $candidates = @(
            (Join-Path $env:USERPROFILE 'anaconda3\python.exe'),
            (Join-Path $env:USERPROFILE 'miniconda3\python.exe')
        )
        $basePython = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if ($basePython) {
            & $basePython -m venv (Join-Path $PSScriptRoot '.venv')
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3 -m venv (Join-Path $PSScriptRoot '.venv')
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            & python -m venv (Join-Path $PSScriptRoot '.venv')
        } else {
            throw 'Python 3.13 is required. Install Python and run setup.ps1 again.'
        }
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    }
    & $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Package verification failed.' }
    Write-Host 'Setup complete. Open a Python file in VS Code and press Ctrl+F5.'
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
}
