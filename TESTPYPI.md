# Publishing to TestPyPI

This guide covers building and uploading `riskframe` to TestPyPI for a test
installation. TestPyPI is separate from PyPI and requires its own account and API
token.

## Before publishing

1. Register or sign in at [test.pypi.org](https://test.pypi.org/account/register/).
2. Create a TestPyPI API token. Never commit or share the token.
3. Check the `version` in `pyproject.toml`. An uploaded version cannot be uploaded
   again; bump it for each new release (for example, from `0.1.0.dev0` to
   `0.1.0.dev1`).

For local development, the token can be kept in the root `.env` file:

```text
TESTPYPI_TOKEN=pypi-your-token
```

Keep `.env` out of version control and out of the distribution. Do not paste the
token into a command that will be saved in shell history.

## Build and validate

Run these commands from the repository root using the Python environment intended
for publishing:

```powershell
python -m pip install --upgrade build twine
python -m build
```

Set `$release` below to the distribution name and version from `pyproject.toml`.
The matching artifacts are selected explicitly so an older build in `dist` is not
uploaded accidentally.

```powershell
$release = "riskframe-0.1.0.dev0"
$artifacts = @(Get-ChildItem .\dist -File | Where-Object { $_.Name.StartsWith($release) })
if ($artifacts.Count -eq 0) { throw "No distribution files found for $release" }
python -m twine check $artifacts.FullName
if ($LASTEXITCODE -ne 0) { throw "Distribution validation failed" }
```

The expected artifacts are a `.whl` wheel and a `.tar.gz` source distribution.

## Upload to TestPyPI

The following PowerShell commands read `TESTPYPI_TOKEN` from `.env` without
printing it, pass it to Twine only for this process, and remove the environment
variables afterward:

```powershell
$tokenLine = Get-Content .\.env |
    Where-Object { $_ -match '^\s*TESTPYPI_TOKEN\s*=' } |
    Select-Object -First 1
if (-not $tokenLine) { throw "TESTPYPI_TOKEN was not found in .env" }

$token = ($tokenLine -split '=', 2)[1].Trim()
if ($token.Length -ge 2 -and
    (($token[0] -eq '"' -and $token[$token.Length - 1] -eq '"') -or
     ($token[0] -eq "'" -and $token[$token.Length - 1] -eq "'"))) {
    $token = $token.Substring(1, $token.Length - 2)
}
if ([string]::IsNullOrWhiteSpace($token)) { throw "TESTPYPI_TOKEN is empty" }

$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = $token
try {
    python -m twine upload --non-interactive --disable-progress-bar `
        --repository-url https://test.pypi.org/legacy/ $artifacts.FullName
    if ($LASTEXITCODE -ne 0) { throw "TestPyPI upload failed" }
}
finally {
    Remove-Item Env:\TWINE_PASSWORD -ErrorAction SilentlyContinue
    Remove-Item Env:\TWINE_USERNAME -ErrorAction SilentlyContinue
    $token = $null
}
```

After upload, verify the release at
`https://test.pypi.org/project/riskframe/0.1.0.dev0/` (adjust the version as
needed).

## Install-test the release

Install runtime dependencies from the normal PyPI index, then install the package
from TestPyPI without asking TestPyPI to resolve dependencies:

```powershell
python -m pip install numpy pandas scipy
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps `
    riskframe==0.1.0.dev0
```

Use the version you published. Avoid combining TestPyPI with PyPI using
`--extra-index-url`; pip does not prioritize one index over another.

For more details, see the Python Packaging User Guide's
[TestPyPI guide](https://packaging.python.org/en/latest/guides/using-testpypi/) and
[packaging projects tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
