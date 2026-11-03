$ErrorActionPreference = 'Stop'
$hnetProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$hnetModelDir = Join-Path $hnetProjectRoot 'models/hnet_2stage_L'
New-Item -ItemType Directory -Path $hnetModelDir -Force | Out-Null
$hnetModelPath = Join-Path $hnetModelDir 'hnet_2stage_L.pt'
$hnetPartPath = $hnetModelPath + '.part'

if (-not (Test-Path -LiteralPath $hnetModelPath)) {
    curl.exe --fail --location --retry 3 --retry-delay 2 --continue-at - --output $hnetPartPath 'https://huggingface.co/cartesia-ai/hnet_2stage_L/resolve/eb0db7323b1131e6bdb8d1c7367d3f815d09426c/hnet_2stage_L.pt'
    if ($LASTEXITCODE -ne 0) { throw 'Download failed; partial file retained for resuming.' }
    $hnetCheckPath = $hnetPartPath
} else {
    $hnetCheckPath = $hnetModelPath
}
if ((Get-Item -LiteralPath $hnetCheckPath).Length -ne 3497374019) { throw 'Checkpoint size mismatch.' }
$hnetHash = (Get-FileHash -LiteralPath $hnetCheckPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hnetHash -ne '5711de80965d7ec8430a2db05a4859bc0700576e3e7841658fe1640d0c0e011c') { throw 'Checkpoint SHA-256 mismatch.' }
if ($hnetCheckPath -eq $hnetPartPath) { Move-Item -LiteralPath $hnetPartPath -Destination $hnetModelPath }

$hnetCodeBase = 'https://raw.githubusercontent.com/goombalab/hnet/3673fe1217ebeb0d1438c7c71d58d32bdd190ec2/'
Invoke-WebRequest -Uri ($hnetCodeBase + 'configs/hnet_2stage_L.json') -OutFile (Join-Path $hnetModelDir 'config.json') -TimeoutSec 30
$hnetSourceDir = Join-Path $hnetModelDir 'source'
New-Item -ItemType Directory -Path $hnetSourceDir -Force | Out-Null
$hnetSourceFiles = @('hnet/models/hnet.py', 'hnet/modules/block.py', 'hnet/modules/dc.py', 'hnet/modules/isotropic.py', 'hnet/modules/mha.py', 'hnet/modules/mlp.py', 'hnet/modules/rotary.py', 'hnet/utils/tokenizers.py', 'LICENSE')
foreach ($hnetSourceFile in $hnetSourceFiles) {
    Invoke-WebRequest -Uri ($hnetCodeBase + $hnetSourceFile) -OutFile (Join-Path $hnetSourceDir $hnetSourceFile.Replace('/', '__')) -TimeoutSec 30
}
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/state-spaces/mamba/a6a1dae6efbf804c9944a0c2282b437deb4886d8/mamba_ssm/modules/mamba2.py' -OutFile (Join-Path $hnetSourceDir 'mamba2.py') -TimeoutSec 30
Write-Output "Verified checkpoint and pinned source snapshots: $hnetModelDir"
