param(
  [string]$PythonLauncher = "py",
  [string]$PythonVersionArg = "-3.12",
  [string]$VenvPath = ".venv-audio-splitter",
  [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cpu"
)

$ErrorActionPreference = "Stop"

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
$modelDir = ".audio-separator-models"

if (!(Test-Path $venvPython)) {
  if ([string]::IsNullOrWhiteSpace($PythonVersionArg)) {
    & $PythonLauncher -m venv $VenvPath
  } else {
    & $PythonLauncher $PythonVersionArg -m venv $VenvPath
  }
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install --upgrade torch torchaudio --index-url $TorchIndexUrl
& $venvPython -m pip install --upgrade audio-separator
& $venvPython -m pip install --upgrade imageio-ffmpeg
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

& $venvPython -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda)"
& $venvPython -c "import audio_separator; print('audio_separator import ok')"
& $venvPython -c "import imageio_ffmpeg; print('ffmpeg', imageio_ffmpeg.get_ffmpeg_exe())"

Write-Host ""
Write-Host "Set these environment variables for the app:"
Write-Host "AUDIO_SPLITTER_ENGINE=audio-separator"
Write-Host "AUDIO_SPLITTER_AUDIO_SEPARATOR_COMMAND=$venvPython"
Write-Host "AUDIO_SPLITTER_DEVICE=cpu"
Write-Host "AUDIO_SPLITTER_OUTPUT_BIT_DEPTH=16"
Write-Host "AUDIO_SPLITTER_AUDIO_SEPARATOR_NORMALIZATION=0.98"
Write-Host "AUDIO_SPLITTER_AUDIO_SEPARATOR_USE_AUTOCAST=0"
