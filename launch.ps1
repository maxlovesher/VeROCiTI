<#
  VeROCiTI one-click launcher (run it through start.bat).

  Installs anything missing on first run, starts the Python backend
  (port 5000), the React dashboard (port 5173) and the local AI engine
  (port 8000, plate/vehicle detection on this computer's NVIDIA GPU, or its
  CPU and RAM when no usable GPU is found) in the background, waits until the
  backend and dashboard answer, then opens the dashboard in the browser.

  Everything runs inside a Windows job object owned by this window, so
  pressing Q or simply closing the window stops it all; nothing is left
  running in the background.

  Usage:
    start.bat                     launch everything
    start.bat -Cpu                run detection on the CPU even if a GPU is available
    start.bat -NoAI               skip the local AI engine (use a Colab GPU instead)
    start.bat -InstallGpu         install the NVIDIA (CUDA) build of PyTorch, then exit
    start.bat -CreateShortcut     put a "VeROCiTI" button on the desktop
#>
param([switch]$CreateShortcut, [switch]$NoAI, [switch]$Cpu, [switch]$InstallGpu)

$ErrorActionPreference = 'Stop'
$Root        = $PSScriptRoot
$BackendDir  = Join-Path $Root 'city flow model'
$FrontendDir = Join-Path $Root 'frontend'
$LogDir      = Join-Path $Root 'logs'
$IconPath    = Join-Path $FrontendDir 'public\verociti.ico'
$AiEngine    = Join-Path $Root 'colab_verociti_gpu.py'
$BackendPort  = 5000
$FrontendPort = 5173
$AiPort       = 8000
$CudaWheelIndex = 'https://download.pytorch.org/whl/cu128'
$DashboardUrl = "http://localhost:$FrontendPort/?portal=traffic"   # traffic dashboard, which opens on the physical board

function Say([string]$text, [string]$color = 'Gray') { Write-Host $text -ForegroundColor $color }
function Step([string]$text) { Say "  > $text" 'Cyan' }
function Fail([string]$text) { Say ""; Say "  X $text" 'Red'; Say ""; exit 1 }

if ($CreateShortcut) {
    $desktop = [Environment]::GetFolderPath('Desktop')
    $link = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop 'VeROCiTI.lnk'))
    $link.TargetPath = Join-Path $Root 'start.bat'
    $link.WorkingDirectory = $Root
    $link.Description = 'Start VeROCiTI (backend + dashboard)'
    if (Test-Path $IconPath) { $link.IconLocation = $IconPath }
    $link.Save()
    Say "Created the VeROCiTI button on your desktop: $desktop\VeROCiTI.lnk" 'Green'
    exit 0
}

# Every process started from here on joins this job; when this window goes
# away the job handle closes and Windows ends them all.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class VerocitiJob {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern IntPtr CreateJobObject(IntPtr attrs, string name);
    [DllImport("kernel32.dll")] static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint length);
    [DllImport("kernel32.dll")] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();
    [StructLayout(LayoutKind.Sequential)] struct Basic { public long UserTime, JobTime; public uint LimitFlags; public UIntPtr MinWs, MaxWs; public uint ActiveProcesses; public UIntPtr Affinity; public uint Priority, Scheduling; }
    [StructLayout(LayoutKind.Sequential)] struct Io { public ulong A, B, C, D, E, F; }
    [StructLayout(LayoutKind.Sequential)] struct Extended { public Basic BasicInfo; public Io IoInfo; public UIntPtr ProcMem, JobMem, PeakProcMem, PeakJobMem; }
    const uint KillOnJobClose = 0x2000;
    const int ExtendedLimitInformation = 9;
    public static bool AdoptCurrentProcess() {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) return false;
        var info = new Extended();
        info.BasicInfo.LimitFlags = KillOnJobClose;
        int size = Marshal.SizeOf(typeof(Extended));
        IntPtr ptr = Marshal.AllocHGlobal(size);
        try {
            Marshal.StructureToPtr(info, ptr, false);
            if (!SetInformationJobObject(job, ExtendedLimitInformation, ptr, (uint)size)) return false;
        } finally { Marshal.FreeHGlobal(ptr); }
        return AssignProcessToJobObject(job, GetCurrentProcess());
    }
}
'@

function Test-Port([int]$port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $wait = $client.BeginConnect('127.0.0.1', $port, $null, $null)
        return ($wait.AsyncWaitHandle.WaitOne(300) -and $client.Connected)
    } catch { return $false } finally { $client.Close() }
}

function Open-Dashboard {
    # Handing the URL to explorer.exe lets the already-running shell start the
    # browser, so the browser is never part of this window's job.
    Start-Process -FilePath 'explorer.exe' -ArgumentList $DashboardUrl
}

function Find-Python {
    $venv = Join-Path $Root 'venv\Scripts\python.exe'
    if (Test-Path $venv) { return $venv }
    foreach ($name in 'python', 'py') {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $exe = $cmd.Source
        # Skip the Microsoft Store "python" alias, which only opens the Store.
        & $exe -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { return $exe }
    }
    return $null
}

function Get-NvidiaGpu {
    # Includes cards Windows knows about but has powered off (e.g. laptop "Eco" GPU mode).
    try {
        return Get-PnpDevice -Class Display -ErrorAction Stop |
            Where-Object { $_.FriendlyName -match 'NVIDIA' } | Select-Object -First 1
    } catch { return $null }
}

function Get-AiHealth {
    try { return Invoke-RestMethod -Uri "http://127.0.0.1:$AiPort/health" -TimeoutSec 2 } catch { return $null }
}

function Show-GpuAdvice($gpu) {
    if (-not $gpu) { return }
    Say "  Your $($gpu.FriendlyName) is not being used. To detect on it:" 'Yellow'
    if ($gpu.Status -ne 'OK') {
        Say "    - Windows reports it as switched off. Turn the GPU back on (on ASUS laptops: Armoury Crate >" 'Yellow'
        Say "      GPU Mode > Standard or Ultimate, not Eco), then restart VeROCiTI." 'Yellow'
    }
    Say "    - Run '.\start.bat -InstallGpu' once to install the NVIDIA build of PyTorch (about 3 GB)." 'Yellow'
}

function Show-LogTail([string]$path) {
    if (Test-Path $path) {
        Say "  Last lines of $path :" 'Yellow'
        Get-Content $path -Tail 15 | ForEach-Object { Say "    $_" 'DarkGray' }
    }
}

function Start-Hidden([string]$workDir, [string]$command, [string]$logFile) {
    # cmd does the output redirection so the service needs no console window of its own.
    Start-Process -FilePath 'cmd.exe' -ArgumentList "/s /c `"$command > `"$logFile`" 2>&1`"" `
        -WorkingDirectory $workDir -WindowStyle Hidden -PassThru
}

function Wait-Ready([string]$name, [int]$port, $process, [string]$logFile, [int]$timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $port) { return }
        if ($process.HasExited) {
            Show-LogTail $logFile
            Fail "$name stopped while starting. Full log: $logFile"
        }
        Start-Sleep -Milliseconds 500
    }
    Show-LogTail $logFile
    Fail "$name did not come up on port $port within $timeoutSec s. Full log: $logFile"
}

$Host.UI.RawUI.WindowTitle = 'VeROCiTI'
Say ''
Say '  ============================================================' 'DarkCyan'
Say '     VeROCiTI  -  Vehicle Location and City Traffic Intelligence' 'White'
Say '  ============================================================' 'DarkCyan'
Say ''

if ($InstallGpu) {
    $python = Find-Python
    if (-not $python) { Fail 'Python 3.10+ was not found. Install it from https://www.python.org/downloads/ (tick "Add to PATH").' }
    Step 'Installing the NVIDIA (CUDA) build of PyTorch - about 3 GB, one time only'
    & $python -m pip install --upgrade torch torchvision --index-url $CudaWheelIndex
    if ($LASTEXITCODE -ne 0) { Fail 'The CUDA PyTorch install failed (see the messages above).' }
    $cuda = & $python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    if ($cuda) {
        Say "  Done. Detection will run on your $cuda." 'Green'
    } else {
        Say '  Installed, but PyTorch still cannot see an NVIDIA GPU right now.' 'Yellow'
        Show-GpuAdvice (Get-NvidiaGpu)
    }
    exit 0
}

# Already running (e.g. the button was clicked twice)? Just show it.
$backendUp  = Test-Port $BackendPort
$frontendUp = Test-Port $FrontendPort
if ($backendUp -and $frontendUp) {
    Say '  VeROCiTI is already running - opening the dashboard.' 'Green'
    Open-Dashboard
    Start-Sleep -Seconds 2
    exit 0
}
if ($backendUp)  { Fail "Port $BackendPort is already in use by another program. Close it and try again." }
if ($frontendUp) { Fail "Port $FrontendPort is already in use by another program. Close it and try again." }

# --- 1. Tools ---------------------------------------------------------------
Step 'Checking Python and Node.js'
$python = Find-Python
if (-not $python) { Fail 'Python 3.10+ was not found. Install it from https://www.python.org/downloads/ (tick "Add to PATH").' }
$node = (Get-Command node -ErrorAction SilentlyContinue).Source
$npm  = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
if (-not $node -or -not $npm) { Fail 'Node.js 18+ was not found. Install it from https://nodejs.org/.' }

# --- 2. First-run installs --------------------------------------------------
& $python -c "import flask" 2>$null
if ($LASTEXITCODE -ne 0) {
    Step 'Installing Python packages (first run only - this can take several minutes)'
    & $python -m pip install -r (Join-Path $BackendDir 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Fail 'Installing the Python packages failed (see the messages above).' }
}
if (-not $NoAI) {
    & $python -c "import fastapi, uvicorn, multipart, ultralytics, easyocr, torch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Step 'Installing the local AI engine (first run only - several minutes)'
        & $python -m pip install fastapi uvicorn python-multipart ultralytics easyocr
        if ($LASTEXITCODE -ne 0) { Fail 'Installing the AI engine packages failed. Run .\start.bat -NoAI to start without local detection.' }
    }
}
$viteEntry = Join-Path $FrontendDir 'node_modules\vite\bin\vite.js'
if (-not (Test-Path $viteEntry)) {
    Step 'Installing dashboard packages (first run only)'
    Push-Location $FrontendDir
    try { & $npm install --no-audit --no-fund } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $viteEntry)) { Fail 'npm install failed (see the messages above).' }
}

# --- 3. Start services ------------------------------------------------------
if (-not [VerocitiJob]::AdoptCurrentProcess()) {
    Say '  (Could not tie the services to this window; stop them from Task Manager if the window is closed.)' 'Yellow'
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$backendLog  = Join-Path $LogDir 'backend.log'
$frontendLog = Join-Path $LogDir 'frontend.log'
$aiLog       = Join-Path $LogDir 'ai-engine.log'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'   # the server prints emoji; the default Windows code page can't encode them

$ai = $null
if (-not $NoAI) {
    if (Test-Port $AiPort) {
        Say "  (Port $AiPort is already serving; using the AI engine that's already running there.)" 'DarkGray'
    } else {
        if ($Cpu) { $env:VEROCITI_AI_DEVICE = 'cpu' }
        $env:VEROCITI_AI_PORT = "$AiPort"
        Step "Starting the local AI engine (plate + vehicle detection) on port $AiPort"
        $ai = Start-Hidden $Root "`"$python`" `"$AiEngine`"" $aiLog
    }
    # The backend reads this at startup and sends every image, video and webcam frame here.
    $env:AI_BACKEND_URL = "http://127.0.0.1:$AiPort"
}

Step "Starting the backend (signal controller, tracking, analytics) on port $BackendPort"
$backend = Start-Hidden $BackendDir "`"$python`" server_standalone.py" $backendLog

Step "Starting the dashboard on port $FrontendPort"
$frontend = Start-Hidden $FrontendDir "`"$node`" `"$viteEntry`"" $frontendLog

Step 'Waiting for both to be ready'
Wait-Ready 'The backend' $BackendPort $backend $backendLog 120
Wait-Ready 'The dashboard' $FrontendPort $frontend $frontendLog 60

# --- 4. Running ---------------------------------------------------------------
Open-Dashboard
Say ''
Say '  VeROCiTI is running.' 'Green'
Say "    Dashboard : $DashboardUrl" 'White'
Say "    API       : http://localhost:$BackendPort/" 'White'
Say "    Logs      : $LogDir" 'DarkGray'
if ($NoAI) {
    Say '    Detection : local AI engine off (-NoAI); connect a Colab GPU for image detection' 'DarkGray'
} else {
    Say '    Detection : AI engine loading its models (the first run also downloads them)...' 'DarkGray'
}
Say ''
Say '  Press Q (or close this window) to stop everything.' 'Yellow'

$aiReported = $NoAI
while ($true) {
    if (-not $aiReported) {
        $health = Get-AiHealth
        if ($health) {
            $aiReported = $true
            $where = if ($health.device -eq 'cuda') { 'GPU' } else { 'CPU + RAM' }
            Say "  Detection ready on this computer's $where : $($health.device_name)" 'Green'
            if ($health.device -ne 'cuda' -and -not $Cpu) { Show-GpuAdvice (Get-NvidiaGpu) }
        } elseif ($ai -and $ai.HasExited) {
            $aiReported = $true
            Show-LogTail $aiLog
            Say '  The AI engine stopped, so image detection is unavailable. Everything else keeps running.' 'Yellow'
        }
    }
    if ($backend.HasExited) {
        Show-LogTail $backendLog
        Fail 'The backend stopped unexpectedly. Everything has been shut down.'
    }
    if ($frontend.HasExited) {
        Show-LogTail $frontendLog
        Fail 'The dashboard stopped unexpectedly. Everything has been shut down.'
    }
    try {
        if ([Console]::KeyAvailable -and [Console]::ReadKey($true).Key -eq 'Q') { break }
    } catch { }
    Start-Sleep -Milliseconds 400
}
Say '  Stopping VeROCiTI...' 'Yellow'
exit 0
