param(
    [Parameter(Mandatory = $true)]
    [string]$Scene,

    [int]$MaxRegister = 70,
    [double]$GlobalBaGrowthRatio = 1.3,
    [int]$GlobalBaMinInterval = 4,
    [int]$DegenerateMinObservations = 20,
    [switch]$RemoveDegenerateCameras,
    [string]$ReportName = ""
)

$ErrorActionPreference = "Stop"

$SceneConfig = "configs/scenes/$Scene.yaml"
$ImageDir = "data/scenes/$Scene/images"
$GsInputRoot = "data/3dgs_inputs"
$GsInputName = "${Scene}_sfm"
$GsInputPath = "$GsInputRoot/$GsInputName"
$CheckReport = "outputs/$Scene/reports/${Scene}_sfm_3dgs_scene_check.json"

if ([string]::IsNullOrWhiteSpace($ReportName)) {
    $ReportName = "${Scene}_paper_aligned_sfm_report.json"
}

if (-not (Test-Path $SceneConfig)) {
    throw "Scene config not found: $SceneConfig"
}

if (-not (Test-Path $ImageDir)) {
    throw "Image directory not found: $ImageDir"
}

function Invoke-PythonStep {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed with exit code ${LASTEXITCODE}: python $($Arguments -join ' ')"
    }
}

Write-Host "Running full SfM pipeline for scene: $Scene" -ForegroundColor Cyan
Write-Host "Scene config: $SceneConfig"

Invoke-PythonStep -m src.tools.extract_features --scene $SceneConfig

Invoke-PythonStep -m src.tools.match_features `
    --scene $SceneConfig `
    --strategy exhaustive `
    --force

Invoke-PythonStep -m src.tools.verify_matches --scene $SceneConfig --force

Invoke-PythonStep -m src.tools.initialize_reconstruction --scene $SceneConfig

Invoke-PythonStep -m src.tools.run_paper_aligned_sfm `
    --scene $SceneConfig `
    --max-register $MaxRegister `
    --strict-pnp-median-error 4.0 `
    --max-pnp-median-error 6.0 `
    --strict-pnp-mean-error 6.0 `
    --max-pnp-mean-error 8 `
    --local-ba-after-registration `
    --run-final-refinement `
    --global-ba-growth-ratio $GlobalBaGrowthRatio `
    --global-ba-min-interval $GlobalBaMinInterval `
    --global-ba-max-iterations 40 `
    --global-ba-max-points 0 `
    --global-ba-max-observations 0 `
    --diagnose-degenerate-cameras `
    --degenerate-min-observations $DegenerateMinObservations `
    --report-name $ReportName

if ($RemoveDegenerateCameras) {
    $LatestReportDir = Get-ChildItem -Path "outputs/$Scene/reports" -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $LatestReportDir) {
        throw "No controller report directory found under outputs/$Scene/reports"
    }
    $ResidualReport = Join-Path $LatestReportDir.FullName "final_global_ba2_registered_residual_after_filtering_report.json"
    if (-not (Test-Path $ResidualReport)) {
        $ResidualReport = Join-Path $LatestReportDir.FullName "paper_aligned_final_registered_residual_report.json"
    }
    if (-not (Test-Path $ResidualReport)) {
        throw "No final residual report found in $($LatestReportDir.FullName)"
    }
    Invoke-PythonStep -m src.tools.evaluate_degenerate_cameras `
        --scene $SceneConfig `
        --residual-report $ResidualReport `
        --min-observations $DegenerateMinObservations `
        --center-outlier-mad-scale 0 `
        --min-remaining-images 8 `
        --report-name "${Scene}_post_sfm_degenerate_camera_report.json" `
        --remove-candidates
}

Invoke-PythonStep -m src.tools.prepare_3dgs_from_sfm `
    --scene $SceneConfig `
    --output-root $GsInputRoot `
    --output-name $GsInputName

Invoke-PythonStep -m src.tools.check_3dgs_scene `
    --source-path $GsInputPath `
    --write-ply `
    --report-name $CheckReport

Write-Host "Done: $Scene" -ForegroundColor Green
Write-Host "SfM sparse output: data/scenes/$Scene/sparse/0"
Write-Host "3DGS input: $GsInputPath"
Write-Host "Check report: $CheckReport"
