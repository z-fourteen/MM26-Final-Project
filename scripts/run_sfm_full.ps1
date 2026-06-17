param(
    [Parameter(Mandatory = $true)]
    [string]$Scene,

    [int]$MaxRegister = 70,
    [double]$GlobalBaGrowthRatio = 1.3,
    [int]$GlobalBaMinInterval = 4,
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

Write-Host "Running full SfM pipeline for scene: $Scene" -ForegroundColor Cyan
Write-Host "Scene config: $SceneConfig"

python -m src.tools.extract_features --scene $SceneConfig

python -m src.tools.match_features `
    --scene $SceneConfig `
    --strategy exhaustive

python -m src.tools.verify_matches --scene $SceneConfig

python -m src.tools.initialize_reconstruction --scene $SceneConfig

python -m src.tools.run_paper_aligned_sfm `
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
    --report-name $ReportName

python -m src.tools.prepare_3dgs_from_sfm `
    --scene $SceneConfig `
    --output-root $GsInputRoot `
    --output-name $GsInputName

python -m src.tools.check_3dgs_scene `
    --source-path $GsInputPath `
    --write-ply `
    --report-name $CheckReport

Write-Host "Done: $Scene" -ForegroundColor Green
Write-Host "SfM sparse output: data/scenes/$Scene/sparse/0"
Write-Host "3DGS input: $GsInputPath"
Write-Host "Check report: $CheckReport"
