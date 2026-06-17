#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${VGGT_DTU_TMUX_SESSION:-vggt_dtu_benchmark}"
ENV_NAME="${VGGT_DTU_ENV:-lewm_rt}"
DATA_ROOT="${VGGT_DTU_DATA_ROOT:-$ROOT/dtu/DTU}"
OUTPUT_ROOT="${VGGT_DTU_OUTPUT_ROOT:-$ROOT/outputs/vggt_dtu_benchmark}"
CKPT="${VGGT_DTU_CKPT:-$ROOT/vggt/hf_cache/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt}"
POINTS_ROOT="${VGGT_DTU_POINTS_ROOT:-$ROOT/dtu/Points}"
CONFIDENCE_THRESHOLD="${VGGT_DTU_CONFIDENCE_THRESHOLD:-5.0}"

# ---- shared point-cloud evaluation parameters (keep aligned with SfM eval) ----
PC_NUM_SAMPLES="${VGGT_DTU_NUM_SAMPLES:-200000}"
PC_ALIGNMENT="${VGGT_DTU_ALIGNMENT:-icp}"
PC_ICP_SAMPLES="${VGGT_DTU_ICP_SAMPLES:-50000}"
PC_ICP_MAX_ITER="${VGGT_DTU_ICP_MAX_ITER:-100}"
PC_ICP_THRESHOLD_RATIO="${VGGT_DTU_ICP_THRESHOLD_RATIO:-0.05}"
PC_VOXEL_SIZE="${VGGT_DTU_VOXEL_SIZE:-0}"
PC_MAX_DISTANCE="${VGGT_DTU_MAX_DISTANCE:-0}"
PC_SEED="${VGGT_DTU_SEED:-42}"

run_scan() {
    local gpu="$1"
    local scan="$2"
    local scan_dir="$DATA_ROOT/scan$scan"
    local output_dir="$OUTPUT_ROOT/scan$scan"
    local gt_ply="$POINTS_ROOT/stl/stl$(printf '%03d' "$scan")_total.ply"
    mkdir -p "$output_dir"

    echo "[$(date '+%F %T')] gpu=$gpu scan=$scan inference"
    CUDA_VISIBLE_DEVICES="$gpu" conda run -n "$ENV_NAME" --no-capture-output \
        python -m src.tools.run_vggt_dtu_benchmark \
        --scan-dir "$scan_dir" \
        --output-dir "$output_dir" \
        --ckpt "$CKPT" \
        --confidence-threshold "$CONFIDENCE_THRESHOLD" \
        2>&1 | tee "$output_dir/inference.log"

    echo "[$(date '+%F %T')] gpu=$gpu scan=$scan evaluation"
    conda run -n "$ENV_NAME" --no-capture-output \
        python -m src.tools.evaluate_pointcloud \
        --pred "$output_dir/points_depth.ply" \
        --gt "$gt_ply" \
        --num-samples "$PC_NUM_SAMPLES" \
        --alignment "$PC_ALIGNMENT" \
        --icp-samples "$PC_ICP_SAMPLES" \
        --icp-max-iterations "$PC_ICP_MAX_ITER" \
        --icp-threshold-ratio "$PC_ICP_THRESHOLD_RATIO" \
        --voxel-size "$PC_VOXEL_SIZE" \
        --max-distance "$PC_MAX_DISTANCE" \
        --seed "$PC_SEED" \
        --output-json "$output_dir/pointcloud_eval.json" \
        2>&1 | tee "$output_dir/evaluation.log"
}

worker() {
    local gpu="$1"
    shift
    mkdir -p "$OUTPUT_ROOT/status"
    rm -f "$OUTPUT_ROOT/status/gpu${gpu}.done" "$OUTPUT_ROOT/status/gpu${gpu}.failed"
    mark_failed() {
        touch "$OUTPUT_ROOT/status/gpu${gpu}.failed"
    }
    trap mark_failed ERR
    for scan in "$@"; do
        run_scan "$gpu" "$scan"
    done
    touch "$OUTPUT_ROOT/status/gpu${gpu}.done"
}

aggregate_when_done() {
    mkdir -p "$OUTPUT_ROOT/status"
    while [[ "$(find "$OUTPUT_ROOT/status" -name 'gpu*.done' -type f | wc -l)" -lt 8 ]]; do
        if find "$OUTPUT_ROOT/status" -name 'gpu*.failed' -type f | grep -q .; then
            echo "At least one GPU worker failed; refusing to aggregate incomplete results." >&2
            exit 1
        fi
        sleep 30
    done

    echo "[$(date '+%F %T')] All GPU workers finished. Aggregating results..."
    conda run -n "$ENV_NAME" --no-capture-output python -c "
import json, csv
from pathlib import Path

output_root = Path('$OUTPUT_ROOT')
reports = []
for path in sorted(output_root.glob('scan*/pointcloud_eval.json'), key=lambda p: int(p.parent.name[4:])):
    report = json.loads(path.read_text(encoding='utf-8'))
    reports.append(report)

summary = {
    'num_scans': len(reports),
    'scan_ids': [r['scene'] for r in reports],
    'mean_accuracy_mm': sum(r['metrics']['accuracy']['mean'] for r in reports) / len(reports),
    'mean_completeness_mm': sum(r['metrics']['completeness']['mean'] for r in reports) / len(reports),
    'mean_overall_mm': sum(r['metrics']['overall'] for r in reports) / len(reports),
    'alignment': reports[0]['alignment']['method'] if reports else 'icp',
    'parameters': reports[0]['parameters'] if reports else {},
}
(output_root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

with (output_root / 'metrics.csv').open('w', newline='', encoding='utf-8') as fh:
    writer = csv.DictWriter(fh, fieldnames=['scan', 'accuracy_mm', 'completeness_mm', 'overall_mm',
                                             'accuracy_median', 'completeness_median', 'pred_points', 'gt_points'])
    writer.writeheader()
    for r in reports:
        writer.writerow({
            'scan': r['scene'],
            'accuracy_mm': r['metrics']['accuracy']['mean'],
            'completeness_mm': r['metrics']['completeness']['mean'],
            'overall_mm': r['metrics']['overall'],
            'accuracy_median': r['metrics']['accuracy']['median'],
            'completeness_median': r['metrics']['completeness']['median'],
            'pred_points': r['prediction']['evaluated_points'],
            'gt_points': r['gt']['evaluated_points'],
        })

print(json.dumps(summary, indent=2))
" 2>&1 | tee "$OUTPUT_ROOT/aggregate.log"

    echo "[$(date '+%F %T')] Aggregation complete."
}

launch() {
    command -v tmux >/dev/null
    [[ -f "$CKPT" ]] || { echo "Missing VGGT checkpoint: $CKPT" >&2; exit 1; }
    tmux has-session -t "$SESSION" 2>/dev/null && {
        echo "tmux session already exists: $SESSION" >&2
        exit 1
    }
    mkdir -p "$OUTPUT_ROOT/status"
    rm -f "$OUTPUT_ROOT/status"/gpu*.done "$OUTPUT_ROOT/status"/gpu*.failed
    local assignments=(
        "0 24 97"
        "1 37 105"
        "2 40 106"
        "3 55 110"
        "4 63 114"
        "5 65 118"
        "6 69 122"
        "7 83"
    )
    local first=1
    for assignment in "${assignments[@]}"; do
        read -r -a fields <<< "$assignment"
        local gpu="${fields[0]}"
        local command="cd '$ROOT' && bash '$0' worker ${fields[*]}"
        if [[ "$first" -eq 1 ]]; then
            tmux new-session -d -s "$SESSION" -n "gpu$gpu" "$command"
            first=0
        else
            tmux new-window -t "$SESSION" -n "gpu$gpu" "$command"
        fi
    done
    tmux new-window -t "$SESSION" -n aggregate "cd '$ROOT' && bash '$0' aggregate"
    echo "Started tmux session: $SESSION"
    echo "Attach: tmux attach -t $SESSION"
}

case "${1:-launch}" in
    launch) launch ;;
    worker) shift; worker "$@" ;;
    aggregate) aggregate_when_done ;;
    status)
        tmux list-windows -t "$SESSION"
        find "$OUTPUT_ROOT" -name pointcloud_eval.json -printf '%h\n' | sort -V
        ;;
    *) echo "Usage: $0 {launch|worker GPU SCAN...|aggregate|status}" >&2; exit 2 ;;
esac
