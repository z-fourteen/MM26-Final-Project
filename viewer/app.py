from __future__ import annotations

import argparse
import html
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from viewer.example_catalog import load_example_specs
from viewer.scene_registry import SceneRecord, SceneRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEWER_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = VIEWER_ROOT / "static"
EXAMPLES_ROOT = VIEWER_ROOT / "examples"
DEFAULT_PRESET_NAME = "VGGT + BA · Truck · 7K"

registry = SceneRegistry()
preset_records: dict[str, SceneRecord] = {}
for example in load_example_specs(EXAMPLES_ROOT, PROJECT_ROOT):
    camera_available = example.camera_path is None or example.camera_path.is_file()
    if example.ply_path.is_file() and camera_available:
        preset_records[example.name] = registry.register(
            label=example.name,
            method=example.method,
            ply_path=example.ply_path,
            camera_path=example.camera_path,
            source="preset",
            scene_id=f"example-{example.example_id}",
        )


def iframe_for(record: SceneRecord) -> str:
    scene_id = html.escape(record.scene_id, quote=True)
    return (
        '<iframe class="splat-viewer-frame" '
        f'src="/viewer-frame?scene={scene_id}" '
        'allow="fullscreen" loading="eager"></iframe>'
    )


def summary_for(record: SceneRecord) -> str:
    ply = record.ply_summary
    size_mb = ply["file_size"] / (1024 * 1024)
    kind = "3D Gaussian Splat" if ply["is_gaussian_splat"] else "普通 PLY 点云"
    oriented = sum(camera["rotation"] is not None for camera in record.cameras)
    return (
        f"**方法：** {record.method}  \n"
        f"**模型：** `{ply['file_name']}`，{size_mb:.1f} MB，{ply['vertex_count']:,} 个元素  \n"
        f"**类型：** {kind}（`{ply['format']}`）  \n"
        f"**相机：** {len(record.cameras)} 个，其中 {oriented} 个包含朝向"
    )


def load_preset(name: str) -> tuple[str, str]:
    record = preset_records.get(name)
    if record is None:
        raise gr.Error(f"预置场景不可用：{name}")
    return iframe_for(record), summary_for(record)


def load_upload(method: str, label: str, ply_file: str | None, camera_file: str | None) -> tuple[str, str]:
    if not ply_file:
        raise gr.Error("请先上传 3DGS PLY 文件")
    try:
        record = registry.register(
            label=label,
            method=method,
            ply_path=Path(ply_file),
            camera_path=Path(camera_file) if camera_file else None,
            source="upload",
        )
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    return iframe_for(record), summary_for(record)


initial_record = preset_records.get(DEFAULT_PRESET_NAME) or next(iter(preset_records.values()), None)
initial_html = iframe_for(initial_record) if initial_record else "<p>没有找到可用的预置场景。</p>"
initial_summary = summary_for(initial_record) if initial_record else "请上传一个 PLY 文件。"

CSS = """
.gradio-container { max-width: 1680px !important; }
.splat-viewer-frame { width: 100%; height: 760px; border: 1px solid #30363d; border-radius: 12px; background: #080b10; }
.viewer-note { color: #6b7280; font-size: 0.92rem; }
"""

with gr.Blocks(title="3DGS Camera Pose Viewer", fill_width=True) as demo:
    gr.Markdown("# 3DGS 相机位姿查看平台\n单次加载一个 SfM、VGGT 或自定义结果，不进行跨方法坐标对齐。")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 预置实验")
            preset = gr.Dropdown(
                choices=list(preset_records),
                value=DEFAULT_PRESET_NAME if DEFAULT_PRESET_NAME in preset_records else next(iter(preset_records), None),
                label="实验结果",
            )
            preset_button = gr.Button("加载预置结果", variant="primary")
            gr.Markdown("### 自定义上传")
            method = gr.Dropdown(choices=["SfM", "VGGT", "Custom"], value="Custom", label="方法标签")
            label = gr.Textbox(value="Uploaded scene", label="场景名称")
            ply_upload = gr.File(label="3DGS PLY", file_types=[".ply"], type="filepath")
            camera_upload = gr.File(
                label="相机参数（JSON / CSV / COLMAP images.txt，可选）",
                file_types=[".json", ".csv", ".txt"],
                type="filepath",
            )
            upload_button = gr.Button("加载上传结果")
            scene_summary = gr.Markdown(initial_summary)
            gr.Markdown(
                "支持 Graphdeco/VGGT `cameras.json`、相机中心 CSV 和 COLMAP `images.txt`。"
                "仅有相机中心时可显示轨迹，但不能显示真实朝向。",
                elem_classes="viewer-note",
            )
        with gr.Column(scale=4):
            viewer_html = gr.HTML(initial_html, padding=False)

    preset_button.click(load_preset, inputs=[preset], outputs=[viewer_html, scene_summary])
    upload_button.click(
        load_upload,
        inputs=[method, label, ply_upload, camera_upload],
        outputs=[viewer_html, scene_summary],
    )


app = FastAPI(title="3DGS Camera Pose Viewer")
app.mount("/viewer-assets", StaticFiles(directory=STATIC_ROOT), name="viewer-assets")


@app.get("/viewer-frame")
def viewer_frame() -> FileResponse:
    return FileResponse(STATIC_ROOT / "viewer.html", media_type="text/html")


@app.get("/api/scenes/{scene_id}")
def scene_metadata(scene_id: str) -> dict:
    try:
        return registry.get(scene_id).public_payload()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Scene not found") from exc


@app.get("/scene-files/{scene_id}/model.ply")
def scene_model(scene_id: str) -> FileResponse:
    try:
        record = registry.get(scene_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Scene not found") from exc
    return FileResponse(record.ply_path, media_type="application/octet-stream", filename=record.ply_path.name)


app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    css=CSS,
    max_file_size="2gb",
    show_error=True,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio 3DGS camera-pose viewer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
