"""
Pendulum Motion Analysis Web App
=================================
Flask-based web application for YOLOv8 pendulum video analysis.
Supports: simple pendulum, magnetic damping pendulum, torsional pendulum.
"""

# Fix OpenMP runtime conflict between PyTorch (ultralytics) and OpenCV on Windows.
# Must be set BEFORE any imports that load these libraries.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import os
import sys
import json
import uuid
import base64
import hashlib
import threading
import traceback
import time as _time
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processors.danbai_processor import process_danbai
from processors.cizuni_processor import process_cizuni
from processors.niubai_processor import process_niubai
from processors.ciliniudun_processor import process_ciliniudun
from processors.symbolic_regression import (
    run_sr_danbai, run_sr_ci_underdamped, run_sr_ci_critical,
    run_sr_ci_overdamped, run_sr_niubai
)
from processors.openmv_manager import openmv_manager, OPENMV_USB_VID, rgb_to_jpeg_bytes

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024 * 1024  # 1 GB

# ---------- Paths ----------
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# 上传文件扩展名白名单（同时防任意文件类型与路径穿越）
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ---------- Model paths ----------
# 模型查找优先级：
#   1. 环境变量 MODEL_W_BEST_{key}（可传多个路径，用分号/英文逗号分隔，取第一个存在的）
#   2. models/ 目录下的 {key}_best.pt
# 不再硬编码任何本机绝对路径（避免泄漏用户名/具体目录）。
_MODEL_ENV_NAMES = {
    "danbai": "MODEL_W_BEST_DANBAI",
    "cizuni": "MODEL_W_BEST_CIZUNI",
    "niubai": "MODEL_W_BEST_NIUBAI",
    "ciliniudun": "MODEL_W_BEST_CILINIUDUN",
}


def _split_model_paths(raw):
    for sep in (";", ","):
        if sep in raw:
            return [p.strip() for p in raw.split(sep) if p.strip()]
    return [raw.strip()] if raw.strip() else []


MODEL_PATHS = {}
for _k, _env in _MODEL_ENV_NAMES.items():
    _candidates = []
    _env_val = os.environ.get(_env, "")
    if _env_val:
        _candidates.extend(_split_model_paths(_env_val))
    _candidates.append(str(BASE_DIR / "models" / f"{_k}_best.pt"))
    _found = next((p for p in _candidates if os.path.exists(p)), None)
    if _found is not None:
        MODEL_PATHS[_k] = _found
    else:
        # 全都不存在时回退到第一个候选（便于启动时给出清晰告警）
        MODEL_PATHS[_k] = _candidates[0]

# ---------- Validate model files at startup ----------
_missing = [k for k, p in MODEL_PATHS.items() if not os.path.exists(p)]
if _missing:
    print(f"[WARNING] Model files not found: {_missing}", flush=True)
    print(f"  Checked paths:", flush=True)
    for k in _missing:
        print(f"    {k}: {MODEL_PATHS[k]}", flush=True)

# ---------- MIMO AI configuration ----------
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.5")

MIMO_SYSTEM_PROMPT = r"""你是一位物理实验教学助手，专注于阻尼振动实验教学。你的知识范围包括：
1. 阻尼振动的物理原理和数学推导（微分方程、解析解、参数物理意义）
2. 单摆、磁阻尼摆、扭摆、磁力牛顿摆等实验的操作指导和数据分析
3. 振动、阻尼、周期、频率、阻尼比、动量守恒与能量传递等相关物理概念
4. YOLOv8 视频分析在物理实验中的应用（含多目标跟踪、轨迹绑定、水平几何约束）

回答格式要求：
- 使用 Markdown 格式，支持标题(###)、加粗(**)、列表(-)、代码块(```)等
- 数学公式用 LaTeX 格式：行内公式用 $...$，独立公式用 $$...$$
- 例如：阻尼比 $\zeta = \frac{\beta}{\omega_0}$，微分方程 $$\ddot{\theta} + 2\beta\dot{\theta} + \omega_0^2\theta = 0$$
- 回答应严谨但友好，适合大学物理实验教学场景
- 适当使用表格对比不同阻尼类型的特征"""

# ---------- Teaching content ----------
TEACHING_DIR = BASE_DIR / "teaching"
TEACHING_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Pre-load YOLO models into memory ----------
print("Loading YOLO models...", flush=True)
LOADED_MODELS = {}
for _k, _path in MODEL_PATHS.items():
    if os.path.exists(_path):
        try:
            from ultralytics import YOLO
            LOADED_MODELS[_k] = YOLO(_path)
            print(f"  [OK] {_k}: {_path}", flush=True)
        except Exception as e:
            print(f"  [FAIL] {_k}: {e}", flush=True)
    else:
        print(f"  [SKIP] {_k}: file not found", flush=True)
print(f"Loaded {len(LOADED_MODELS)}/{len(MODEL_PATHS)} models.\n", flush=True)

# ---------- Mutex: only one YOLO process at a time ----------
_process_lock = threading.Lock()

# ---------- Calibration specifications ----------
CALIB_SPECS = {
    "danbai": {
        "name": "单摆 (Simple Pendulum)",
        "num_points": 3,
        "points": [
            {"id": 0, "label": "悬挂点 / 圆心 (Center/Pivot)", "color": "#00ffff"},
            {"id": 1, "label": "竖直方向起点 (Vertical Start)", "color": "#00ffff"},
            {"id": 2, "label": "竖直方向终点 (Vertical End, 方向 start→end)", "color": "#ff5050"},
        ],
        "keys": ["center", "vertical_start", "vertical_end"],
    },
    "cizuni": {
        "name": "磁阻尼摆 (Magnetic Damping Pendulum)",
        "num_points": 3,
        "points": [
            {"id": 0, "label": "竖直参考点1 (Vertical Ref Point 1)", "color": "#00ffff"},
            {"id": 1, "label": "竖直参考点2 (Vertical Ref Point 2)", "color": "#00ffff"},
            {"id": 2, "label": "转轴点 / 原点 (Pivot / Origin)", "color": "#ff5050"},
        ],
        "keys": ["vertical_ref_1", "vertical_ref_2", "pivot"],
    },
    "niubai": {
        "name": "扭摆 (Torsional Pendulum)",
        "num_points": 5,
        "points": [
            {"id": 0, "label": "原点 (Origin)", "color": "#00ffff"},
            {"id": 1, "label": "+X 轴端点 (+X Axis End)", "color": "#5050ff"},
            {"id": 2, "label": "+Y 轴端点 (+Y Axis End)", "color": "#32dc32"},
            {"id": 3, "label": "标尺参考点1 (Scale Ref Point 1)", "color": "#ffb41e"},
            {"id": 4, "label": "标尺参考点2 (Scale Ref Point 2)", "color": "#ffb41e"},
        ],
        "keys": ["origin", "x_axis_end", "y_axis_end", "scale_p1", "scale_p2"],
    },
    "ciliniudun": {
        "name": "磁力牛顿摆 (Magnetic Newton's Cradle)",
        # 摆球数量可变（min~max），分阶段标定（与底层 ManualMarker 一致）：
        # N 个摆球悬挂点 + 竖直方向 2 点 + 水平方向 2 点
        "num_pivots": 4,
        "num_points": 8,   # 默认摆球数 4 时的总点数（前端会按实际摆球数重算）
        "min_pivots": 2,
        "max_pivots": 5,
        "dynamic": True,
        "pivot_template": {"label": "摆球 {i} 悬挂点 (Pivot {i})", "color": "#00ffff"},
        "points": [
            {"id": 0, "kind": "pivot", "label": "摆球 1 悬挂点 (Pivot 1)", "color": "#00ffff"},
            {"id": 1, "kind": "pivot", "label": "摆球 2 悬挂点 (Pivot 2)", "color": "#00ffff"},
            {"id": 2, "kind": "pivot", "label": "摆球 3 悬挂点 (Pivot 3)", "color": "#00ffff"},
            {"id": 3, "kind": "pivot", "label": "摆球 4 悬挂点 (Pivot 4)", "color": "#00ffff"},
            {"id": 4, "kind": "vertical", "label": "竖直方向起点 (Vertical Start)", "color": "#00ffff"},
            {"id": 5, "kind": "vertical", "label": "竖直方向终点 (Vertical End)", "color": "#ff5050"},
            {"id": 6, "kind": "horizontal", "label": "水平方向起点 (Horizontal Start, 左)", "color": "#32dc32"},
            {"id": 7, "kind": "horizontal", "label": "水平方向终点 (Horizontal End, 右)", "color": "#32dc32"},
        ],
        "keys": ["pivot_1", "pivot_2", "pivot_3", "pivot_4",
                 "vertical_1", "vertical_2", "horizontal_1", "horizontal_2"],
    },
}

# ---------- Session store ----------
session_store = {}
SESSION_TTL = 1800  # 30 minutes


def cleanup_stale_sessions():
    now = _time.time()
    stale = [sid for sid, s in session_store.items() if now - s.get("created_at", now) > SESSION_TTL]
    for sid in stale:
        session = session_store.pop(sid)
        vpath = session.get("video_path", "")
        if vpath and os.path.exists(vpath):
            try:
                os.remove(vpath)
            except Exception:
                pass


# ---------- Background cleanup timer ----------
_cleanup_stop = threading.Event()


def _cleanup_loop():
    while not _cleanup_stop.wait(300):  # every 5 minutes
        try:
            cleanup_stale_sessions()
        except Exception:
            pass


_cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
_cleanup_thread.start()

# ---------- Helpers ----------


def get_first_frame_base64(video_path, max_width=1200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame.")

    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8"), frame.shape[1], frame.shape[0]


def file_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_cache_key(video_path, calibration, params):
    h = hashlib.md5()
    h.update(json.dumps(calibration, sort_keys=True).encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    try:
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except Exception:
        pass
    return h.hexdigest()


# ---------- Progress tracking ----------
_progress = {}  # session_id -> {"current": int, "total": int}


# ---------- Routes ----------

@app.route("/")
def index():
    resp = app.make_response(render_template("index.html", calib_specs=CALIB_SPECS))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/calibration_spec/<motion_type>")
def get_calibration_spec(motion_type):
    if motion_type not in CALIB_SPECS:
        return jsonify({"error": f"Unknown motion type: {motion_type}"}), 400
    return jsonify(CALIB_SPECS[motion_type])


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "文件过大，最大支持 1 GB。"}), 413


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    cleanup_stale_sessions()
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    motion_type = request.form.get("motion_type", "")
    if motion_type not in MODEL_PATHS:
        return jsonify({"error": f"Invalid motion type: {motion_type}"}), 400

    session_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    if ext not in ALLOWED_VIDEO_EXTS:
        return jsonify({"error": f"不支持的文件类型：{ext}。请上传常见视频格式（.mp4/.avi/.mov/.mkv/.webm/.m4v）。"}), 400
    video_path = str(UPLOAD_FOLDER / f"{session_id}{ext}")
    file.save(video_path)

    try:
        frame_b64, disp_w, disp_h = get_first_frame_base64(video_path)
    except Exception as e:
        os.remove(video_path)
        return jsonify({"error": f"Failed to read video: {str(e)}"}), 400

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    session_store[session_id] = {
        "video_path": video_path,
        "motion_type": motion_type,
        "original_filename": file.filename,
        "total_frames": total_frames,
        "created_at": _time.time(),
    }

    return jsonify({
        "session_id": session_id,
        "first_frame_base64": frame_b64,
        "display_width": disp_w,
        "display_height": disp_h,
        "motion_type": motion_type,
        "num_calib_points": CALIB_SPECS[motion_type]["num_points"],
        "total_frames": total_frames,
    })


def _run_processing(session_id, data, progress_callback=None):
    """Core processing logic shared by sync and SSE endpoints."""
    output_dir = data.get("output_dir", "")
    calibration_points = data.get("calibration_points", [])

    skip_start_sec = float(data.get("skip_start_sec", 0))
    skip_end_sec = float(data.get("skip_end_sec", 0))
    slow_motion_factor = float(data.get("slow_motion_factor", 1.0))
    flip_angle = bool(data.get("flip_angle", False))
    known_physical_dist_m = float(data.get("known_physical_dist_m", 0.064))
    target_cycles = int(data.get("target_cycles", 400))
    samples_per_cycle = int(data.get("samples_per_cycle", 25))
    num_pivots = int(data.get("num_pivots", 4))

    if session_id not in session_store:
        return None, {"error": "Invalid session. Please re-upload the video."}

    session = session_store[session_id]
    video_path = session["video_path"]
    motion_type = session["motion_type"]

    if not output_dir or not os.path.isdir(output_dir):
        return None, {"error": f"Output directory does not exist: {output_dir}"}
    if not os.access(output_dir, os.W_OK):
        return None, {"error": f"Output directory is not writable: {output_dir}"}

    spec = CALIB_SPECS[motion_type]
    if motion_type == "ciliniudun":
        # 摆球数量不预设：以实际标记的圆心数（总点数 - 4）为准
        min_p = spec.get("min_pivots", 2)
        max_p = spec.get("max_pivots", 5)
        num_pivots = len(calibration_points) - 4
        if not (min_p <= num_pivots <= max_p):
            return None, {"error": f"标定的摆球数量为 {num_pivots}，超出支持范围 ({min_p}~{max_p})，请重新标定。"}
    else:
        if len(calibration_points) != spec["num_points"]:
            return None, {"error": f"Expected {spec['num_points']} calibration points, got {len(calibration_points)}."}

    # Check cache
    cache_params = {
        "skip_start_sec": skip_start_sec, "skip_end_sec": skip_end_sec,
        "slow_motion_factor": slow_motion_factor, "target_cycles": target_cycles,
        "samples_per_cycle": samples_per_cycle, "known_physical_dist_m": known_physical_dist_m,
        "damping_subtype": data.get("damping_subtype", "欠阻尼"),
        "num_pivots": num_pivots,
        "gen_video": bool(data.get("gen_video", True)),
        "target_fps": float(data.get("target_fps", 30)),
    }
    cache_key = make_cache_key(video_path, calibration_points, cache_params)
    if session_id in _progress and "cache" in _progress[session_id]:
        cached = _progress[session_id]["cache"].get(cache_key)
        if cached:
            return cached, None

    # Convert display coords to original image coords
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    max_width = 1200
    scale = max_width / orig_w if orig_w > max_width else 1.0

    calibration = {}
    if motion_type == "ciliniudun":
        keys = [f"pivot_{i}" for i in range(1, num_pivots + 1)] + \
               ["vertical_1", "vertical_2", "horizontal_1", "horizontal_2"]
    else:
        keys = spec["keys"]
    for i, key in enumerate(keys):
        pt = calibration_points[i]
        calibration[key] = [pt["x"] / scale, pt["y"] / scale]

    # Progress callback for frame reporting
    def _on_frame(current, total):
        if progress_callback:
            progress_callback(current, total)

    model = LOADED_MODELS.get(motion_type)

    try:
        if motion_type == "danbai":
            result = process_danbai(
                video_path, output_dir, MODEL_PATHS[motion_type], calibration,
                skip_initial_seconds=skip_start_sec,
                slow_motion_factor=slow_motion_factor,
                target_cycles=target_cycles,
                model=model, progress_callback=_on_frame)
        elif motion_type == "cizuni":
            result = process_cizuni(
                video_path, output_dir, MODEL_PATHS[motion_type], calibration,
                skip_start_sec=skip_start_sec,
                skip_end_sec=skip_end_sec,
                slow_motion_factor=slow_motion_factor,
                flip_angle=flip_angle,
                model=model, progress_callback=_on_frame)
        elif motion_type == "niubai":
            result = process_niubai(
                video_path, output_dir, MODEL_PATHS[motion_type], calibration,
                skip_start_sec=skip_start_sec,
                skip_end_sec=skip_end_sec,
                slow_motion_factor=slow_motion_factor,
                known_physical_dist_m=known_physical_dist_m,
                num_cycles=target_cycles,
                samples_per_cycle=samples_per_cycle,
                model=model, progress_callback=_on_frame)
        elif motion_type == "ciliniudun":
            result = process_ciliniudun(
                video_path, output_dir, MODEL_PATHS[motion_type], calibration,
                num_pivots=num_pivots,
                skip_start_sec=skip_start_sec,
                skip_end_sec=skip_end_sec,
                target_fps=float(data.get("target_fps", 30)),
                conf=float(data.get("conf", 0.18)),
                iou=float(data.get("iou", 0.5)),
                imgsz=int(data.get("imgsz", 1536)),
                sg_window=int(data.get("sg_window", 11)),
                sg_polyorder=int(data.get("sg_polyorder", 3)),
                right_positive=not flip_angle,
                gen_video=bool(data.get("gen_video", True)),
                model=model, progress_callback=_on_frame)
        else:
            return None, {"error": f"Unknown motion type: {motion_type}"}

        # Plot preview as base64
        plot_path = result.get("plot_path", "")
        if plot_path and os.path.exists(plot_path):
            result["plot_preview"] = file_to_base64(plot_path)

        # Symbolic Regression
        damping_subtype = data.get("damping_subtype", "欠阻尼")
        csv_path = result.get("csv_path", "")
        try:
            if motion_type == "danbai":
                sr_result = run_sr_danbai(csv_path, output_dir)
            elif motion_type == "cizuni":
                if damping_subtype == "临界阻尼":
                    sr_result = run_sr_ci_critical(csv_path, output_dir)
                elif damping_subtype == "过阻尼":
                    sr_result = run_sr_ci_overdamped(csv_path, output_dir)
                else:
                    sr_result = run_sr_ci_underdamped(csv_path, output_dir)
            elif motion_type == "niubai":
                sr_result = run_sr_niubai(csv_path, output_dir)
            else:
                # 磁力牛顿摆：多摆碰撞/能量传递系统，不做阻尼振荡符号回归
                sr_result = None
        except Exception as e:
            print(f"Symbolic regression warning: {e}")
            sr_result = {"error": str(e)}

        if sr_result:
            result["sr_result"] = sr_result

        # Cache result
        if session_id in _progress:
            _progress[session_id].setdefault("cache", {})[cache_key] = result

        return result, None

    except Exception as e:
        traceback.print_exc()
        return None, {"error": f"Processing failed: {str(e)}"}


@app.route("/api/process", methods=["POST"])
def process_video():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided."}), 400

    session_id = data.get("session_id", "")
    _progress[session_id] = {"current": 0, "total": 0, "cache": _progress.get(session_id, {}).get("cache", {})}

    if not _process_lock.acquire(blocking=False):
        return jsonify({"error": "服务器正忙，请稍后再试。"}), 503

    try:
        result, error = _run_processing(session_id, data)
        if error:
            return jsonify(error), 400
        return jsonify({"success": True, "result": result, "output_dir": data.get("output_dir", "")})
    finally:
        _process_lock.release()
        _progress.pop(session_id, None)


@app.route("/api/process_stream", methods=["POST"])
def process_stream():
    """SSE endpoint: streams progress events + final result."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided."}), 400

    session_id = data.get("session_id", "")

    if not _process_lock.acquire(blocking=False):
        return jsonify({"error": "服务器正忙，请稍后再试。"}), 503

    def generate():
        try:
            _progress[session_id] = {"current": 0, "total": 0, "cache": _progress.get(session_id, {}).get("cache", {})}

            def on_progress(current, total):
                _progress[session_id]["current"] = current
                _progress[session_id]["total"] = total

            result, error = _run_processing(session_id, data, progress_callback=on_progress)

            if error:
                yield f"data: {json.dumps({'type': 'error', 'message': error['error']})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'result', 'result': result, 'output_dir': data.get('output_dir', '')})}\n\n"
        finally:
            _process_lock.release()
            _progress.pop(session_id, None)

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/progress/<session_id>")
def get_progress(session_id):
    p = _progress.get(session_id, {})
    current = p.get("current", 0)
    total = p.get("total", 1)
    pct = min(int(current / total * 100), 100) if total > 0 else 0
    return jsonify({"current": current, "total": total, "percent": pct})


@app.route("/api/cleanup", methods=["POST"])
def cleanup():
    data = request.get_json() or {}
    session_id = data.get("session_id", "")
    if session_id in session_store:
        session = session_store.pop(session_id)
        vpath = session.get("video_path", "")
        if os.path.exists(vpath):
            try:
                os.remove(vpath)
            except Exception:
                pass
    return jsonify({"success": True})


# ---------- STM32 serial control (electromagnet release) ----------
# 浏览器无法直接访问串口，因此由 Flask 代理转发（协议与 电脑端代码/controller.py 一致）：
#   GET  /api/serial/ports        列出可用串口
#   POST /api/serial/connect      打开串口（115200 8N1），连接后先发送 '0' 建立已知断电状态
#   POST /api/serial/command      发送 '0'(断电释放) / '1'(通电吸合)，返回提交时刻时间戳
#   POST /api/serial/disconnect   关闭串口（若通电则先断电保护）
# pyserial 为可选依赖：未安装时相关接口返回友好错误提示。
try:
    import serial as _pyserial
    from serial.tools import list_ports as _list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    _pyserial = None
    _list_ports = None
    SERIAL_AVAILABLE = False

SERIAL_BAUD = 115200
_SERIAL_LOCK = threading.Lock()
_serial_conn = None  # 全局唯一串口连接（serial.Serial 实例）


def _serial_unavailable():
    return jsonify({"error": "未安装 pyserial，请先执行: pip install pyserial"}), 503


def _is_bluetooth_port(p):
    """判断 pyserial 端口是否为蓝牙（蓝牙串口不提供，避免误选）。"""
    desc = (p.description or "")
    low = desc.lower()
    return "bluetooth" in low or "蓝牙" in desc


def _list_usable_ports(extra_label="手动指定"):
    """枚举可用串口：过滤蓝牙设备，并补齐 COM1~COM20 全部可选。

    枚举到的端口显示真实描述；未枚举到的常用 COM 号也一并列出
    （标为「手动指定」），保证任何设备号都能在网页下拉中选择。
    """
    ports = []
    found = set()
    for p in _list_ports.comports():
        if _is_bluetooth_port(p):
            continue
        ports.append({"device": p.device, "description": p.description or p.device})
        found.add(p.device.upper())
    for n in range(1, 21):
        dev = "COM%d" % n
        if dev not in found:
            ports.append({"device": dev, "description": extra_label})
    return ports


@app.route("/api/serial/ports")
def serial_list_ports():
    if not SERIAL_AVAILABLE:
        return _serial_unavailable()
    try:
        ports = _list_usable_ports()
    except Exception as e:
        return jsonify({"error": f"枚举串口失败: {e}"}), 400
    connected = _serial_conn is not None and _serial_conn.is_open
    return jsonify({
        "ports": ports,
        "connected": connected,
        "port": _serial_conn.port if connected else None,
    })


@app.route("/api/serial/connect", methods=["POST"])
def serial_connect():
    if not SERIAL_AVAILABLE:
        return _serial_unavailable()
    data = request.get_json() or {}
    port = str(data.get("port", "")).strip()
    if not port:
        return jsonify({"error": "缺少串口号 port。"}), 400
    global _serial_conn
    with _SERIAL_LOCK:
        if _serial_conn is not None and _serial_conn.is_open:
            return jsonify({"error": f"串口已连接: {_serial_conn.port}，请先断开。"}), 400
        try:
            conn = _pyserial.Serial(
                port=port, baudrate=SERIAL_BAUD,
                bytesize=_pyserial.EIGHTBITS, parity=_pyserial.PARITY_NONE,
                stopbits=_pyserial.STOPBITS_ONE, timeout=1.0, write_timeout=1.0,
            )
            # USB-TTL 适配器可能在打开串口时复位 STM32，等待其完成启动
            _time.sleep(1.5)
            conn.reset_input_buffer()
            # 协议无状态查询，先发送 '0' 建立已知安全状态（断电、无磁力）
            conn.write(b"0")
            conn.flush()
            _serial_conn = conn
            return jsonify({"success": True, "port": port, "state": 0})
        except Exception as e:
            return jsonify({"error": f"无法打开串口 {port}: {e}"}), 400


@app.route("/api/serial/command", methods=["POST"])
def serial_command():
    if not SERIAL_AVAILABLE:
        return _serial_unavailable()
    data = request.get_json() or {}
    cmd = str(data.get("command", "")).strip()
    if cmd not in ("0", "1"):
        return jsonify({"error": "command 仅支持 '0'(断电释放) 或 '1'(通电吸合)。"}), 400
    with _SERIAL_LOCK:
        if _serial_conn is None or not _serial_conn.is_open:
            return jsonify({"error": "串口未连接，请先连接。"}), 400
        try:
            _serial_conn.write(cmd.encode("ascii"))
            _serial_conn.flush()
            # 记录电脑提交指令的墙上时钟时间戳，供后续 OpenMV 帧时间对齐参考
            return jsonify({"success": True, "command": cmd,
                            "state": 1 if cmd == "1" else 0,
                            "timestamp_ns": _time.time_ns()})
        except Exception as e:
            return jsonify({"error": f"发送命令失败: {e}"}), 400


@app.route("/api/serial/disconnect", methods=["POST"])
def serial_disconnect():
    if not SERIAL_AVAILABLE:
        return _serial_unavailable()
    global _serial_conn
    with _SERIAL_LOCK:
        if _serial_conn is None or not _serial_conn.is_open:
            return jsonify({"success": True, "already": True})
        try:
            # 断电保护：关闭前确保电磁铁处于断电状态
            _serial_conn.write(b"0")
            _serial_conn.flush()
        except Exception:
            pass
        try:
            _serial_conn.close()
        except Exception:
            pass
        _serial_conn = None
    return jsonify({"success": True})


# ---------- Gripper gimbal (2D gimbal + gripper, STM32F103C8T6) ----------
# 协议与「二维云台＋夹爪」源码一致：USART1 115200 8N1，单字节 ASCII 命令，每命令执行一次动作：
#   'U' 上电机上转约 2° / 'D' 上电机下转约 2° / 'L' 下电机左转约 2° / 'R' 下电机右转约 2°
#   'O' 开夹 / 'C' 关夹（固件转发 C6 驱动器指令，重复的同向命令会被固件忽略）
# 夹爪只用于扭摆 (niubai) 与磁阻尼摆 (cizuni) 的释放；单 USB 串口（与电磁铁各自独立）。
_gripper_conn = None
_GRIPPER_LOCK = threading.Lock()
GRIPPER_BAUD = 115200
GRIPPER_COMMANDS = ("U", "D", "L", "R", "O", "C")


@app.route("/api/gripper/ports")
def gripper_list_ports():
    if not SERIAL_AVAILABLE:
        return _serial_unavailable()
    try:
        ports = _list_usable_ports()
    except Exception as e:
        return jsonify({"error": f"枚举串口失败: {e}"}), 400
    connected = _gripper_conn is not None and _gripper_conn.is_open
    return jsonify({
        "ports": ports,
        "connected": connected,
        "port": _gripper_conn.port if connected else None,
    })


@app.route("/api/gripper/connect", methods=["POST"])
def gripper_connect():
    if not SERIAL_AVAILABLE:
        return _serial_unavailable()
    data = request.get_json() or {}
    port = str(data.get("port", "")).strip()
    if not port:
        return jsonify({"error": "缺少串口号 port。"}), 400
    global _gripper_conn
    with _GRIPPER_LOCK:
        if _gripper_conn is not None and _gripper_conn.is_open:
            return jsonify({"error": f"串口已连接: {_gripper_conn.port}，请先断开。"}), 400
        try:
            conn = _pyserial.Serial(
                port=port, baudrate=GRIPPER_BAUD,
                bytesize=_pyserial.EIGHTBITS, parity=_pyserial.PARITY_NONE,
                stopbits=_pyserial.STOPBITS_ONE, timeout=1.0, write_timeout=1.0,
            )
            # USB-TTL 适配器可能在打开串口时复位 STM32，等待其完成启动
            _time.sleep(1.5)
            conn.reset_input_buffer()
            _gripper_conn = conn
            return jsonify({"success": True, "port": port})
        except Exception as e:
            return jsonify({"error": f"无法打开串口 {port}: {e}"}), 400


@app.route("/api/gripper/command", methods=["POST"])
def gripper_command():
    if not SERIAL_AVAILABLE:
        return _serial_unavailable()
    data = request.get_json() or {}
    cmd = str(data.get("command", "")).strip().upper()
    if cmd not in GRIPPER_COMMANDS:
        return jsonify({"error": "command 仅支持 U/D/L/R(方向) 或 O/C(开合)。"}), 400
    with _GRIPPER_LOCK:
        if _gripper_conn is None or not _gripper_conn.is_open:
            return jsonify({"error": "串口未连接，请先连接。"}), 400
        try:
            _gripper_conn.write(cmd.encode("ascii"))
            _gripper_conn.flush()
            return jsonify({"success": True, "command": cmd,
                            "timestamp_ns": _time.time_ns()})
        except Exception as e:
            return jsonify({"error": f"发送命令失败: {e}"}), 400


@app.route("/api/gripper/disconnect", methods=["POST"])
def gripper_disconnect():
    if not SERIAL_AVAILABLE:
        return _serial_unavailable()
    global _gripper_conn
    with _GRIPPER_LOCK:
        if _gripper_conn is None or not _gripper_conn.is_open:
            return jsonify({"success": True, "already": True})
        try:
            _gripper_conn.close()
        except Exception:
            pass
        _gripper_conn = None
    return jsonify({"success": True})


# ---------- OpenMV camera (real-time preview, recording, gimbal) ----------
# 双串口架构（与 source controller.py 一致）：
#   串口 1：OpenMV USB-C，USBDBG V1 @ 921600 → 帧采集
#   串口 2：USB-TTL 适配器，UART @ 115200     → 云台控制
# OpenMV 与 STM32 电磁铁使用各自独立的串口。

@app.route("/api/openmv/ports")
def openmv_list_ports():
    """List serial ports, highlighting OpenMV devices (bluetooth excluded)."""
    if not SERIAL_AVAILABLE:
        return _serial_unavailable()
    ports = []
    found = set()
    try:
        for p in _list_ports.comports():
            if _is_bluetooth_port(p):
                continue
            is_openmv = (getattr(p, "vid", None) == OPENMV_USB_VID
                         or "openmv" in (p.description or "").lower())
            ports.append({
                "device": p.device,
                "description": p.description or p.device,
                "is_openmv": is_openmv,
            })
            found.add(p.device.upper())
        # 补齐 COM1~COM20：未枚举到的也可手动选择（非 OpenMV）
        for n in range(1, 21):
            dev = "COM%d" % n
            if dev not in found:
                ports.append({"device": dev, "description": "手动指定", "is_openmv": False})
    except Exception as e:
        return jsonify({"error": f"枚举串口失败: {e}"}), 400
    return jsonify({
        "ports": ports,
        "camera_connected": openmv_manager.camera_connected,
        "control_connected": openmv_manager.control_connected,
    })


@app.route("/api/openmv/connect", methods=["POST"])
def openmv_connect():
    """Connect to OpenMV camera and optionally gimbal control port.

    Body: {"camera_port": "COM5", "control_port": "COM6"}
    """
    if not SERIAL_AVAILABLE:
        return _serial_unavailable()
    data = request.get_json() or {}
    camera_port = str(data.get("camera_port", "")).strip()
    control_port = str(data.get("control_port", "")).strip()

    if not camera_port:
        return jsonify({"error": "缺少摄像头串口号 camera_port。"}), 400

    result = openmv_manager.connect_camera(camera_port)
    if "error" in result:
        return jsonify(result), 400

    # Optionally connect control port
    if control_port:
        ctrl_result = openmv_manager.connect_control(control_port)
        if "error" in ctrl_result:
            result["control_warning"] = ctrl_result["error"]
        else:
            result["control_connected"] = True
            result["control_port"] = control_port

    return jsonify(result)


@app.route("/api/openmv/disconnect", methods=["POST"])
def openmv_disconnect():
    """Disconnect from OpenMV camera and gimbal."""
    result = openmv_manager.disconnect()
    return jsonify(result)


@app.route("/api/openmv/frame")
def openmv_frame():
    """Return a single JPEG frame for canvas-based preview."""
    if not openmv_manager.connected:
        return jsonify({"error": "OpenMV 未连接。"}), 400
    jpeg = openmv_manager.get_latest_jpeg(quality=70)
    if jpeg is None:
        return Response(b"", status=204)
    return Response(jpeg, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.route("/api/openmv/stream")
def openmv_stream():
    """MJPEG streaming endpoint for real-time preview.

    Returns a multipart/x-mixed-replace stream of JPEG frames.
    Recording is handled by the camera worker thread (decoupled from stream).
    """
    if not openmv_manager.connected:
        return jsonify({"error": "OpenMV 未连接。"}), 400

    def generate():
        # Send an initial empty boundary so the browser starts rendering immediately
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
        while openmv_manager.connected:
            frame = openmv_manager.get_latest_frame()
            if frame is None:
                _time.sleep(0.03)
                continue
            jpeg = rgb_to_jpeg_bytes(frame.rgb, quality=70)
            if not jpeg:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.route("/api/openmv/record/start", methods=["POST"])
def openmv_record_start():
    """Start recording frames from OpenMV."""
    result = openmv_manager.start_recording()
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/openmv/record/stop", methods=["POST"])
def openmv_record_stop():
    """Stop recording and save as MP4 video.

    Returns the video path and automatically creates a processing session
    (same as upload_video).
    """
    result = openmv_manager.stop_recording(str(UPLOAD_FOLDER))
    if "error" in result:
        return jsonify(result), 400

    video_path = result["video_path"]

    # Create a session for the recorded video (reuse upload flow)
    motion_type = request.get_json(silent=True)
    if motion_type is None:
        motion_type = {}
    motion_type = str(motion_type.get("motion_type", "danbai")).strip()
    if motion_type not in MODEL_PATHS:
        motion_type = "danbai"  # default fallback

    session_id = str(uuid.uuid4())
    try:
        frame_b64, disp_w, disp_h = get_first_frame_base64(video_path)
    except Exception as e:
        return jsonify({"error": f"录制成功但读取首帧失败: {e}"}), 500

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    session_store[session_id] = {
        "video_path": video_path,
        "motion_type": motion_type,
        "original_filename": result.get("video_name", "openmv_capture.mp4"),
        "total_frames": total_frames,
        "created_at": _time.time(),
    }

    return jsonify({
        **result,
        "session_id": session_id,
        "first_frame_base64": frame_b64,
        "display_width": disp_w,
        "display_height": disp_h,
        "motion_type": motion_type,
        "num_calib_points": CALIB_SPECS[motion_type]["num_points"],
        "total_frames": total_frames,
    })


@app.route("/api/openmv/gimbal", methods=["POST"])
def openmv_gimbal():
    """Send a gimbal command (U/D/L/R/C) to OpenMV via TX_INPUT."""
    data = request.get_json() or {}
    command = str(data.get("command", "")).strip().upper()
    result = openmv_manager.send_gimbal_command(command)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/openmv/status")
def openmv_status():
    """Query OpenMV connection and recording status."""
    return jsonify(openmv_manager.get_status())


# ---------- Teaching & AI Chat ----------

@app.route("/api/teaching_content/<topic>")
def get_teaching_content(topic):
    """Return teaching content JSON from the teaching/ directory."""
    allowed = {"theory", "guide_danbai", "guide_cizuni", "guide_niubai", "guide_ciliniudun"}
    if topic not in allowed:
        return jsonify({"error": f"Unknown topic: {topic}"}), 400
    json_path = TEACHING_DIR / f"{topic}.json"
    if not json_path.exists():
        return jsonify({"error": f"Content not found: {topic}"}), 404
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/ai_chat", methods=["POST"])
def ai_chat():
    """AI knowledge Q&A — SSE stream from mimo v2.5."""
    if not MIMO_API_KEY:
        return jsonify({"error": "AI 服务未配置。请设置环境变量 MIMO_API_KEY。"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided."}), 400

    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    # Inject system prompt
    full_messages = [{"role": "system", "content": MIMO_SYSTEM_PROMPT}] + messages

    try:
        from openai import OpenAI
        client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)

        def generate():
            try:
                stream = client.chat.completions.create(
                    model=MIMO_MODEL,
                    messages=full_messages,
                    stream=True,
                    max_tokens=2048,
                    temperature=0.7,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except ImportError:
        return jsonify({"error": "请安装 openai 包: pip install openai"}), 503


# ---------- Main ----------

def get_lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


if __name__ == "__main__":
    pid_file = BASE_DIR / "server.pid"
    pid_file.write_text(str(os.getpid()))

    HOST = "0.0.0.0"
    PORT = 5000
    lan_ip = get_lan_ip()

    print("=" * 60)
    print("  Pendulum Motion Analysis Web App")
    print("=" * 60)
    print(f"  Upload folder: {UPLOAD_FOLDER}")
    print(f"  Local:    http://127.0.0.1:{PORT}")
    if lan_ip:
        print(f"  Network:  http://{lan_ip}:{PORT}")
    print("=" * 60)

    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True)
    finally:
        _cleanup_stop.set()
        if pid_file.exists():
            pid_file.unlink()
