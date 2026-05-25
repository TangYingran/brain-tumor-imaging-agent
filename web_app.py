"""
Web Interface Backend - Flask API Server
Provides REST API endpoints for the modern web frontend
"""

import os
import re
import uuid
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='web_interface', static_url_path='')
CORS(app)

image_store = {
    "t1": None,
    "flair": None,
    "t1ce": None,
    "t2": None,
    "patient_id": None,
    "case_id": None,
    "preview_png": None,
    "seg_nifti": None,
    "seg_array": None,
    "flair_array": None,
    "num_slices": None,
    "init_slice": None,
}

PATIENT_ID_PATTERN = re.compile(r"^P\d{3,6}$")


def generate_patient_id():
    return f"P{str(uuid.uuid4().int)[-6:]}"


def normalize_patient_id(patient_id):
    patient_id = (patient_id or "").strip().upper()
    if not patient_id:
        return generate_patient_id()
    if not PATIENT_ID_PATTERN.fullmatch(patient_id):
        raise ValueError("patient_id 格式错误，应为 P + 3~6位数字，例如 P001、P1024")
    return patient_id


@app.route('/')
def index():
    return send_from_directory('web_interface', 'index.html')


@app.route('/api/status')
def status():
    return jsonify({
        "status": "online",
        "patient_id": image_store.get("patient_id"),
        "has_modalities": all([image_store["t1"], image_store["flair"],
                              image_store["t1ce"], image_store["t2"]]),
        "has_segmentation": image_store.get("seg_array") is not None,
        "num_slices": image_store.get("num_slices"),
    })


@app.route('/api/upload_modalities', methods=['POST'])
def upload_modalities():
    if 't1' not in request.files or 'flair' not in request.files:
        return jsonify({"success": False, "message": "请上传完整的 T1 / FLAIR / T1CE / T2 四个模态！"}), 400

    t1 = request.files['t1']
    flair = request.files['flair']
    t1ce = request.files['t1ce']
    t2 = request.files['t2']
    patient_id = request.form.get('patient_id', '')

    if not all([t1, flair, t1ce, t2]):
        return jsonify({"success": False, "message": "请上传完整的 T1 / FLAIR / T1CE / T2 四个模态！"}), 400

    try:
        patient_id = normalize_patient_id(patient_id)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)}), 400

    def load_nifti_to_memory(file):
        file_content = BytesIO(file.read())
        return file_content

    image_store["t1"] = load_nifti_to_memory(t1)
    image_store["flair"] = load_nifti_to_memory(flair)
    image_store["t1ce"] = load_nifti_to_memory(t1ce)
    image_store["t2"] = load_nifti_to_memory(t2)
    image_store["patient_id"] = patient_id

    return jsonify({
        "success": True,
        "message": f"✅ 患者 {patient_id} 的四个模态已成功上传！",
        "patient_id": patient_id
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '').strip()

    if not user_msg:
        return jsonify({"success": False, "message": "请输入你的问题"}), 400

    if not image_store["t1"]:
        return jsonify({
            "success": False,
            "message": "你尚未上传四个模态的 NIfTI 文件，请先上传 T1 / FLAIR / T1CE / T2。"
        }), 400

    from agent import create_agent
    agent = create_agent(image_store)
    bot_msg = agent.process_request(user_msg)

    response = {
        "success": True,
        "message": str(bot_msg),
        "has_preview": image_store.get("preview_png") is not None,
        "num_slices": image_store.get("num_slices"),
        "init_slice": image_store.get("init_slice"),
    }

    if image_store.get("preview_png") is not None:
        if isinstance(image_store["preview_png"], np.ndarray):
            response["preview_data"] = base64.b64encode(
                image_store["preview_png"].tobytes()
            ).decode('utf-8')
        elif isinstance(image_store["preview_png"], str):
            response["preview_path"] = image_store["preview_png"]

    return jsonify(response)


@app.route('/api/slice', methods=['GET'])
def get_slice():
    z_idx = int(request.args.get('z', 0))
    view_mode = request.args.get('mode', 'overlay')
    alpha = float(request.args.get('alpha', 0.4))

    seg = image_store.get("seg_array")
    flair_np = image_store.get("flair_array")

    if flair_np is None:
        return jsonify({"success": False, "message": "No data available"}), 400

    z = int(z_idx)

    if view_mode == "flair":
        sl = flair_np[z]
        sl = sl.astype(np.float32)
        vmin, vmax = np.percentile(sl, 1), np.percentile(sl, 99)
        sl = np.clip(sl, vmin, vmax)
        sl = (sl - vmin) / (vmax - vmin + 1e-7)
        img_data = (sl * 255).astype(np.uint8)
    else:
        if seg is None:
            return jsonify({"success": False, "message": "No segmentation available"}), 400

        sl = flair_np[z].astype(np.float32)
        seg_sl = seg[z]

        vmin, vmax = np.percentile(sl, 1), np.percentile(sl, 99)
        sl = np.clip(sl, vmin, vmax)
        sl = (sl - vmin) / (vmax - vmin + 1e-7)

        rgb = np.stack([sl, sl, sl], axis=-1)

        def _hex_to_rgb01(h):
            h = h.lstrip("#")
            return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)

        WT_COLOR = _hex_to_rgb01("#01FF83")
        TC_COLOR = _hex_to_rgb01("#019FFF")
        ET_COLOR = _hex_to_rgb01("#FF00AA")

        color_map = {1: WT_COLOR, 2: TC_COLOR, 4: ET_COLOR}

        for label, color in color_map.items():
            mask = seg_sl == label
            if not np.any(mask):
                continue
            rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color

        img_data = (rgb * 255).astype(np.uint8)

    img_pil = Image.fromarray(img_data)
    buffer = BytesIO()
    img_pil.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return jsonify({
        "success": True,
        "image": img_str,
        "z": z,
        "num_slices": image_store.get("num_slices")
    })


@app.route('/api/clear', methods=['POST'])
def clear_session():
    global image_store
    image_store = {
        "t1": None, "flair": None, "t1ce": None, "t2": None,
        "patient_id": None, "case_id": None, "preview_png": None,
        "seg_nifti": None, "seg_array": None, "flair_array": None,
        "num_slices": None, "init_slice": None,
    }
    return jsonify({"success": True, "message": "会话已清空"})


@app.route('/api/download/segmentation')
def download_segmentation():
    if image_store.get("seg_nifti") and os.path.exists(image_store["seg_nifti"]):
        return send_from_directory(
            os.path.dirname(image_store["seg_nifti"]),
            os.path.basename(image_store["seg_nifti"]),
            as_attachment=True,
            download_name=f"segmentation_{image_store.get('patient_id', 'unknown')}.nii.gz"
        )
    return jsonify({"success": False, "message": "No segmentation file available"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)