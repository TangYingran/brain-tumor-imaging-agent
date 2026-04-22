"""
Gradio 混合式界面（ChatGPT 风格 + 切片滑条）：
- 左侧：上传四模态图像（T1 / FLAIR / T1CE / T2），只需上传一次
- 右侧：ChatGPT 式多轮对话，由 Agent 控制所有分割和分析任务
- 下方：通过滑动条浏览处理后空间中的不同切片（FLAIR 背景 + 分割掩膜）
"""

import os
import re
import uuid
import numpy as np
import gradio as gr
from PIL import Image

from agent import create_agent
from config import GRADIO_CONFIG


# ============================================================
# patient_id 规则
# 格式：P + 3~6 位数字，例如 P001 / P1024 / P483920
# ============================================================
PATIENT_ID_PATTERN = re.compile(r"^P\d{3,6}$")


def generate_patient_id() -> str:
    """
    自动生成 patient_id，例如 P483920
    """
    return f"P{str(uuid.uuid4().int)[-6:]}"


def normalize_patient_id(patient_id: str) -> str:
    """
    规则：
    - 若留空：自动生成
    - 若手动输入：必须满足 P + 3~6 位数字
    """
    patient_id = (patient_id or "").strip().upper()

    if not patient_id:
        return generate_patient_id()

    if not PATIENT_ID_PATTERN.fullmatch(patient_id):
        raise ValueError("patient_id 格式错误，应为 P + 3~6位数字，例如 P001、P1024")

    return patient_id


# ============================================================
# 全局 image_store
# ============================================================
image_store = {
    "t1": None,
    "flair": None,
    "t1ce": None,
    "t2": None,

    "patient_id": None,
    "case_id": None,

    "preview_png": None,   # 当前展示用 PNG 或 numpy
    "seg_nifti": None,     # 分割结果 NIfTI 路径

    # 由 segment_nifti 在分割完成后写入（处理后空间）
    "seg_array": None,         # (D,H,W) 值 {0,1,2,4}
    "flair_array": None,       # (D,H,W)
    "num_slices": None,        # D
    "init_slice": None,        # 推荐默认切片
}

# 初始化 Agent（内部会使用 image_store）
agent = create_agent(image_store)


# ============================================================
# 工具函数
# ============================================================
def to_ui_image(img):
    """
    统一把 preview 转成 gr.Image 可显示的 numpy RGB 图
    """
    if img is None:
        return None

    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return np.stack([img, img, img], axis=-1)
        return img

    if isinstance(img, str) and os.path.exists(img):
        return np.array(Image.open(img).convert("RGB"))

    return None


def _normalize_slice_to_rgb(sl: np.ndarray):
    """
    单张灰度切片 -> RGB numpy
    """
    sl = sl.astype(np.float32)
    vmin, vmax = np.percentile(sl, 1), np.percentile(sl, 99)
    sl = np.clip(sl, vmin, vmax)
    sl = (sl - vmin) / (vmax - vmin + 1e-7)
    gray = (sl * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _render_flair_only(flair_np, z):
    """
    仅显示 FLAIR 灰度图（不落盘）
    """
    sl = flair_np[z]
    return _normalize_slice_to_rgb(sl)


def render_seg_overlay_numpy(seg, flair_np, z, alpha=0.4):
    """
    FLAIR 灰度 + 多类别分割叠加（不落盘）
    返回 RGB numpy array (H, W, 3)
    """
    sl = flair_np[z].astype(np.float32)
    seg_sl = seg[z]

    # ---------- FLAIR 灰度归一化 ----------
    vmin, vmax = np.percentile(sl, 1), np.percentile(sl, 99)
    sl = np.clip(sl, vmin, vmax)
    sl = (sl - vmin) / (vmax - vmin + 1e-7)

    rgb = np.stack([sl, sl, sl], axis=-1)

    def _hex_to_rgb01(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)

    # 你自定义的颜色
    WT_COLOR = _hex_to_rgb01("#01FF83")
    TC_COLOR = _hex_to_rgb01("#019FFF")
    ET_COLOR = _hex_to_rgb01("#FF00AA")

    # 注意：如果你的标签定义是 BraTS 原始值 {1,2,4}，这里保持一致
    color_map = {
        1: WT_COLOR,
        2: TC_COLOR,
        4: ET_COLOR,
    }

    for label, color in color_map.items():
        mask = seg_sl == label
        if not np.any(mask):
            continue
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color

    return (rgb * 255).astype(np.uint8)


# ============================================================
# 第一步：上传四模态 NIfTI
# ============================================================
def upload_modalities(patient_id, t1, flair, t1ce, t2):
    """
    保存 patient_id 和四个模态文件到 image_store
    - patient_id 留空则自动生成
    - 手动输入时必须满足固定格式：P + 3~6 位数字
    """
    if not (t1 and flair and t1ce and t2):
        return "❗ 请上传完整的 T1 / FLAIR / T1CE / T2 四个模态！", gr.update()

    try:
        patient_id = normalize_patient_id(patient_id)
    except ValueError as e:
        return f"❗ {str(e)}", gr.update()

    image_store["patient_id"] = patient_id
    image_store["t1"] = t1
    image_store["flair"] = flair
    image_store["t1ce"] = t1ce
    image_store["t2"] = t2

    return (
        f"✅ 患者 {patient_id} 的四个模态已成功上传！"
        "现在可以在右侧聊天框输入：例如 “请进行脑肿瘤三维分割”",
        patient_id,
    )


# ============================================================
# 切片滑条回调（仅保留 分割叠加 / 仅FLAIR）
# ============================================================
def on_slice_change(z_idx, view_mode):
    seg = image_store.get("seg_array")
    flair_np = image_store.get("flair_array")

    if flair_np is None:
        return to_ui_image(image_store.get("preview_png"))

    z = int(z_idx)

    # 仅 FLAIR
    if view_mode == "仅FLAIR":
        img_np = _render_flair_only(flair_np=flair_np, z=z)
        image_store["preview_png"] = img_np
        return img_np

    # 分割叠加
    if view_mode == "分割叠加":
        if seg is None:
            return to_ui_image(image_store.get("preview_png"))

        img_np = render_seg_overlay_numpy(
            seg=seg,
            flair_np=flair_np,
            z=z,
            alpha=0.4
        )
        image_store["preview_png"] = img_np
        return img_np

    return to_ui_image(image_store.get("preview_png"))


# ============================================================
# 第二步：聊天对话回调
# ============================================================
def chat_with_agent(user_msg, chat_history, view_mode):
    """
    和 Agent 对话：多轮 + 工具调用
    """
    chat_history = chat_history or []

    user_msg = (user_msg or "").strip()
    if not user_msg:
        bot_msg = "请先输入你的问题，例如：`请进行脑肿瘤三维分割`。"
        chat_history.append({"role": "assistant", "content": bot_msg})
        slider_update = gr.update()
        return chat_history, None, None, "", slider_update

    # 用户必须先上传模态
    if not image_store["t1"]:
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({
            "role": "assistant",
            "content": "你尚未上传四个模态的 NIfTI 文件，请先在左侧输入患者 ID 并上传 T1 / FLAIR / T1CE / T2。"
        })
        slider_update = gr.update()
        return chat_history, None, None, "", slider_update

    # 先记录用户消息
    chat_history.append({"role": "user", "content": user_msg})

    # 调用 Agent（内部带多轮记忆和工具调用）
    bot_msg = agent.process_request(user_msg)
    bot_msg = str(bot_msg)

    preview = to_ui_image(image_store.get("preview_png"))
    nifti = image_store.get("seg_nifti")

    # 更新滑条状态
    num_slices = image_store.get("num_slices")
    if num_slices is not None and num_slices > 0:
        init_slice = image_store.get("init_slice", num_slices // 2)
        slider_update = gr.update(
            minimum=0,
            maximum=num_slices - 1,
            value=int(init_slice),
            visible=True,
        )
    else:
        slider_update = gr.update(visible=False)

    # 如果当前模式是“仅FLAIR”或“分割叠加”，重新按当前模式渲染一次
    if num_slices is not None and num_slices > 0:
        try:
            preview = on_slice_change(
                z_idx=image_store.get("init_slice", num_slices // 2),
                view_mode=view_mode
            )
        except Exception:
            preview = to_ui_image(image_store.get("preview_png"))

    # 记录助手回复
    chat_history.append({"role": "assistant", "content": bot_msg})

    return chat_history, preview, nifti, "", slider_update


# ============================================================
# 清空对话 & 结果（保留上传模态 和 patient_id）
# ============================================================
def clear_conversation():
    """
    清空对话历史与可视化结果，但保留 patient_id 和四模态路径
    """
    agent.clear_history()

    image_store["case_id"] = None

    image_store["preview_png"] = None
    image_store["seg_nifti"] = None

    image_store["seg_array"] = None
    image_store["flair_array"] = None
    image_store["num_slices"] = None
    image_store["init_slice"] = None

    slider_update = gr.update(visible=False, value=0)
    return [], None, None, "", slider_update


# ============================================================
# Gradio UI（ChatGPT 风格 + 滑条，仅保留分割结果展示）
# ============================================================
def create_interface():
    with gr.Blocks(title="脑肿瘤 3D 分割 Agent") as demo:

        # ======================
        # CSS：压缩 gr.File 上传框高度
        # ======================
        gr.HTML("""
<style>
.compact-upload {
    min-height: 120px !important;
    max-height: 120px !important;
}
.compact-upload .file-upload {
    padding: 10px 10px !important;
}
.compact-upload svg {
    width: 22px !important;
    height: 22px !important;
}
.compact-upload span {
    font-size: 12px !important;
}
.compact-upload .wrap,
.compact-upload .container {
    min-height: 120px !important;
    max-height: 120px !important;
}
</style>
""")

        # ======================
        # 标题 + 说明
        # ======================
        gr.Markdown("# 🧠 脑肿瘤 3D 分割 Agent")

        with gr.Accordion("📌 使用说明（点击展开）", open=False):
            gr.Markdown(
                """
- 上方：一次性上传 **T1 / FLAIR / T1CE / T2** 四个模态
- 上传前可输入 **患者 ID**，格式建议为'P + 3~6 位数字'；若留空则系统自动生成
- 左侧：通过滑动条浏览不同切片，并支持显示模式切换（分割叠加 / 仅FLAIR）
- 右侧：像 ChatGPT 一样和 Agent 对话，完成：
  - 3D 脑肿瘤分割
  - 分割结果解释（ET / TC / WT）
  - 体积/占比定量分析
                """
            )

        # ======================
        # 顶部：患者ID + 四模态上传
        # ======================
        gr.Markdown("### 📤 上传四模态 NIfTI（一次上传即可）")

        patient_id_input = gr.Textbox(
            label="患者 ID（可选）",
            placeholder="格式：P + 3~6 位数字；若留空则自动生成",
        )

        with gr.Row(equal_height=True):
            t1_input = gr.File(label="T1", type="filepath", elem_classes="compact-upload")
            flair_input = gr.File(label="FLAIR", type="filepath", elem_classes="compact-upload")
            t1ce_input = gr.File(label="T1CE", type="filepath", elem_classes="compact-upload")
            t2_input = gr.File(label="T2", type="filepath", elem_classes="compact-upload")

        with gr.Row():
            upload_btn = gr.Button("上传模态", scale=1)
            upload_output = gr.Textbox(label="上传状态", interactive=False, scale=5)

        upload_btn.click(
            fn=upload_modalities,
            inputs=[patient_id_input, t1_input, flair_input, t1ce_input, t2_input],
            outputs=[upload_output, patient_id_input],
        )

        gr.Markdown("---")

        # ======================
        # 主体：左可视化，右对话
        # ======================
        with gr.Row(equal_height=True):

            # ---------- 左列：显示模式 + 可视化 + 分割结果 ----------
            with gr.Column(scale=2):

                gr.Markdown("### 🖼️ 分割结果查看")

                view_mode = gr.Radio(
                    choices=["分割叠加", "仅FLAIR"],
                    value="分割叠加",
                    label="显示模式",
                )

                preview_out = gr.Image(
                    label="当前切片可视化（处理后空间）",
                    height=320,
                    type="numpy",
                )

                nifti_out = gr.File(label="分割结果 NIfTI (.nii.gz)")

                slice_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=1,
                    value=0,
                    label="切片索引（处理后空间 D 方向）",
                    visible=False,
                )

                slice_slider.change(
                    fn=on_slice_change,
                    inputs=[slice_slider, view_mode],
                    outputs=preview_out,
                )

                view_mode.change(
                    fn=on_slice_change,
                    inputs=[slice_slider, view_mode],
                    outputs=preview_out,
                )

            # ---------- 右列：Agent 对话 ----------
            with gr.Column(scale=3):

                gr.Markdown("### 💬 与 Agent 对话")

                chatbot = gr.Chatbot(label="Agent 对话", height=560)

                with gr.Row():
                    user_input = gr.Textbox(
                        label="",
                        placeholder="例如：请进行脑肿瘤三维分割；或：分析肿瘤体积；或：解释分割结果",
                        lines=3,
                        scale=8,
                    )

                with gr.Row():
                    send_btn = gr.Button("发送", scale=1)
                    clear_btn = gr.Button("清空对话", scale=1)

                # 回车发送
                user_input.submit(
                    fn=chat_with_agent,
                    inputs=[user_input, chatbot, view_mode],
                    outputs=[chatbot, preview_out, nifti_out, user_input, slice_slider],
                )

                # 点击发送
                send_btn.click(
                    fn=chat_with_agent,
                    inputs=[user_input, chatbot, view_mode],
                    outputs=[chatbot, preview_out, nifti_out, user_input, slice_slider],
                )

                # 清空对话 + 可视化结果（保留 patient_id 和上传模态）
                clear_btn.click(
                    fn=clear_conversation,
                    inputs=None,
                    outputs=[chatbot, preview_out, nifti_out, user_input, slice_slider],
                )

    return demo


def main():
    print("启动脑肿瘤 3D 分割 Agent...")
    demo = create_interface()
    demo.launch(
        server_name=GRADIO_CONFIG["server_name"],
        server_port=GRADIO_CONFIG["server_port"],
        share=GRADIO_CONFIG.get("share", False),
    )


if __name__ == "__main__":
    main()