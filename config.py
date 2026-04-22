"""
配置文件（适用于脑肿瘤 3D 分割）
"""

import os

# ==================== 3D 分割模型配置 ====================
MODEL_CONFIG = {

    # ⭐ 你自己的 SwinUNETR 训练权重路径（请换成你的）
    "checkpoint_path": "./model/swinunetr_0.8599.pt",
    # "checkpoint_path": "/root/logs/model/final_model_0.8581.pt",

    # ⭐ 4 个输入通道（Flair, T1, T1CE, T2）
    "in_channels": 4,

    # 输出 3 个类别通道 (TC, WT, ET)
    "out_channels": 3,

    # SwinUNETR 结构参数
    "feature_size": 48,
    "use_checkpoint": False,

    # Sliding Window 推理参数 (与训练一致)
    "roi_size": [96, 96, 96],     # patch size
    "sw_batch_size": 1,
    "overlap": 0.5,
}

# ==================== LLM配置 ====================
LLM_CONFIG = {
    # DeepSeek API配置
    'api_key': os.getenv('DEEPSEEK_API_KEY', 'your_api_key_here'),  # 需要设置环境变量或直接填写
    'base_url': 'https://api.deepseek.com',  # DeepSeek API地址
    'model_name': 'deepseek-chat',
    'temperature': 0.1,  # 低温度让模型更精准
    'max_tokens': 500,
}

# ==================== Gradio配置 ====================
GRADIO_CONFIG = {
    "server_name": "0.0.0.0",
    "server_port": 7860,
    "share": False,
}

# ==================== 可视化配置（ET/WT/TC 颜色） ====================
VIS_CONFIG = {
    "color_TC": [1, 159, 255],   # TC 蓝色系
    "color_WT": [1, 255, 131],   # WT 绿色系
    "color_ET": [255, 0, 170],   # ET 粉红色
    "alpha": 0.40,
}

# ==================== Agent配置 ====================
AGENT_CONFIG = {
    "max_iterations": 5,
    "verbose": True,
}

LABEL_COLOR_MAP = {
    "WT": {"label": 2, "color_name": "蓝色", "emoji": "🟦"},
    "ET": {"label": 4, "color_name": "绿色", "emoji": "🟩"},
    "TC": {"label": 1, "color_name": "红色", "emoji": "🟥"},
}