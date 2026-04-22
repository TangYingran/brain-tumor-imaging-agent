"""
Brain Tumor 3D Segmentation Inference (SwinUNETR)
- 输入：4 个模态 NIfTI 路径 (T1, FLAIR, T1CE, T2)
- 输出：
    1) 分割 NIfTI (.nii.gz)
    2) 供滑条使用的分割 / FLAIR 缓存（已回填到原始 FLAIR 空间）
"""

import os
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt

from monai.networks.nets import SwinUNETR
from monai.inferers import SlidingWindowInferer

from config import MODEL_CONFIG


# ============================================================
#   一些基础工具函数
# ============================================================
def _extract_case_id(path: str) -> str:
    fname = os.path.basename(path)
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    return os.path.splitext(fname)[0]


def _compute_foreground_bbox(vols: np.ndarray):
    """
    vols: (C, D, H, W)
    返回 bbox = (z0, z1, y0, y1, x0, x1)，右边界开区间
    """
    mask = np.any(np.abs(vols) > 1e-6, axis=0)  # (D,H,W)

    if not np.any(mask):
        _, D, H, W = vols.shape
        return (0, D, 0, H, 0, W)

    zz, yy, xx = np.where(mask)
    z0, z1 = int(zz.min()), int(zz.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    return (z0, z1, y0, y1, x0, x1)


def _crop_with_bbox(vols: np.ndarray, bbox):
    z0, z1, y0, y1, x0, x1 = bbox
    return vols[:, z0:z1, y0:y1, x0:x1]


def _restore_seg_to_full(seg_crop: np.ndarray, full_shape, bbox):
    """
    seg_crop: (D',H',W')
    full_shape: (D,H,W)
    """
    seg_full = np.zeros(full_shape, dtype=np.uint8)
    z0, z1, y0, y1, x0, x1 = bbox
    seg_full[z0:z1, y0:y1, x0:x1] = seg_crop
    return seg_full


def _normalize_intensity_nonzero_channelwise(vols: np.ndarray):
    """
    对每个通道在非零区域做 z-score 归一化
    vols: (C,D,H,W)
    """
    out = vols.astype(np.float32).copy()

    for c in range(out.shape[0]):
        x = out[c]
        mask = np.abs(x) > 1e-6
        if np.any(mask):
            mean = x[mask].mean()
            std = x[mask].std()
            if std < 1e-8:
                std = 1.0
            x[mask] = (x[mask] - mean) / std
            x[~mask] = 0.0
        else:
            x[:] = 0.0
        out[c] = x

    return out


# ============================================================
#   3D 分割模型
# ============================================================
class BrainSegmentationModel3D:
    """3D Brain Tumor segmentation inference using SwinUNETR"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = MODEL_CONFIG
        self.model = None

        self.inferer = SlidingWindowInferer(
            roi_size=tuple(self.config.get("roi_size", [96, 96, 96])),
            sw_batch_size=1,
            overlap=0.5,
        )

    # --------------------------------------------------------
    #   加载模型
    # --------------------------------------------------------
    def load_model(self):
        if self.model is not None:
            return

        ckpt_path = self.config["checkpoint_path"]
        print(f"[BrainSeg] Loading model from: {ckpt_path}")

        self.model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=self.config.get("feature_size", 48),
            use_checkpoint=False,
        ).to(self.device)

        sd = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

        print("[BrainSeg] Model loaded correctly!")

    # --------------------------------------------------------
    #   数据加载
    # --------------------------------------------------------
    def load_4_modalities(self, t1, flair, t1ce, t2):
        img_t1 = sitk.ReadImage(t1)
        img_flair = sitk.ReadImage(flair)
        img_t1ce = sitk.ReadImage(t1ce)
        img_t2 = sitk.ReadImage(t2)

        vol_t1 = sitk.GetArrayFromImage(img_t1).astype(np.float32)
        vol_flair = sitk.GetArrayFromImage(img_flair).astype(np.float32)
        vol_t1ce = sitk.GetArrayFromImage(img_t1ce).astype(np.float32)
        vol_t2 = sitk.GetArrayFromImage(img_t2).astype(np.float32)

        # ⚠ 训练顺序：T1, FLAIR, T2, T1CE
        vols = np.stack([vol_t1, vol_flair, vol_t2, vol_t1ce], axis=0)
        return vols, img_flair

    # --------------------------------------------------------
    #   预处理：手动裁剪 + 归一化
    # --------------------------------------------------------
    def preprocess(self, t1, flair, t1ce, t2):
        vols_full, ref_flair_img = self.load_4_modalities(t1, flair, t1ce, t2)

        bbox = _compute_foreground_bbox(vols_full)
        vols_crop = _crop_with_bbox(vols_full, bbox)
        vols_crop = _normalize_intensity_nonzero_channelwise(vols_crop)

        img_tensor = torch.from_numpy(vols_crop).unsqueeze(0).to(self.device, dtype=torch.float32)
        flair_full = vols_full[1]  # 原始 FLAIR，全空间，供 UI 和回填后展示使用

        return img_tensor, bbox, flair_full, ref_flair_img

    # --------------------------------------------------------
    #   推理
    # --------------------------------------------------------
    def predict_volume(self, t1, flair, t1ce, t2):
        self.load_model()
        inputs, bbox, flair_full, ref_flair_img = self.preprocess(t1, flair, t1ce, t2)

        with torch.inference_mode():
            logits = self.inferer(inputs, self.model)
            probs = torch.sigmoid(logits)

        output = (probs > 0.5).cpu().numpy().astype(np.uint8)
        pred = output[0]  # (3, D', H', W')

        # pred[0] = TC, pred[1] = WT, pred[2] = ET
        tc_mask = pred[0] == 1
        wt_mask = pred[1] == 1
        et_mask = pred[2] == 1

        # 生成单标签分割：
        # 2 = WT \ TC
        # 1 = TC \ ET
        # 4 = ET
        seg_crop = np.zeros_like(pred[0], dtype=np.uint8)
        seg_crop[wt_mask] = 2
        seg_crop[tc_mask] = 1
        seg_crop[et_mask] = 4

        # 回填到原始 FLAIR 空间
        seg_full = _restore_seg_to_full(seg_crop, flair_full.shape, bbox)

        return seg_full, flair_full.astype(np.float32), ref_flair_img, bbox

    # --------------------------------------------------------
    #   NIfTI 保存（使用原始 FLAIR 的空间信息）
    # --------------------------------------------------------
    def save_seg_nifti(self, seg, ref_img, out_path):
        seg_img = sitk.GetImageFromArray(seg)
        seg_img.SetSpacing(ref_img.GetSpacing())
        seg_img.SetDirection(ref_img.GetDirection())
        seg_img.SetOrigin(ref_img.GetOrigin())

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sitk.WriteImage(seg_img, out_path)
        return out_path

    # --------------------------------------------------------
    #   选一个“代表性 slice”
    # --------------------------------------------------------
    def pick_representative_slice(self, seg, flair_np, min_brain_ratio=0.2):
        D, H, W = seg.shape
        total_pixels = H * W

        tumor_counts = (seg > 0).sum(axis=(1, 2))
        brain_counts = (np.abs(flair_np) > 1e-6).sum(axis=(1, 2))
        brain_ratio = brain_counts / float(total_pixels)

        candidates = np.where(
            (tumor_counts > 0) & (brain_ratio >= min_brain_ratio)
        )[0]

        if len(candidates) > 0:
            return int(candidates[np.argmax(tumor_counts[candidates])])
        return D // 2

    # --------------------------------------------------------
    #   按 slice 渲染 PNG
    # --------------------------------------------------------
    def render_slice_png(self, seg, flair_np, z_idx, out_path):
        D = seg.shape[0]
        z_idx = int(np.clip(z_idx, 0, D - 1))

        flair_slice = flair_np[z_idx]
        seg_slice = seg[z_idx]

        vmin, vmax = np.percentile(flair_slice, 1), np.percentile(flair_slice, 99)
        flair_clip = np.clip(flair_slice, vmin, vmax)
        flair_norm = (flair_clip - vmin) / (vmax - vmin + 1e-7)

        flair_rgb = np.stack([flair_norm] * 3, axis=-1).astype(np.float32)

        # 这里的语义是：
        # label 1 = TC\ET 的核心区域（显示为 TC 色）
        # label 2 = WT\TC 的外围区域（显示为 WT 色）
        # label 4 = ET
        flair_rgb[seg_slice == 1] = [1, 159 / 255, 255 / 255]   # TC
        flair_rgb[seg_slice == 2] = [1, 255 / 255, 131 / 255]   # WT
        flair_rgb[seg_slice == 4] = [255 / 255, 0, 170 / 255]   # ET

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.imsave(out_path, flair_rgb)
        return out_path


# ============================================================
#   全局模型
# ============================================================
_global_model = None


def get_brain_model() -> BrainSegmentationModel3D:
    global _global_model
    if _global_model is None:
        _global_model = BrainSegmentationModel3D()
        _global_model.load_model()
    return _global_model


# ============================================================
#   对外接口
# ============================================================
def segment_nifti(t1, flair, t1ce, t2, image_store, output_dir="./brain_outputs"):
    """
    - 负责：推理 + 回填到原始 FLAIR 空间 + 缓存 + NIfTI 保存
    """
    model = get_brain_model()

    case_id = _extract_case_id(t1)
    seg_path = os.path.join(output_dir, f"{case_id}_seg.nii.gz")

    seg, flair_np, ref_flair_img, bbox = model.predict_volume(t1, flair, t1ce, t2)
    model.save_seg_nifti(seg, ref_flair_img, seg_path)

    # 缓存给滑条和上层系统
    image_store["seg_array"] = seg
    image_store["flair_array"] = flair_np
    image_store["num_slices"] = seg.shape[0]
    image_store["crop_bbox"] = bbox

    # 初始化 slice
    z0 = model.pick_representative_slice(seg, flair_np)
    image_store["init_slice"] = z0

    init_png = os.path.join(output_dir, f"{case_id}_init_slice.png")
    model.render_slice_png(seg, flair_np, z0, init_png)
    image_store["preview_png"] = init_png

    return init_png, seg_path


def analyze_tumor_volume(seg, reference_nii_path):
    """
    计算 ET / TC / WT 的体积与占比
    seg: numpy array (D, H, W), values {0,1,2,4}
    reference_nii_path: 参考 NIfTI（需具有正确 spacing）
    """

    img = sitk.ReadImage(reference_nii_path)
    sx, sy, sz = img.GetSpacing()
    voxel_volume = sx * sy * sz  # mm³

    stats = {}

    # ET = label 4
    stats["ET_mm3"] = np.sum(seg == 4) * voxel_volume

    # TC = label 1 + label 4
    stats["TC_mm3"] = np.sum((seg == 1) | (seg == 4)) * voxel_volume

    # WT = 所有肿瘤相关标签
    stats["WT_mm3"] = np.sum(seg > 0) * voxel_volume

    if stats["WT_mm3"] > 0:
        stats["ET_ratio"] = stats["ET_mm3"] / stats["WT_mm3"]
        stats["TC_ratio"] = stats["TC_mm3"] / stats["WT_mm3"]
    else:
        stats["ET_ratio"] = 0.0
        stats["TC_ratio"] = 0.0

    return stats