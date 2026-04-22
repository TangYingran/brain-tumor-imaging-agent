import os
import uuid
import numpy as np
import SimpleITK as sitk

from model_inference import segment_nifti, analyze_tumor_volume
from rag.router import answer_with_router
from config import LABEL_COLOR_MAP
from long_term_memory import get_long_term_memory
from report_generator import generate_structured_report

CASE_STORE = {}


# ============================================================
# 内部辅助函数
# ============================================================
def _normalize_patient_id(patient_id: str) -> str:
    patient_id = (patient_id or "").strip()
    return patient_id if patient_id else "P000"


def _build_volume_interpretation(wt_cm3: float, tc_cm3: float, et_cm3: float) -> str:
    lines = []

    if wt_cm3 <= 0:
        lines.append("当前几乎未检测到有效整体肿瘤体积。")
    elif wt_cm3 < 5:
        lines.append("整体肿瘤体积较小。")
    elif wt_cm3 < 20:
        lines.append("整体肿瘤体积处于中等范围。")
    else:
        lines.append("整体肿瘤体积偏大。")

    if wt_cm3 > 0:
        et_ratio = et_cm3 / wt_cm3
        if et_ratio < 0.05:
            lines.append("增强肿瘤占比偏低，活跃区域相对局限。")
        elif et_ratio < 0.15:
            lines.append("增强肿瘤占比中等。")
        else:
            lines.append("增强肿瘤占比较高，活跃区域较突出。")

    return " ".join(lines)


def _build_volume_conclusion(wt_cm3: float, tc_cm3: float, et_cm3: float) -> str:
    """
    生成更适合写进报告的结论 / 总结段。
    """
    if wt_cm3 <= 0:
        return "当前病例未见明确有效整体肿瘤负荷，建议结合原始影像与分割结果进一步复核。"

    lines = []

    # 1. 整体肿瘤负荷
    if wt_cm3 < 5:
        lines.append("当前病例整体肿瘤负荷较轻。")
    elif wt_cm3 < 20:
        lines.append("当前病例整体肿瘤负荷处于中等水平。")
    else:
        lines.append("当前病例整体肿瘤负荷偏高。")

    # 2. 增强活跃区域
    et_ratio = et_cm3 / wt_cm3 if wt_cm3 > 0 else 0.0
    if et_ratio < 0.05:
        lines.append("增强活跃区域相对不明显。")
    elif et_ratio < 0.15:
        lines.append("增强活跃区域存在但不算突出。")
    else:
        lines.append("增强活跃区域较为明显。")

    # 3. 是否重点关注 TC / ET
    tc_ratio = tc_cm3 / wt_cm3 if wt_cm3 > 0 else 0.0
    if tc_ratio >= 0.2 and et_ratio >= 0.15:
        lines.append("建议同时重点关注肿瘤核心（TC）和增强肿瘤（ET）的变化。")
    elif tc_ratio >= 0.2:
        lines.append("建议重点关注肿瘤核心（TC）的变化。")
    elif et_ratio >= 0.15:
        lines.append("建议重点关注增强肿瘤（ET）的变化。")
    else:
        lines.append("当前核心病灶和增强区域负荷相对有限。")

    return "".join(lines)


def _needs_memory_enhancement(question: str) -> bool:
    """
    只有在明显涉及历史比较/既往信息时，才把长期记忆拼进去，
    避免普通定义型问题被历史上下文干扰。
    """
    q = (question or "").strip()
    keywords = [
        "上次", "之前", "既往", "历史", "变化", "趋势",
        "对比", "比较", "这次", "上一次", "有没有增加",
        "有没有变大", "是否加重", "有没有好转"
    ]
    return any(k in q for k in keywords)


def _format_patient_memory_context(profile: dict, history: list) -> str:
    if not history:
        return ""

    lines = ["患者历史上下文："]

    latest_case_id = profile.get("latest_case_id", "")
    latest_summary = profile.get("latest_summary", "")
    num_cases = profile.get("num_cases", 0)
    latest_report_summary = profile.get("latest_report_summary", "")

    if latest_case_id:
        lines.append(f"- 最近病例 ID：{latest_case_id}")
    if num_cases:
        lines.append(f"- 历史病例数：{num_cases}")
    if latest_summary:
        lines.append(f"- 最近摘要：{latest_summary}")
    if latest_report_summary:
        lines.append(f"- 最近报告摘要：{latest_report_summary}")

    lines.append("- 相关历史记录：")
    for i, item in enumerate(history, start=1):
        text = item.get("text", "")
        score = item.get("score", 0.0)
        memory_type = item.get("memory_type", "generic")
        lines.append(f"  {i}. [{memory_type}] (score={score:.4f}) {text}")

    return "\n".join(lines)


def _compare_numeric(current: float, previous: float, name: str, unit: str = "cm³") -> str:
    diff = current - previous
    if abs(diff) < 1e-6:
        return f"{name}与上次相比基本无明显变化。"
    if diff > 0:
        return f"{name}较上次增加 {abs(diff):.2f} {unit}。"
    return f"{name}较上次减少 {abs(diff):.2f} {unit}。"


def _compare_ratio(current: float, previous: float, name: str) -> str:
    diff = current - previous
    if abs(diff) < 1e-9:
        return f"{name}与上次相比基本无明显变化。"

    diff_pct = abs(diff) * 100.0
    if diff > 0:
        return f"{name}较上次上升 {diff_pct:.2f} 个百分点。"
    return f"{name}较上次下降 {diff_pct:.2f} 个百分点。"


# ============================================================
# 1. 分割核心
# ============================================================
def run_segmentation_core(
    case_id: str,
    t1: str,
    flair: str,
    t1ce: str,
    t2: str,
    patient_id: str = "",
) -> dict:
    patient_id = _normalize_patient_id(patient_id)

    image_store = {
        "t1": t1,
        "flair": flair,
        "t1ce": t1ce,
        "t2": t2,
        "patient_id": patient_id,
        "case_id": case_id,
        "preview_png": None,
        "seg_nifti": None,
        "seg_array": None,
        "flair_array": None,
        "num_slices": None,
        "init_slice": None,
    }

    for f in [t1, flair, t1ce, t2]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到文件: {f}")

    preview_png, seg_nifti = segment_nifti(
        t1=t1,
        flair=flair,
        t1ce=t1ce,
        t2=t2,
        image_store=image_store,
    )

    image_store["preview_png"] = preview_png
    image_store["seg_nifti"] = seg_nifti
    CASE_STORE[case_id] = image_store

    # 注意：精简版长期记忆不再写 segmentation_result
    return {
        "patient_id": patient_id,
        "case_id": case_id,
        "preview_png": preview_png,
        "seg_nifti": seg_nifti,
        "num_slices": image_store.get("num_slices"),
        "init_slice": image_store.get("init_slice"),
        "message": "脑肿瘤 3D 分割已完成"
    }


# ============================================================
# 2. 体积分析核心
# ============================================================
def analyze_volume_core(case_id: str) -> dict:
    image_store = CASE_STORE.get(case_id)
    if not image_store:
        raise ValueError(f"未找到 case_id={case_id} 对应的缓存结果")

    patient_id = _normalize_patient_id(image_store.get("patient_id"))
    seg_nifti = image_store.get("seg_nifti")
    flair_path = image_store.get("flair")

    if not seg_nifti or not os.path.exists(seg_nifti):
        raise ValueError("当前还没有可用的分割结果 NIfTI 文件")

    seg_img = sitk.ReadImage(seg_nifti)
    seg_arr = sitk.GetArrayFromImage(seg_img)

    stats = analyze_tumor_volume(seg_arr, seg_nifti)

    et_cm3 = stats["ET_mm3"] / 1000.0
    tc_cm3 = stats["TC_mm3"] / 1000.0
    wt_cm3 = stats["WT_mm3"] / 1000.0

    et_ratio = stats.get("ET_ratio", 0.0)
    tc_ratio = stats.get("TC_ratio", 0.0)

    wt_brain_ratio = 0.0
    if flair_path and os.path.exists(flair_path):
        flair_img = sitk.ReadImage(flair_path)
        flair_arr = sitk.GetArrayFromImage(flair_img)
        brain_mask = (np.abs(flair_arr) > 1e-6)
        brain_voxel_count = brain_mask.sum()
        wt_voxel_count = (seg_arr == 2).sum()

        if brain_voxel_count > 0:
            wt_brain_ratio = wt_voxel_count / float(brain_voxel_count)

    interpretation = _build_volume_interpretation(wt_cm3, tc_cm3, et_cm3)
    conclusion = _build_volume_conclusion(wt_cm3, tc_cm3, et_cm3)

    result = {
        "patient_id": patient_id,
        "case_id": case_id,
        "ET_cm3": round(et_cm3, 2),
        "TC_cm3": round(tc_cm3, 2),
        "WT_cm3": round(wt_cm3, 2),
        "ET_WT_ratio": et_ratio,
        "TC_WT_ratio": tc_ratio,
        "WT_brain_ratio": wt_brain_ratio,
        "interpretation": interpretation,
        "conclusion": conclusion,
    }

    # 写入长期记忆：仅保留 volume_analysis
    try:
        memory_store = get_long_term_memory()
        memory_store.add_volume_analysis_memory(
            patient_id=patient_id,
            case_id=case_id,
            WT_cm3=result["WT_cm3"],
            TC_cm3=result["TC_cm3"],
            ET_cm3=result["ET_cm3"],
            ET_WT_ratio=result["ET_WT_ratio"],
            TC_WT_ratio=result["TC_WT_ratio"],
            WT_brain_ratio=result["WT_brain_ratio"],
            interpretation=interpretation,
        )
    except Exception as e:
        print(f"[WARN] 写入 volume_analysis 长期记忆失败：{e}")

    return result


# ============================================================
# 3. 分割结果语义解释核心
# ============================================================
def analyze_result_core(case_id: str) -> dict:
    image_store = CASE_STORE.get(case_id)
    if not image_store or not image_store.get("preview_png"):
        raise ValueError("当前还没有可供分析的分割结果")

    patient_id = _normalize_patient_id(image_store.get("patient_id"))

    m = LABEL_COLOR_MAP
    result = {
        "patient_id": patient_id,
        "case_id": case_id,
        "ET": {
            "label": 4,
            "color": m["ET"]["color_name"],
            "meaning": "Enhancing Tumor，增强肿瘤"
        },
        "TC": {
            "label": 1,
            "color": m["TC"]["color_name"],
            "meaning": "Tumor Core，肿瘤核心"
        },
        "WT": {
            "label": 2,
            "color": m["WT"]["color_name"],
            "meaning": "Whole Tumor，整体肿瘤"
        }
    }

    # 注意：精简版长期记忆不再写 result_analysis
    return result


# ============================================================
# 4. 报告生成核心
# ============================================================
def generate_report_core(case_id: str, extra_note: str = "") -> dict:
    """
    基于当前病例的体积分析结果生成结构化报告。
    这里只负责：
    - 调 analyze_volume_core(case_id)
    - 调 report_generator.generate_structured_report(volume_result)
    - 将报告摘要写入长期记忆（structured_report）
    - 返回可直接给 Agent 展示的报告文本
    """
    image_store = CASE_STORE.get(case_id)
    if not image_store:
        raise ValueError(f"未找到 case_id={case_id} 对应的缓存结果")

    patient_id = _normalize_patient_id(image_store.get("patient_id"))

    volume_result = analyze_volume_core(case_id)

    report = generate_structured_report(
        volume_result=volume_result,
        extra_note=extra_note,
    )

    # 写入长期记忆：仅保留 structured_report
    try:
        memory_store = get_long_term_memory()
        memory_store.add_report_memory(
            patient_id=patient_id,
            case_id=case_id,
            report_text=report.get("text", ""),
            report_type="structured_report",
        )
    except Exception as e:
        print(f"[WARN] 写入 structured_report 长期记忆失败：{e}")

    return {
        "patient_id": patient_id,
        "case_id": case_id,
        "generation_time_sec": report.get("generation_time_sec"),
        "report_text": report.get("text", ""),
        "report_markdown": report.get("markdown", ""),
        "report": report,
    }


# ============================================================
# 5. 肿瘤变化分析核心
# ============================================================
def compare_tumor_change_core(case_id: str) -> dict:
    """
    比较当前病例与该患者上一份病例的肿瘤体积变化。
    依赖长期记忆中的 volume_analysis 记录。

    返回：
        {
            "patient_id": ...,
            "current_case_id": ...,
            "previous_case_id": ...,
            "current_volume": {...},
            "previous_volume": {...},
            "comparison_text": ...
        }
    """
    image_store = CASE_STORE.get(case_id)
    if not image_store:
        raise ValueError(f"未找到 case_id={case_id} 对应的缓存结果")

    patient_id = _normalize_patient_id(image_store.get("patient_id"))
    if patient_id == "P000":
        raise ValueError("当前没有有效 patient_id，无法进行历史变化分析")

    memory_store = get_long_term_memory()
    all_memories = memory_store.get_patient_memories(patient_id)

    # 只保留 volume_analysis
    volume_memories = [
        r for r in all_memories
        if r.get("memory_type") == "volume_analysis"
    ]

    # 先按 case_id 去重：保留每个 case_id 最新的一条
    unique_case_memories = []
    seen_case_ids = set()
    for rec in volume_memories:
        cid = rec.get("case_id", "")
        if cid in seen_case_ids:
            continue
        seen_case_ids.add(cid)
        unique_case_memories.append(rec)

    # 当前病例的 volume_analysis 是否已存在
    current_memory = None
    for rec in unique_case_memories:
        if rec.get("case_id") == case_id:
            current_memory = rec
            break

    # 如果当前病例还没有 volume_analysis，就先计算一次
    if current_memory is None:
        current_result = analyze_volume_core(case_id)
        current_memory = {
            "case_id": case_id,
            "memory_type": "volume_analysis",
            "metadata": {
                "WT_cm3": current_result.get("WT_cm3"),
                "TC_cm3": current_result.get("TC_cm3"),
                "ET_cm3": current_result.get("ET_cm3"),
                "ET_WT_ratio": current_result.get("ET_WT_ratio"),
                "TC_WT_ratio": current_result.get("TC_WT_ratio"),
                "WT_brain_ratio": current_result.get("WT_brain_ratio"),
                "interpretation": current_result.get("interpretation", ""),
                "conclusion": current_result.get("conclusion", ""),
            }
        }

    # 找上一份不同 case_id 的记录
    previous_memory = None
    for rec in unique_case_memories:
        if rec.get("case_id") != case_id:
            previous_memory = rec
            break

    current_meta = current_memory.get("metadata", {})

    # 如果没有上一份病例，只返回当前结果提示
    if previous_memory is None:
        comparison_text = (
            "当前仅检测到这一份病例的体积分析结果，暂时无法进行“与上次相比”的变化分析。\n\n"
            f"当前结果：WT={current_meta.get('WT_cm3', 'N/A')} cm³，"
            f"TC={current_meta.get('TC_cm3', 'N/A')} cm³，"
            f"ET={current_meta.get('ET_cm3', 'N/A')} cm³。"
        )

        return {
            "patient_id": patient_id,
            "current_case_id": case_id,
            "previous_case_id": None,
            "current_volume": current_meta,
            "previous_volume": None,
            "comparison_text": comparison_text,
        }

    previous_meta = previous_memory.get("metadata", {})

    current_wt = float(current_meta.get("WT_cm3", 0) or 0)
    current_tc = float(current_meta.get("TC_cm3", 0) or 0)
    current_et = float(current_meta.get("ET_cm3", 0) or 0)

    previous_wt = float(previous_meta.get("WT_cm3", 0) or 0)
    previous_tc = float(previous_meta.get("TC_cm3", 0) or 0)
    previous_et = float(previous_meta.get("ET_cm3", 0) or 0)

    current_et_ratio = float(current_meta.get("ET_WT_ratio", 0) or 0)
    previous_et_ratio = float(previous_meta.get("ET_WT_ratio", 0) or 0)

    current_tc_ratio = float(current_meta.get("TC_WT_ratio", 0) or 0)
    previous_tc_ratio = float(previous_meta.get("TC_WT_ratio", 0) or 0)

    current_wt_brain_ratio = float(current_meta.get("WT_brain_ratio", 0) or 0)
    previous_wt_brain_ratio = float(previous_meta.get("WT_brain_ratio", 0) or 0)

    comparison_lines = [
        f"患者 {patient_id} 当前病例（{case_id}）与上一病例（{previous_memory.get('case_id')}）相比：",
        _compare_numeric(current_wt, previous_wt, "WT（整体肿瘤）体积"),
        _compare_numeric(current_tc, previous_tc, "TC（肿瘤核心）体积"),
        _compare_numeric(current_et, previous_et, "ET（增强肿瘤）体积"),
        _compare_ratio(current_et_ratio, previous_et_ratio, "ET / WT 占比"),
        _compare_ratio(current_tc_ratio, previous_tc_ratio, "TC / WT 占比"),
        _compare_ratio(current_wt_brain_ratio, previous_wt_brain_ratio, "WT / 全脑体积占比"),
    ]

    if current_wt > previous_wt and current_et >= previous_et:
        comparison_lines.append("总体来看，本次整体肿瘤负荷较上次增加，增强活跃区域未见下降。")
    elif current_wt < previous_wt and current_et <= previous_et:
        comparison_lines.append("总体来看，本次整体肿瘤负荷较上次减轻，增强区域也呈下降趋势。")
    else:
        comparison_lines.append("总体来看，本次与上次相比存在部分指标升高、部分指标下降，需结合具体影像分布综合判断。")

    return {
        "patient_id": patient_id,
        "current_case_id": case_id,
        "previous_case_id": previous_memory.get("case_id"),
        "current_volume": current_meta,
        "previous_volume": previous_meta,
        "comparison_text": "\n".join(comparison_lines),
    }


# ============================================================
# 6. 知识问答核心
# ============================================================
def query_knowledge_core(question: str, patient_id: str = "") -> str:
    """
    知识问答统一入口：
    - 先尝试读取患者长期记忆（现在只会读取 volume_analysis + structured_report）
    - 对“历史比较/趋势分析”类问题进行记忆增强
    - 再走 router（内部自动决定 rule_kb 还是 rag_kb）
    """
    patient_id = _normalize_patient_id(patient_id)

    enhanced_question = question

    # 仅在“历史比较/趋势分析”类问题中，拼接患者长期记忆
    try:
        if patient_id and patient_id != "P000" and _needs_memory_enhancement(question):
            memory_store = get_long_term_memory()
            history = memory_store.search(
                patient_id=patient_id,
                query=question,
                topk=3
            )
            profile = memory_store.build_patient_profile(patient_id)

            memory_context = _format_patient_memory_context(profile, history)
            if memory_context:
                enhanced_question = (
                    f"{question}\n\n"
                    f"{memory_context}\n\n"
                    "请在回答时结合以上患者历史信息进行说明。"
                )
    except Exception as e:
        print(f"[WARN] 检索患者长期记忆失败：{e}")

    result = answer_with_router(
        question=enhanced_question,
        index_dir="/root/code/agent1/rag/index",
        embedding_model="/root/models/bge-m3",
        reranker_model="/root/models/bge-reranker-v2-m3",
        device="cuda",          # 会自动降级到 cpu
        local_files_only=True,
        dense_topk=20,
        sparse_topk=20,
        candidate_topk=20,
        final_topk=5,
        prefer_jieba=True,
    )

    answer = result.get("answer", "未获得有效回答。")
    route = result.get("route", "unknown")
    refs = result.get("references", [])

    if refs:
        ref_lines = []
        for i, ref in enumerate(refs[:3], start=1):
            title = ref.get("title", "")
            topic = ref.get("topic", "")
            preview = ref.get("preview", "")
            ref_lines.append(
                f"{i}. {title}（topic={topic}）\n   {preview}"
            )

        return (
            f"{answer}\n\n"
            f"【知识路由】{route}\n"
            f"【参考片段】\n" + "\n".join(ref_lines)
        )

    return f"{answer}\n\n【知识路由】{route}"