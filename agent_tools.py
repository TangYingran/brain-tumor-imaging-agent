"""
LangChain 工具定义
适用于 3D 脑肿瘤分割（SwinUNETR）

当前版本说明：
- 核心业务逻辑统一放在 brain_core.py
- 本文件仅负责：
    1) 将核心函数封装为 LangChain Tool
    2) 从 Gradio 的 image_store 中读取输入
    3) 将 brain_core 中的结果同步回当前界面 image_store
"""

import os
import uuid
from langchain.tools import Tool

from brain_core import (
    run_segmentation_core,
    analyze_volume_core,
    analyze_result_core,
    generate_report_core,
    compare_tumor_change_core,
    query_knowledge_core,
    CASE_STORE,
)


# -------------------------------------------------------------
# 1. 脑肿瘤分割工具
# -------------------------------------------------------------
def create_brain_segmentation_tool(image_store):
    def run_segmentation(_: str) -> str:
        """
        执行 3D 脑肿瘤分割（使用 image_store 中已上传的四模态路径）
        """

        # 0) 检查 patient_id
        patient_id = (image_store.get("patient_id") or "").strip()
        if not patient_id:
            return (
                "当前还没有有效的患者 ID。\n"
                "请先在左侧界面输入 patient_id（例如：P001），"
                "然后再上传四模态图像。"
            )

        # 1) 确认四个模态都已上传
        modal_keys = ["t1", "flair", "t1ce", "t2"]
        missing = [k for k in modal_keys if not image_store.get(k)]
        if missing:
            return (
                "错误：缺少以下模态文件："
                + "、".join(missing)
                + "。\n请在左侧界面上传完整的 T1 / FLAIR / T1CE / T2 NIfTI 后，"
                  "再在聊天框中让我执行脑肿瘤分割，例如：`请进行脑肿瘤三维分割`。"
            )

        t1 = image_store["t1"]
        flair = image_store["flair"]
        t1ce = image_store["t1ce"]
        t2 = image_store["t2"]

        # 2) 检查路径是否存在
        for f in [t1, flair, t1ce, t2]:
            if not os.path.exists(f):
                return f"错误：找不到文件：{f}，请确认上传是否成功。"

        try:
            # 3) 准备 case_id（用于 core 层缓存）
            case_id = image_store.get("case_id") or str(uuid.uuid4())

            # 4) 调用核心分割逻辑（传入 patient_id）
            result = run_segmentation_core(
                patient_id=patient_id,
                case_id=case_id,
                t1=t1,
                flair=flair,
                t1ce=t1ce,
                t2=t2,
            )

            # 5) 将 case_id 写回当前 UI 状态
            image_store["case_id"] = case_id

            # 6) 将 core 层缓存同步回当前 UI 的 image_store
            core_store = CASE_STORE.get(case_id, {})
            sync_keys = [
                "patient_id",
                "preview_png",
                "seg_nifti",
                "seg_array",
                "flair_array",
                "num_slices",
                "init_slice",
            ]
            for k in sync_keys:
                image_store[k] = core_store.get(k, image_store.get(k))

            num_slices = result.get("num_slices")
            init_slice = result.get("init_slice")

            extra_slider_info = ""
            if num_slices is not None and init_slice is not None:
                extra_slider_info = (
                    f"\n当前处理后空间的体数据深度为 {num_slices} 个切片，"
                    f"默认展示的是第 {init_slice} 层。你可以通过界面下方滑动条查看其他切片。"
                )

            return f"""
脑肿瘤 3D 分割已完成！

- patient_id：{result.get("patient_id")}
- case_id：{result.get("case_id")}
- 初始预览 PNG 路径：{result.get("preview_png")}
- 分割结果 NIfTI 路径：{result.get("seg_nifti")}{extra_slider_info}

你可以继续让我：
- “分析肿瘤体积”
- “分析分割结果”
- “生成结构化报告”
- “分析肿瘤变化”
- “解释 ET / TC / WT 的含义”
- “查询相关医学知识”
""".strip()

        except Exception as e:
            return f"分割过程中发生错误：{str(e)}"

    return Tool(
        name="brain_tumor_segmentation",
        description=(
            "对当前已上传的四模态脑肿瘤 MRI（T1 / FLAIR / T1CE / T2）执行 3D 分割。"
            "输入任意中文描述，例如“请分割脑肿瘤”“请进行脑肿瘤三维分割”即可。"
        ),
        func=run_segmentation,
    )


# -------------------------------------------------------------
# 2. 肿瘤体积 / 占比分析工具（基于分割结果）
# -------------------------------------------------------------
def create_volume_analysis_tool(image_store):
    def analyze(_: str) -> str:
        """
        基于 core 层缓存的 case_id，对 ET / TC / WT 进行体积与占比分析。
        工具层只负责调用 core 和格式化展示，不再重复业务规则。
        """
        case_id = image_store.get("case_id")
        if not case_id:
            return (
                "当前还没有可用的分割结果。\n"
                "请先在聊天框中让我执行一次脑肿瘤分割，例如：`请进行脑肿瘤三维分割`，"
                "分割完成后再让我分析肿瘤体积。"
            )

        try:
            result = analyze_volume_core(case_id)

            patient_id = result.get("patient_id", "未知")
            et_cm3 = result.get("ET_cm3", 0.0)
            tc_cm3 = result.get("TC_cm3", 0.0)
            wt_cm3 = result.get("WT_cm3", 0.0)

            et_ratio = result.get("ET_WT_ratio", 0.0)
            tc_ratio = result.get("TC_WT_ratio", 0.0)
            wt_brain_ratio = result.get("WT_brain_ratio", 0.0)

            interpretation_text = result.get("interpretation", "暂无自动解读。")

            return (
                f"📊 **肿瘤体积定量分析结果**（患者 {patient_id}，基于当前分割掩膜，单位均为近似值，仅作科研/教学参考）：\n\n"
                f"- ET（增强肿瘤）体积：**{et_cm3:.2f} cm³**\n"
                f"- TC（肿瘤核心）体积：**{tc_cm3:.2f} cm³**\n"
                f"- WT（整体肿瘤）体积：**{wt_cm3:.2f} cm³**\n\n"
                f"- ET / WT 体积占比：**{et_ratio:.2%}**\n"
                f"- TC / WT 体积占比：**{tc_ratio:.2%}**\n"
                f"- WT / 全脑体积占比：**{wt_brain_ratio:.2%}**\n\n"
                f"简要解读（仅作科研 / 教学参考）：\n- {interpretation_text}\n\n"
                "其中：\n"
                "- WT 反映了肿瘤及其水肿对脑组织整体影响的范围；\n"
                "- TC 反映肿瘤主体病灶的体积负荷；\n"
                "- ET 的体积和占比常用于评估肿瘤活跃区域的大小及治疗反应；\n"
                "- WT / 全脑占比可以粗略反映肿瘤对整个脑组织体积负荷的相对大小。\n"
            )

        except Exception as e:
            return f"体积分析过程中发生错误：{str(e)}"

    return Tool(
        name="analyze_tumor_volume",
        description="在已有分割结果的基础上，定量分析 ET / TC / WT 的体积（cm³）、相对占比，以及 WT / 全脑体积占比，并返回结构化解读。",
        func=analyze,
    )


# -------------------------------------------------------------
# 3. 分割结果分析（语义解释）
# -------------------------------------------------------------
def create_result_analysis_tool(image_store):
    def analyze(_: str) -> str:
        case_id = image_store.get("case_id")
        if not case_id:
            return (
                "当前还没有可供分析的分割结果。\n"
                "请先在右侧聊天框中让我执行一次脑肿瘤分割，例如：`请进行脑肿瘤三维分割`，"
                "分割完成后再让我分析结果。"
            )

        try:
            result = analyze_result_core(case_id)

            return f"""
这是脑肿瘤分割结果中各标签的大致含义（基于 BraTS 任务设定）：

- **ET ({result["ET"]["meaning"]}，标签 {result["ET"]["label"]}，{result["ET"]["color"]})**
  通常对应在 T1CE 序列中表现为明显强化的肿瘤活跃区域。

- **TC ({result["TC"]["meaning"]}，标签 {result["TC"]["label"]}，{result["TC"]["color"]})**
  一般表示肿瘤内部最核心的实体病灶区域。

- **WT ({result["WT"]["meaning"]}，标签 {result["WT"]["label"]}，{result["WT"]["color"]})**
  表示包含肿瘤核心及周围异常区域在内的整体肿瘤范围。

你可以通过界面中的切片滑条逐层查看不同层面的分割结果。
如果需要，我也可以进一步结合当前分割结果，从“病灶范围、可能浸润程度、对治疗规划的潜在影响”等角度做更详细说明（仅作科研 / 教学参考）。
""".strip()

        except Exception as e:
            return f"结果分析过程中发生错误：{str(e)}"

    return Tool(
        name="analyze_segmentation_result",
        description="分析脑肿瘤分割结果，并解释 ET / TC / WT 等分割区域的含义与大致临床意义。",
        func=analyze,
        return_direct=True,
    )


# -------------------------------------------------------------
# 4. 报告生成工具
# -------------------------------------------------------------
def create_report_generation_tool(image_store):
    def generate(_: str) -> str:
        """
        基于当前病例的体积分析结果生成结构化报告。
        直接返回报告文本，在对话中显示。
        """
        case_id = image_store.get("case_id")
        if not case_id:
            return (
                "当前还没有可用的分割结果，暂时无法生成结构化报告。\n"
                "请先在聊天框中让我执行一次脑肿瘤分割，例如：`请进行脑肿瘤三维分割`，"
                "分割完成后再让我生成报告。"
            )

        try:
            result = generate_report_core(case_id)

            patient_id = result.get("patient_id", "未知")
            generation_time = result.get("generation_time_sec", "N/A")
            report_text = result.get("report_text", "").strip()

            if not report_text:
                return "报告生成失败：未获得有效报告内容。"

            return (
                f"📝 **结构化报告已生成**（患者 {patient_id}，耗时 {generation_time} 秒）\n\n"
                f"{report_text}"
            )

        except Exception as e:
            return f"报告生成过程中发生错误：{str(e)}"

    return Tool(
        name="generate_structured_report",
        description=(
            "基于当前病例的体积分析结果生成结构化报告。"
            "当用户要求“生成报告”“出具结构化报告”“总结当前病例”时使用。"
        ),
        func=generate,
        return_direct=True,
    )


# -------------------------------------------------------------
# 5. 肿瘤变化分析工具
# -------------------------------------------------------------
def create_tumor_change_analysis_tool(image_store):
    def analyze(_: str) -> str:
        """
        基于当前病例与历史 volume_analysis 记录，分析肿瘤变化。
        """
        case_id = image_store.get("case_id")
        if not case_id:
            return (
                "当前还没有可用的分割结果，暂时无法分析肿瘤变化。\n"
                "请先完成一次脑肿瘤分割和体积分析，再让我比较变化。"
            )

        try:
            result = compare_tumor_change_core(case_id)

            patient_id = result.get("patient_id", "未知")
            current_case_id = result.get("current_case_id", "")
            previous_case_id = result.get("previous_case_id", None)
            comparison_text = result.get("comparison_text", "").strip()

            if not previous_case_id:
                return (
                    f"📈 **肿瘤变化分析**（患者 {patient_id}）\n\n"
                    f"当前病例：{current_case_id}\n\n"
                    f"{comparison_text}"
                )

            return (
                f"📈 **肿瘤变化分析**（患者 {patient_id}）\n\n"
                f"- 当前病例：{current_case_id}\n"
                f"- 上一病例：{previous_case_id}\n\n"
                f"{comparison_text}"
            )

        except Exception as e:
            return f"肿瘤变化分析过程中发生错误：{str(e)}"

    return Tool(
        name="compare_tumor_change",
        description=(
            "比较当前病例与上一病例的肿瘤体积变化。"
            "当用户要求“分析肿瘤变化”“和上次相比有没有变大”“比较本次与上次结果”时使用。"
        ),
        func=analyze,
        return_direct=True,
    )


# -------------------------------------------------------------
# 6. 医学知识问答工具（脑肿瘤相关）
# -------------------------------------------------------------
def create_knowledge_query_tool(image_store):
    def query(question: str) -> str:
        try:
            patient_id = image_store.get("patient_id", "")
            return query_knowledge_core(
                question=question,
                patient_id=patient_id,
            )
        except Exception as e:
            return f"知识查询过程中发生错误：{str(e)}"

    return Tool(
        name="knowledge_query",
        description=(
            "查询与脑肿瘤、MRI 模态、多模态影像、分割结果及患者历史记录相关的医学知识。"
            "当用户询问 ET/TC/WT、MRI 模态、脑肿瘤治疗、药物、历史变化趋势等问题时使用。"
        ),
        func=query,
    )


# -------------------------------------------------------------
# 7. 工具列表
# -------------------------------------------------------------
def get_tools(image_store):
    return [
        create_brain_segmentation_tool(image_store),
        create_volume_analysis_tool(image_store),
        create_result_analysis_tool(image_store),
        create_report_generation_tool(image_store),
        create_tumor_change_analysis_tool(image_store),
        create_knowledge_query_tool(image_store),
    ]