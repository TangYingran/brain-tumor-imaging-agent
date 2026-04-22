# report_generator.py
# 作用：
# 1. 读取 analyze_volume_core() 的结果
# 2. 将已有体积分析结果组织成结构化报告
# 3. 展示 summary / quantitative_analysis / interpretation / conclusion

import time
from typing import Dict, Any


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def _fmt_num(x, ndigits=2):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def _fmt_ratio(x):
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.2%}"
    except Exception:
        return str(x)


def build_summary(volume_result: Dict[str, Any]) -> str:
    """
    根据 analyze_volume_core() 的结果生成摘要
    """
    case_id = volume_result.get("case_id", "unknown_case")
    wt = _fmt_num(volume_result.get("WT_cm3"))
    tc = _fmt_num(volume_result.get("TC_cm3"))
    et = _fmt_num(volume_result.get("ET_cm3"))
    interpretation = normalize_text(volume_result.get("interpretation", ""))

    summary = (
        f"病例 {case_id} 的体积分析结果显示："
        f"WT={wt} cm³，TC={tc} cm³，ET={et} cm³。"
    )

    if interpretation:
        summary += interpretation

    return summary


def build_quantitative_section(volume_result: Dict[str, Any]) -> str:
    wt = _fmt_num(volume_result.get("WT_cm3"))
    tc = _fmt_num(volume_result.get("TC_cm3"))
    et = _fmt_num(volume_result.get("ET_cm3"))

    et_ratio = _fmt_ratio(volume_result.get("ET_WT_ratio"))
    tc_ratio = _fmt_ratio(volume_result.get("TC_WT_ratio"))
    wt_brain_ratio = _fmt_ratio(volume_result.get("WT_brain_ratio"))

    lines = [
        f"- WT（整体肿瘤）体积：{wt} cm³",
        f"- TC（肿瘤核心）体积：{tc} cm³",
        f"- ET（增强肿瘤）体积：{et} cm³",
        f"- ET / WT 占比：{et_ratio}",
        f"- TC / WT 占比：{tc_ratio}",
        f"- WT / 全脑体积占比：{wt_brain_ratio}",
    ]
    return "\n".join(lines)


def generate_structured_report(
    volume_result: Dict[str, Any],
    extra_note: str = ""
) -> Dict[str, Any]:
    """
    根据 analyze_volume_core() 的结果组织报告
    """
    start = time.perf_counter()

    patient_id = volume_result.get("patient_id", "P000")
    case_id = volume_result.get("case_id", "unknown_case")
    interpretation = normalize_text(volume_result.get("interpretation", ""))
    conclusion = normalize_text(volume_result.get("conclusion", ""))

    report = {
        "patient_id": patient_id,
        "case_id": case_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "sections": {
            "summary": build_summary(volume_result),
            "quantitative_analysis": build_quantitative_section(volume_result),
            "interpretation": interpretation or "暂无自动解读。",
            "conclusion": conclusion or "暂无结论。",
            "extra_note": normalize_text(extra_note),
            "disclaimer": "本报告仅用于科研/教学辅助，不替代临床诊断与治疗决策。"
        },
    }

    elapsed = time.perf_counter() - start
    report["generation_time_sec"] = round(elapsed, 3)

    report["markdown"] = render_report_markdown(report)
    report["text"] = render_report_text(report)

    return report


def render_report_markdown(report: Dict[str, Any]) -> str:
    s = report["sections"]

    md = f"""# 脑肿瘤体积分析报告

## 1. 基本信息
- **patient_id**：{report.get("patient_id", "N/A")}
- **case_id**：{report.get("case_id", "N/A")}
- **生成时间**：{report.get("generated_at", "N/A")}
- **生成耗时**：{report.get("generation_time_sec", "N/A")} 秒

## 2. 摘要
{s.get("summary", "")}

## 3. 定量分析
{s.get("quantitative_analysis", "")}

## 4. 简要解读
{s.get("interpretation", "")}

## 5. 结论 / 总结
{s.get("conclusion", "")}
"""

    if s.get("extra_note"):
        md += f"\n## 6. 附加备注\n{s.get('extra_note')}\n"

    md += f"\n## 7. 免责声明\n{s.get('disclaimer', '')}\n"
    return md.strip()


def render_report_text(report: Dict[str, Any]) -> str:
    s = report["sections"]

    lines = [
        "脑肿瘤体积分析报告",
        "=" * 28,
        f"patient_id：{report.get('patient_id', 'N/A')}",
        f"case_id：{report.get('case_id', 'N/A')}",
        f"生成时间：{report.get('generated_at', 'N/A')}",
        f"生成耗时：{report.get('generation_time_sec', 'N/A')} 秒",
        "",
        "【摘要】",
        s.get("summary", ""),
        "",
        "【定量分析】",
        s.get("quantitative_analysis", ""),
        "",
        "【简要解读】",
        s.get("interpretation", ""),
        "",
        "【结论 / 总结】",
        s.get("conclusion", ""),
    ]

    if s.get("extra_note"):
        lines.extend(["", "【附加备注】", s.get("extra_note", "")])

    lines.extend(["", "【免责声明】", s.get("disclaimer", "")])
    return "\n".join(lines).strip()


# ============================================================
# 命令行测试
# ============================================================
def _demo():
    volume_result = {
        "patient_id": "P001",
        "case_id": "case_demo_001",
        "ET_cm3": 2.10,
        "TC_cm3": 6.20,
        "WT_cm3": 18.60,
        "ET_WT_ratio": 0.113,
        "TC_WT_ratio": 0.333,
        "WT_brain_ratio": 0.021,
        "interpretation": "整体肿瘤体积处于中等范围。增强肿瘤占比中等。",
        "conclusion": "当前病例整体肿瘤负荷处于中等水平。增强活跃区域存在但不算突出。建议重点关注肿瘤核心（TC）的变化。"
    }

    report = generate_structured_report(
        volume_result=volume_result,
        extra_note="该报告直接基于 analyze_volume_core() 的输出组织生成。"
    )

    print(report["text"])


if __name__ == "__main__":
    _demo()