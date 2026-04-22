"""
rule_kb.py

- 面向 ET / TC / WT 区域 + MRI 模态解释
- 支持问题意图识别（定义 / 影像 / 临床）
- 纯规则 + 本地字典，快速稳定，适合 Agent 调度
"""

import re
from typing import Optional


class BrainTumorKnowledgeBase:
    """脑肿瘤分割专用知识库（Task-aware & Intent-aware）"""

    def __init__(self):

        # =============================
        # 1. 肿瘤区域知识（ET / TC / WT）
        # =============================
        self.region_knowledge = {
            "ET": {
                "name": "Enhancing Tumor（增强肿瘤）",
                "aliases": ["et", "增强肿瘤", "强化肿瘤", "标签4", "4类", "4标签"],
                "definition": (
                    "ET 区域指在注射造影剂后的 T1 加权增强像（T1CE）上出现明显强化的肿瘤实性区域，"
                    "通常反映血脑屏障破坏和肿瘤的活跃生长部分。"
                ),
                "mri": (
                    "在 T1CE 上表现为边界清晰的高信号强化灶；"
                    "在 FLAIR / T2 上多呈高信号，但范围不一定与强化区完全一致。"
                ),
                "clinical": (
                    "ET 通常代表肿瘤中恶性程度高、生物活性强的区域，"
                    "是手术切除、放疗靶区设计和疗效评估的关键参考。"
                ),
                "relation": (
                    "ET 是肿瘤核心（TC）的一部分，而 TC 又包含在整体肿瘤（WT）之内。"
                ),
            },
            "TC": {
                "name": "Tumor Core（肿瘤核心）",
                "aliases": ["tc", "肿瘤核心", "核心区", "标签1", "1类", "1标签"],
                "definition": (
                    "TC 区域通常包括增强肿瘤、非增强实性肿瘤以及坏死组织，"
                    "反映肿瘤内部的主要病理成分。"
                ),
                "mri": (
                    "在 T1CE 上可见强化区，在 T1 上坏死区域多为低信号；"
                    "在 FLAIR / T2 上整体呈高信号或混合信号。"
                ),
                "clinical": (
                    "TC 与肿瘤侵袭性和恶性程度密切相关，"
                    "在手术切除范围评估和预后判断中具有重要作用。"
                ),
                "relation": (
                    "TC 通常包含 ET，是 WT 内部的核心病灶区域。"
                ),
            },
            "WT": {
                "name": "Whole Tumor（整体肿瘤）",
                "aliases": ["wt", "整体肿瘤", "全肿瘤", "标签2", "2类", "2标签"],
                "definition": (
                    "WT 区域涵盖增强肿瘤、肿瘤核心以及周围水肿和浸润区域，"
                    "代表肿瘤对脑组织影响的整体范围。"
                ),
                "mri": (
                    "在 FLAIR 和 T2 序列上通常显示为范围广泛的高信号区，"
                    "明显大于增强肿瘤范围。"
                ),
                "clinical": (
                    "WT 可用于评估肿瘤总体负荷、脑组织受累情况及潜在的功能区影响，"
                    "在术前评估和随访中具有重要临床价值。"
                ),
                "relation": (
                    "WT 是包含 TC 和 ET 的最外层肿瘤相关异常区域。"
                ),
            },
        }

        # =============================
        # 2. MRI 模态知识
        # =============================
        self.modality_knowledge = {
            "T1": {
                "aliases": ["t1"],
                "role": (
                    "T1 加权像主要用于显示脑部解剖结构，"
                    "对水肿和肿瘤浸润敏感性相对较低。"
                ),
                "use": "常用于与 T1CE 联合分析肿瘤是否强化。",
            },
            "T1CE": {
                "aliases": ["t1ce", "增强t1", "对比增强"],
                "role": (
                    "T1CE 能清晰显示造影剂增强区域，"
                    "是识别增强肿瘤（ET）的关键模态。"
                ),
                "use": "用于确定肿瘤活跃区域和血脑屏障破坏情况。",
            },
            "T2": {
                "aliases": ["t2"],
                "role": (
                    "T2 序列对水分变化高度敏感，"
                    "水肿和囊性成分通常呈高信号。"
                ),
                "use": "辅助评估水肿和病灶形态。",
            },
            "FLAIR": {
                "aliases": ["flair"],
                "role": (
                    "FLAIR 抑制脑脊液信号，"
                    "突出显示脑实质内水肿与浸润区域。"
                ),
                "use": "常作为整体肿瘤（WT）观察和可视化的主要背景模态。",
            },
        }

        self._digit_label_pattern = re.compile(r"标签?\s*([0-9])")

    # =============================
    # 内部辅助函数
    # =============================
    @staticmethod
    def _normalize(text: str) -> str:
        return text.lower().strip()

    def _detect_intent(self, q: str) -> str:
        """
        简单判断问题意图：definition / mri / clinical / relation / general
        """
        if any(k in q for k in ["定义", "是什么", "含义"]):
            return "definition"
        if any(k in q for k in ["mri", "影像", "序列", "表现"]):
            return "mri"
        if any(k in q for k in ["临床", "意义", "作用", "价值"]):
            return "clinical"
        if any(k in q for k in ["关系", "区别", "比较"]):
            return "relation"
        return "general"

    def _match_region(self, q: str) -> Optional[str]:
        m = self._digit_label_pattern.search(q)
        if m:
            return {"4": "ET", "1": "TC", "2": "WT"}.get(m.group(1))
        for k, info in self.region_knowledge.items():
            if any(alias in q for alias in info["aliases"]):
                return k
        return None

    def _match_modality(self, q: str) -> Optional[str]:
        for k, info in self.modality_knowledge.items():
            if any(alias in q for alias in info["aliases"]):
                return k
        return None

    # =============================
    # 对外主接口
    # =============================
    def query(self, question: str) -> str:
        if not question:
            return (
                "我可以解释脑肿瘤分割中的 ET / TC / WT 区域含义，"
                "以及 T1 / T1CE / T2 / FLAIR 等 MRI 模态的作用。"
            )

        q = self._normalize(question)
        intent = self._detect_intent(q)

        region = self._match_region(q)
        if region:
            info = self.region_knowledge[region]
            if intent == "definition":
                return info["definition"]
            if intent == "mri":
                return info["mri"]
            if intent == "clinical":
                return info["clinical"]
            if intent == "relation":
                return info["relation"]

            return (
                f"{info['name']}：\n\n"
                f"**定义**：{info['definition']}\n\n"
                f"**MRI 表现**：{info['mri']}\n\n"
                f"**临床意义**：{info['clinical']}"
            )

        modality = self._match_modality(q)
        if modality:
            info = self.modality_knowledge[modality]
            return (
                f"{modality} 模态：\n\n"
                f"**作用**：{info['role']}\n\n"
                f"**典型用途**：{info['use']}"
            )

        return (
            "当前知识库支持解释：\n"
            "- 脑肿瘤分割区域（ET / TC / WT）；\n"
            "- MRI 模态（T1 / T1CE / T2 / FLAIR）。\n\n"
            "你可以尝试问：\n"
            "- “ET 区域在 MRI 上有什么表现？”\n"
            "- “WT 有什么临床意义？”\n"
            "- “FLAIR 在脑肿瘤分割里主要看什么？”"
        )


# 单例接口
_kb = None


def get_knowledge_base() -> BrainTumorKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = BrainTumorKnowledgeBase()
    return _kb