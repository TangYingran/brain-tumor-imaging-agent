# long_term_memory.py
# 1. volume_analysis
# 2. structured_report

import os
import re
import json
import uuid
import time
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer


# 允许两类长期记忆
ALLOWED_MEMORY_TYPES = {
    "volume_analysis",
    "structured_report",
}


# ============================================================
# 基础工具函数
# ============================================================
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def resolve_device(device: str) -> str:
    if device == "cpu":
        print("[INFO] 使用 CPU")
        return "cpu"

    if device != "cuda":
        print(f"[WARN] 未知 device={device}，自动切换到 CPU")
        return "cpu"

    if not torch.cuda.is_available():
        print("[WARN] 未检测到可用 CUDA，自动切换到 CPU")
        return "cpu"

    try:
        _ = torch.zeros(1).to("cuda")
        print("[INFO] CUDA 检查通过，使用 GPU")
        return "cuda"
    except Exception as e:
        print(f"[WARN] CUDA 初始化失败：{e}")
        print("[WARN] 自动切换到 CPU")
        return "cpu"


# ============================================================
# 长期记忆主类
# ============================================================
class LongTermMemoryStore:
    def __init__(
        self,
        model_path: str = "/root/models/bge-m3",
        save_dir: str = "/root/code/agent1/memory",
        device: str = "cuda",
        local_files_only: bool = True
    ):
        self.model_path = model_path
        self.save_dir = save_dir
        self.device = resolve_device(device)
        self.local_files_only = local_files_only

        os.makedirs(self.save_dir, exist_ok=True)

        self.index_path = os.path.join(self.save_dir, "patient_memory.index")
        self.meta_path = os.path.join(self.save_dir, "patient_memory.json")

        print(f"[INFO] 加载长期记忆 embedding 模型: {self.model_path}")
        self.model = SentenceTransformer(
            self.model_path,
            device=self.device,
            trust_remote_code=True,
            local_files_only=self.local_files_only
        )

        self.records: List[Dict[str, Any]] = []
        self.index: Optional[faiss.Index] = None

        self._load()

    # --------------------------------------------------------
    # 内部读写
    # --------------------------------------------------------
    def _load(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
            if not isinstance(self.records, list):
                raise ValueError("patient_memory.json 格式错误，应为 list")
        else:
            self.records = []

        self.records = [
            r for r in self.records
            if r.get("memory_type") in ALLOWED_MEMORY_TYPES
        ]

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None

        # 若索引数与记录数不一致，则重建
        if self.index is not None and self.index.ntotal != len(self.records):
            print(
                f"[WARN] 长期记忆索引条目数({self.index.ntotal})与记录数({len(self.records)})不一致，开始自动重建索引..."
            )
            self.rebuild_index()

    def _save(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts 为空，无法生成向量")

        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vecs = np.asarray(vecs, dtype="float32")

        if vecs.ndim == 1:
            vecs = vecs[None, :]

        return vecs

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

    # --------------------------------------------------------
    # 对外主接口
    # --------------------------------------------------------
    def add_memory(
        self,
        patient_id: str,
        case_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: str = "volume_analysis"
    ) -> str:
        """
        添加一条长期记忆（允许 volume_analysis / structured_report）
        """
        patient_id = normalize_text(patient_id)
        case_id = normalize_text(case_id)
        text = normalize_text(text)

        if not patient_id:
            raise ValueError("patient_id 不能为空")
        if not case_id:
            raise ValueError("case_id 不能为空")
        if not text:
            raise ValueError("text 不能为空")
        if memory_type not in ALLOWED_MEMORY_TYPES:
            raise ValueError(
                f"当前精简版长期记忆只允许: {sorted(ALLOWED_MEMORY_TYPES)}，收到: {memory_type}"
            )

        metadata = metadata or {}

        vec = self._encode_texts([text])
        self._ensure_index(vec.shape[1])
        self.index.add(vec)

        memory_id = str(uuid.uuid4())
        record = {
            "memory_id": memory_id,
            "patient_id": patient_id,
            "case_id": case_id,
            "memory_type": memory_type,
            "text": text,
            "metadata": metadata,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }

        self.records.append(record)
        self._save()
        return memory_id

    def search(
        self,
        patient_id: str,
        query: str,
        topk: int = 3
    ) -> List[Dict[str, Any]]:
        """
        仅在指定 patient_id 下进行历史语义召回
        """
        patient_id = normalize_text(patient_id)
        query = normalize_text(query)

        if not patient_id or not query:
            return []

        if self.index is None or not self.records:
            return []

        q_vec = self._encode_texts([query])

        search_k = min(max(topk * 10, 50), len(self.records))
        scores, idxs = self.index.search(q_vec, search_k)

        results = []
        used_memory_ids = set()

        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.records):
                continue

            rec = self.records[idx]
            if rec["patient_id"] != patient_id:
                continue

            if rec["memory_id"] in used_memory_ids:
                continue
            used_memory_ids.add(rec["memory_id"])

            item = dict(rec)
            item["score"] = float(score)
            results.append(item)

            if len(results) >= topk:
                break

        return results

    def get_patient_memories(self, patient_id: str) -> List[Dict[str, Any]]:
        patient_id = normalize_text(patient_id)
        results = [r for r in self.records if r["patient_id"] == patient_id]
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results

    def rebuild_index(self):
        """
        根据当前 records 重建 FAISS 索引
        """
        if not self.records:
            self.index = None
            self._save()
            return

        texts = [normalize_text(r["text"]) for r in self.records]
        vecs = self._encode_texts(texts)

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self.index = index
        self._save()

    def delete_patient_memories(self, patient_id: str):
        """
        删除某个患者的全部长期记忆，并自动重建索引
        """
        patient_id = normalize_text(patient_id)
        self.records = [r for r in self.records if r["patient_id"] != patient_id]
        self.rebuild_index()

    # --------------------------------------------------------
    # 便捷接口：1) 体积分析
    # --------------------------------------------------------
    def add_volume_analysis_memory(
        self,
        patient_id: str,
        case_id: str,
        WT_cm3: float,
        TC_cm3: float,
        ET_cm3: float,
        ET_WT_ratio: Optional[float] = None,
        TC_WT_ratio: Optional[float] = None,
        WT_brain_ratio: Optional[float] = None,
        interpretation: str = ""
    ) -> str:
        text = (
            f"病例 {case_id} 体积分析结果："
            f"WT={WT_cm3:.2f} cm3，"
            f"TC={TC_cm3:.2f} cm3，"
            f"ET={ET_cm3:.2f} cm3。"
        )

        if ET_WT_ratio is not None:
            text += f" ET/WT={ET_WT_ratio:.2%}。"
        if TC_WT_ratio is not None:
            text += f" TC/WT={TC_WT_ratio:.2%}。"
        if WT_brain_ratio is not None:
            text += f" WT/全脑={WT_brain_ratio:.2%}。"
        if interpretation:
            text += f" 解读：{normalize_text(interpretation)}"

        metadata = {
            "type": "volume_analysis",
            "WT_cm3": WT_cm3,
            "TC_cm3": TC_cm3,
            "ET_cm3": ET_cm3,
            "ET_WT_ratio": ET_WT_ratio,
            "TC_WT_ratio": TC_WT_ratio,
            "WT_brain_ratio": WT_brain_ratio,
            "interpretation": interpretation,
            "keywords": ["体积分析", "WT", "TC", "ET"],
        }

        return self.add_memory(
            patient_id=patient_id,
            case_id=case_id,
            text=text,
            metadata=metadata,
            memory_type="volume_analysis"
        )

    # --------------------------------------------------------
    # 便捷接口：2) 结构化报告
    # --------------------------------------------------------
    def add_report_memory(
        self,
        patient_id: str,
        case_id: str,
        report_text: str,
        report_type: str = "structured_report"
    ) -> str:
        if report_type != "structured_report":
            raise ValueError("精简版只允许 structured_report 作为报告类型")

        text = f"病例 {case_id} 报告摘要：{normalize_text(report_text)}"
        metadata = {
            "type": report_type,
            "keywords": ["报告", "结构化报告"]
        }

        return self.add_memory(
            patient_id=patient_id,
            case_id=case_id,
            text=text,
            metadata=metadata,
            memory_type="structured_report"
        )

    # --------------------------------------------------------
    # 患者画像（基于 volume_analysis + structured_report）
    # --------------------------------------------------------
    def build_patient_profile(self, patient_id: str) -> Dict[str, Any]:
        memories = self.get_patient_memories(patient_id)

        profile = {
            "patient_id": patient_id,
            "num_records": len(memories),
            "num_cases": 0,
            "latest_case_id": "",
            "latest_summary": "",
            "memory_type_count": {},
            "volume_trend": [],
            "latest_report_summary": "",
        }

        if not memories:
            return profile

        case_ids = []
        type_count = {}

        for rec in memories:
            case_ids.append(rec.get("case_id", ""))
            mtype = rec.get("memory_type", "generic")
            type_count[mtype] = type_count.get(mtype, 0) + 1

            meta = rec.get("metadata", {})

            if mtype == "volume_analysis":
                profile["volume_trend"].append({
                    "case_id": rec.get("case_id", ""),
                    "WT_cm3": meta.get("WT_cm3"),
                    "TC_cm3": meta.get("TC_cm3"),
                    "ET_cm3": meta.get("ET_cm3"),
                    "created_at": rec.get("created_at", ""),
                })

            if mtype == "structured_report" and not profile["latest_report_summary"]:
                profile["latest_report_summary"] = rec.get("text", "")[:300]

        profile["num_cases"] = len(set(case_ids))
        profile["latest_case_id"] = memories[0].get("case_id", "")
        profile["latest_summary"] = memories[0].get("text", "")[:300]
        profile["memory_type_count"] = type_count

        return profile


# ============================================================
# 单例接口
# ============================================================
_memory_store = None


def get_long_term_memory() -> LongTermMemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = LongTermMemoryStore()
    return _memory_store


# ============================================================
# 命令行测试
# ============================================================
def _demo():
    store = get_long_term_memory()

    patient_id = "P001"
    case_id = "case_demo_001"

    store.add_volume_analysis_memory(
        patient_id=patient_id,
        case_id=case_id,
        WT_cm3=18.6,
        TC_cm3=6.2,
        ET_cm3=2.1,
        ET_WT_ratio=0.113,
        TC_WT_ratio=0.333,
        WT_brain_ratio=0.021,
        interpretation="整体肿瘤负荷中等，增强区域相对局限。"
    )

    store.add_report_memory(
        patient_id=patient_id,
        case_id=case_id,
        report_text="本次病例整体肿瘤负荷中等，增强区域占比中等。",
        report_type="structured_report"
    )

    results = store.search(patient_id="P001", query="这个患者上次 WT 体积是多少？", topk=3)
    print("\n[SEARCH RESULTS]")
    for r in results:
        print(f"- score={r['score']:.4f} | type={r['memory_type']} | text={r['text']}")

    profile = store.build_patient_profile("P001")
    print("\n[PATIENT PROFILE]")
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _demo()