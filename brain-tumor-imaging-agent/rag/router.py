# router.py
# 作用：
# 1. 作为 Agent 统一入口
# 2. 先判断问题是否适合走 rule_kb
# 3. 规则库命中则直接回答
# 4. 否则转交 rag_kb.py

import json
import argparse
from typing import Dict, Any

from rag.rule_kb import get_knowledge_base
from rag.rag_kb import answer_with_rag
from rag.retriever import str2bool, normalize_text


# =============================
# 路由策略关键词
# 这些词一旦出现，优先走 RAG
# 因为 rule_kb 主要擅长 ET / TC / WT / MRI 模态解释，
# 不擅长治疗、药物、预后等开放问题。
# =============================
RAG_PRIORITY_KEYWORDS = [
    "治疗", "方案", "药物", "手术", "放疗", "化疗", "免疫", "靶向",
    "预后", "生存", "复发", "分级", "分型", "症状", "病因", "诊断",
    "检查", "指南", "适应证", "禁忌", "副作用", "管理", "策略",
    "脑肿瘤", "胶质母细胞瘤", "脑转移瘤", "替莫唑胺", "贝伐珠单抗"
]


def should_use_rule_kb(question: str) -> bool:
    """
    判断当前问题是否优先走规则库。
    规则：
    1. 空问题 -> 规则库兜底提示
    2. 若包含明显开放医学问答关键词 -> 走 RAG
    3. 若命中 ET/TC/WT 或 MRI 模态 -> 走 rule_kb
    4. 否则 -> 走 RAG
    """
    kb = get_knowledge_base()

    if not question:
        return True

    q = kb._normalize(question)

    # 明显属于开放知识问答，优先走 RAG
    if any(k in q for k in RAG_PRIORITY_KEYWORDS):
        return False

    # 命中区域或模态，走规则库
    region = kb._match_region(q)
    modality = kb._match_modality(q)

    if region is not None or modality is not None:
        return True

    return False


def answer_with_router(
    question: str,
    index_dir: str = "/root/code/agent1/rag/index",
    embedding_model: str = "/root/models/bge-m3",
    reranker_model: str = "/root/models/bge-reranker-v2-m3",
    device: str = "cuda",
    local_files_only: bool = True,
    dense_topk: int = 20,
    sparse_topk: int = 20,
    candidate_topk: int = 20,
    final_topk: int = 5,
    prefer_jieba: bool = True,
) -> Dict[str, Any]:
    """
    router 总入口：
    - rule_kb 命中：直接返回规则答案
    - 否则：调用 answer_with_rag()
    """
    question = normalize_text(question)
    kb = get_knowledge_base()

    if should_use_rule_kb(question):
        answer = kb.query(question)
        return {
            "route": "rule_kb",
            "question": question,
            "answer": answer,
            "references": [],
            "context": "",
            "num_candidates": 0,
            "num_final_results": 0,
        }

    # 走 RAG
    rag_result = answer_with_rag(
        question=question,
        index_dir=index_dir,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        device=device,
        local_files_only=local_files_only,
        dense_topk=dense_topk,
        sparse_topk=sparse_topk,
        candidate_topk=candidate_topk,
        final_topk=final_topk,
        prefer_jieba=prefer_jieba,
    )
    rag_result["route"] = "rag_kb"
    return rag_result


# =============================
# Agent 可直接注册的工具函数
# =============================
def brain_tumor_qa_tool(question: str) -> str:
    result = answer_with_router(question)
    return result["answer"]


# =============================
# 命令行测试入口
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="用户问题"
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="/root/code/agent1/rag/index",
        help="索引目录"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="/root/models/bge-m3",
        help="embedding 模型路径"
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="/root/models/bge-reranker-v2-m3",
        help="reranker 模型路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备"
    )
    parser.add_argument(
        "--local_files_only",
        type=str2bool,
        default=True,
        help="是否只从本地加载模型"
    )
    parser.add_argument(
        "--dense_topk",
        type=int,
        default=20,
        help="FAISS 召回 topk"
    )
    parser.add_argument(
        "--sparse_topk",
        type=int,
        default=20,
        help="BM25 召回 topk"
    )
    parser.add_argument(
        "--candidate_topk",
        type=int,
        default=20,
        help="送入 reranker 的候选数"
    )
    parser.add_argument(
        "--final_topk",
        type=int,
        default=5,
        help="最终保留结果数"
    )
    parser.add_argument(
        "--prefer_jieba",
        type=str2bool,
        default=True,
        help="是否优先使用 jieba"
    )
    parser.add_argument(
        "--save_json",
        type=str2bool,
        default=False,
        help="是否保存结果到 router_answer.json"
    )
    args = parser.parse_args()

    result = answer_with_router(
        question=args.question,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        device=args.device,
        local_files_only=args.local_files_only,
        dense_topk=args.dense_topk,
        sparse_topk=args.sparse_topk,
        candidate_topk=args.candidate_topk,
        final_topk=args.final_topk,
        prefer_jieba=args.prefer_jieba,
    )

    print(f"\n[ROUTE] {result['route']}")
    print("\n[ANSWER]")
    print(result["answer"])

    if result.get("references"):
        print("\n[REFERENCES]")
        for i, ref in enumerate(result["references"], start=1):
            print(f"{i}. {ref['title']} | topic={ref['topic']} | rerank_score={ref['rerank_score']}")
            print(f"   preview={ref['preview']}")

    if args.save_json:
        save_path = "/root/code/agent1/rag/index/router_answer.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存到: {save_path}")


if __name__ == "__main__":
    main()