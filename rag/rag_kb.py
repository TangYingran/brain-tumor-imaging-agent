# rag_kb.py
# 作用：
# 1. 调用 retriever.py 做 hybrid retrieval
# 2. 调用 reranker.py 做精排
# 3. 拼接上下文
# 4. 调用 LLM 生成最终答案
# 5. 对外暴露 answer_with_rag()，方便 Agent 直接调用

import os
import json
import argparse
from typing import List, Dict, Any, Optional

from rag.retriever import (
    load_retrieval_resources,
    hybrid_retrieve_from_resources,
    normalize_text,
    str2bool,
)
from rag.reranker import (
    load_reranker_model,
    rerank_candidates,
    deduplicate_by_doc,
)

# =============================
# 全局缓存：避免每次都重复加载索引 / reranker
# =============================
_RETRIEVAL_RESOURCE_CACHE: Dict[str, Dict[str, Any]] = {}
_RERANKER_MODEL_CACHE: Dict[str, Any] = {}


# =============================
# 缓存加载函数
# =============================
def get_cached_retrieval_resources(index_dir: str, prefer_jieba: bool = True) -> Dict[str, Any]:
    cache_key = f"{index_dir}__jieba={prefer_jieba}"
    if cache_key not in _RETRIEVAL_RESOURCE_CACHE:
        _RETRIEVAL_RESOURCE_CACHE[cache_key] = load_retrieval_resources(
            index_dir=index_dir,
            prefer_jieba=prefer_jieba
        )
    return _RETRIEVAL_RESOURCE_CACHE[cache_key]


def get_cached_reranker(model_name: str, device: str = "cuda", local_files_only: bool = True):
    cache_key = f"{model_name}__device={device}__local={local_files_only}"
    if cache_key not in _RERANKER_MODEL_CACHE:
        _RERANKER_MODEL_CACHE[cache_key] = load_reranker_model(
            model_name=model_name,
            device=device,
            local_files_only=local_files_only
        )
    return _RERANKER_MODEL_CACHE[cache_key]


# =============================
# 上下文构造
# =============================
def build_context(
    results: List[Dict[str, Any]],
    topk: int = 3,
    max_chars_per_chunk: int = 600
) -> str:
    contexts = []
    for i, item in enumerate(results[:topk], start=1):
        chunk = item["chunk"]
        title = normalize_text(chunk.get("title", ""))
        topic = normalize_text(chunk.get("topic", ""))
        text = normalize_text(chunk.get("text", ""))[:max_chars_per_chunk]
        rerank_score = item.get("rerank_score", None)

        block = [
            f"[参考{i}]",
            f"标题：{title}" if title else "标题：",
            f"主题：{topic}" if topic else "主题：",
            f"相关性分数：{rerank_score}" if rerank_score is not None else "相关性分数：",
            f"内容：{text}",
        ]
        contexts.append("\n".join(block))

    return "\n\n".join(contexts)


def build_references(results: List[Dict[str, Any]], topk: int = 5) -> List[Dict[str, Any]]:
    refs = []
    for item in results[:topk]:
        chunk = item["chunk"]
        refs.append(
            {
                "doc_id": chunk.get("doc_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "title": chunk.get("title", ""),
                "topic": chunk.get("topic", ""),
                "chunk_index": chunk.get("chunk_index", ""),
                "rerank_score": item.get("rerank_score", None),
                "rrf_score": item.get("rrf_score", None),
                "preview": normalize_text(chunk.get("text", ""))[:220]
            }
        )
    return refs


# =============================
# Prompt 构造
# =============================
def build_system_prompt() -> str:
    return (
        "你是一个面向脑肿瘤相关知识问答的专业助手。\n"
        "请严格基于给定参考内容回答问题，不要凭空补充未出现的结论。\n"
        "若参考内容不足以支持完整回答，请明确说明“当前知识库信息不足”。\n"
        "回答要求：\n"
        "1. 用中文回答；\n"
        "2. 先直接回答问题，再补充必要解释；\n"
        "3. 不要编造文献、指南或数据；\n"
        "4. 若问题涉及治疗建议，强调应结合医生评估，不替代临床决策。"
    )


def build_user_prompt(question: str, context: str) -> str:
    return (
        f"用户问题：\n{question}\n\n"
        f"可用参考内容：\n{context}\n\n"
        "请基于以上参考内容作答。"
    )


# =============================
# LLM 调用
# 支持 OpenAI 兼容接口
# 环境变量优先级：
# RAG_LLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY
# RAG_LLM_BASE_URL / OPENAI_BASE_URL / DEEPSEEK_BASE_URL
# RAG_LLM_MODEL / OPENAI_MODEL / DEEPSEEK_MODEL
# =============================
def get_llm_config() -> Dict[str, Optional[str]]:
    api_key = (
        os.getenv("RAG_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    base_url = (
        os.getenv("RAG_LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
    )
    model = (
        os.getenv("RAG_LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
    )
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def call_openai_compatible_llm(
    question: str,
    context: str,
    temperature: float = 0.1
) -> Optional[str]:
    cfg = get_llm_config()
    api_key = cfg["api_key"]
    model = cfg["model"]
    base_url = cfg["base_url"]

    if not api_key or not model:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    try:
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(question, context)},
        ]

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        answer = resp.choices[0].message.content
        if answer is None:
            return None
        return answer.strip()
    except Exception:
        return None


# =============================
# 无 LLM 时的兜底输出
# =============================
def build_fallback_answer(question: str, reranked_results: List[Dict[str, Any]], topk: int = 3) -> str:
    if not reranked_results:
        return "当前知识库未检索到与该问题足够相关的内容。"

    lines = [
        "当前未配置可用的 LLM，因此暂不生成自然语言总结答案。",
        "下面是检索到的最相关知识片段，你可以先据此查看：",
        ""
    ]

    for i, item in enumerate(reranked_results[:topk], start=1):
        chunk = item["chunk"]
        title = chunk.get("title", "")
        preview = normalize_text(chunk.get("text", ""))[:220]
        lines.append(f"{i}. {title}")
        lines.append(f"   {preview}")
        lines.append("")

    return "\n".join(lines).strip()


# =============================
# 对外主函数（给 Agent 用）
# =============================
def answer_with_rag(
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
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60,
    prefer_jieba: bool = True,
    reranker_batch_size: int = 8,
    reranker_max_length: int = 1024,
    dedup_by_doc: bool = True,
    max_chunks_per_doc: int = 1,
    context_topk: int = 3,
    llm_temperature: float = 0.1,
) -> Dict[str, Any]:
    question = normalize_text(question)
    if not question:
        return {
            "route": "rag_kb",
            "answer": "问题不能为空。",
            "references": [],
            "context": "",
        }

    # 1. 加载索引资源（缓存）
    resources = get_cached_retrieval_resources(
        index_dir=index_dir,
        prefer_jieba=prefer_jieba
    )

    # 2. hybrid retrieval
    candidates = hybrid_retrieve_from_resources(
        query=question,
        resources=resources,
        model_name=embedding_model,
        device=device,
        local_files_only=local_files_only,
        dense_topk=dense_topk,
        sparse_topk=sparse_topk,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k
    )

    candidates = candidates[:candidate_topk]

    # 3. 加载 reranker（缓存）
    reranker_tokenizer, reranker_model_obj, reranker_device = get_cached_reranker(
        model_name=reranker_model,
        device=device,
        local_files_only=local_files_only
    )

    # 4. rerank
    reranked_results = rerank_candidates(
        query=question,
        candidates=candidates,
        reranker_tokenizer=reranker_tokenizer,
        reranker_model=reranker_model_obj,
        device=reranker_device,
        batch_size=reranker_batch_size,
        max_length=reranker_max_length
    )

    # 5. 去重
    if dedup_by_doc:
        reranked_results = deduplicate_by_doc(
            reranked_results,
            max_chunks_per_doc=max_chunks_per_doc
        )

    top_results = reranked_results[:final_topk]

    # 6. 拼 context
    context = build_context(
        top_results,
        topk=context_topk
    )

    # 7. 调 LLM
    answer = call_openai_compatible_llm(
        question=question,
        context=context,
        temperature=llm_temperature
    )

    # 8. 无 LLM 时兜底
    if not answer:
        answer = build_fallback_answer(
            question=question,
            reranked_results=top_results,
            topk=min(3, len(top_results))
        )

    # 9. 返回结构化结果
    return {
        "route": "rag_kb",
        "question": question,
        "answer": answer,
        "references": build_references(top_results, topk=final_topk),
        "context": context,
        "num_candidates": len(candidates),
        "num_final_results": len(top_results),
    }


# =============================
# 给 Agent 直接注册成 tool 时可用的简化接口
# =============================
def rag_kb_tool(question: str) -> str:
    result = answer_with_rag(question)
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
        help="最终保留的结果数"
    )
    parser.add_argument(
        "--context_topk",
        type=int,
        default=3,
        help="拼上下文时使用前几个结果"
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
        help="是否保存结果到 rag_answer.json"
    )
    args = parser.parse_args()

    result = answer_with_rag(
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
        context_topk=args.context_topk,
        prefer_jieba=args.prefer_jieba
    )

    print("\n[ANSWER]")
    print(result["answer"])

    print("\n[REFERENCES]")
    for i, ref in enumerate(result["references"], start=1):
        print(f"{i}. {ref['title']} | topic={ref['topic']} | rerank_score={ref['rerank_score']}")
        print(f"   preview={ref['preview']}")

    if args.save_json:
        save_path = os.path.join(args.index_dir, "rag_answer.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存到: {save_path}")


if __name__ == "__main__":
    main()