# reranker.py
# 作用：
# 1. 调用 retriever.py 中的 hybrid_retrieve() 获取候选 chunk
# 2. 使用 bge-reranker 对候选做精排
# 3. 输出最终 top-k 结果

import os
import json
import argparse
from collections import Counter
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from rag.retriever import hybrid_retrieve, str2bool, normalize_text, resolve_device


def build_doc_text_for_rerank(chunk: Dict[str, Any]) -> str:
    """
    构造送入 reranker 的文档文本。
    """
    title = normalize_text(chunk.get("title", ""))
    topic = normalize_text(chunk.get("topic", ""))
    text = normalize_text(chunk.get("text", ""))

    parts = []
    if title:
        parts.append(f"标题：{title}")
    if topic:
        parts.append(f"主题：{topic}")
    if text:
        parts.append(f"正文：{text}")
    return "\n".join(parts).strip()


def load_reranker_model(
    model_name: str,
    device: str = "cuda",
    local_files_only: bool = True
):
    if ("/" in model_name or model_name.startswith(".")) and not os.path.exists(model_name):
        raise FileNotFoundError(f"reranker 模型路径不存在: {model_name}")

    device = resolve_device(device)

    print(f"[INFO] 加载 reranker 模型: {model_name}")
    print(f"[INFO] reranker device = {device}")
    print(f"[INFO] local_files_only = {local_files_only}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only
    )
    model.eval()
    model.to(device)

    return tokenizer, model, device


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    reranker_tokenizer,
    reranker_model,
    device: str,
    batch_size: int = 8,
    max_length: int = 512
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    docs = [build_doc_text_for_rerank(item["chunk"]) for item in candidates]
    scores_all = []

    with torch.no_grad():
        for start in range(0, len(docs), batch_size):
            batch_docs = docs[start:start + batch_size]
            batch_queries = [query] * len(batch_docs)

            inputs = reranker_tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = reranker_model(**inputs).logits.view(-1)
            batch_scores = logits.detach().cpu().float().tolist()
            scores_all.extend(batch_scores)

    reranked = []
    for item, score in zip(candidates, scores_all):
        new_item = dict(item)
        new_item["rerank_score"] = float(score)
        reranked.append(new_item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


def deduplicate_by_doc(
    results: List[Dict[str, Any]],
    max_chunks_per_doc: int = 1
) -> List[Dict[str, Any]]:
    kept = []
    doc_counter = Counter()

    for item in results:
        chunk = item["chunk"]
        doc_id = chunk.get("doc_id", None)

        if not doc_id:
            title = chunk.get("title", "")
            chunk_index = chunk.get("chunk_index", "")
            doc_id = f"{title}::{chunk_index}"

        if doc_counter[doc_id] >= max_chunks_per_doc:
            continue

        doc_counter[doc_id] += 1
        kept.append(item)

    return kept


def pretty_print_results(results: List[Dict[str, Any]], topk: int = 5, preview_chars: int = 180):
    print("\n[RESULT] reranker 精排结果：")
    for rank, item in enumerate(results[:topk], start=1):
        chunk = item["chunk"]
        title = chunk.get("title", "")
        topic = chunk.get("topic", "")
        chunk_index = chunk.get("chunk_index", "")
        preview = normalize_text(chunk.get("text", ""))[:preview_chars]

        print(f"{rank:02d}. idx={item['idx']} | rerank_score={item.get('rerank_score'):.6f}")
        print(f"    title={title} | topic={topic} | chunk_index={chunk_index}")
        print(f"    rrf_score={item.get('rrf_score')}")
        print(f"    dense_rank={item.get('dense_rank')} dense_score={item.get('dense_score')}")
        print(f"    sparse_rank={item.get('sparse_rank')} sparse_score={item.get('sparse_score')}")
        print(f"    preview={preview}")


def main():
    parser = argparse.ArgumentParser()
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
        help="embedding 模型名或本地路径"
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="/root/models/bge-reranker-v2-m3",
        help="reranker 模型名或本地路径"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="用户查询"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备。若指定 cuda 但不可用，会自动切到 cpu"
    )
    parser.add_argument(
        "--local_files_only",
        type=str2bool,
        default=True,
        help="是否只从本地加载模型，默认 true"
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
        help="最终输出 topk"
    )
    parser.add_argument(
        "--dense_weight",
        type=float,
        default=1.0,
        help="dense 路线的 RRF 权重"
    )
    parser.add_argument(
        "--sparse_weight",
        type=float,
        default=1.0,
        help="sparse 路线的 RRF 权重"
    )
    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
        help="RRF 的 k 参数"
    )
    parser.add_argument(
        "--prefer_jieba",
        type=str2bool,
        default=True,
        help="未检测到 bm25 tokenizer 配置时，是否优先尝试 jieba"
    )
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        default=8,
        help="reranker 批大小"
    )
    parser.add_argument(
        "--reranker_max_length",
        type=int,
        default=1024,
        help="reranker 截断长度"
    )
    parser.add_argument(
        "--dedup_by_doc",
        type=str2bool,
        default=True,
        help="是否按 doc_id 去重"
    )
    parser.add_argument(
        "--max_chunks_per_doc",
        type=int,
        default=1,
        help="每个 doc 最多保留几个 chunk"
    )
    parser.add_argument(
        "--save_json",
        type=str2bool,
        default=False,
        help="是否将结果保存为 rerank_results.json"
    )
    args = parser.parse_args()

    print("[INFO] 调用 hybrid_retrieve() 获取候选 ...")
    candidates = hybrid_retrieve(
        query=args.query,
        index_dir=args.index_dir,
        model_name=args.embedding_model,
        device=args.device,
        local_files_only=args.local_files_only,
        dense_topk=args.dense_topk,
        sparse_topk=args.sparse_topk,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        rrf_k=args.rrf_k,
        prefer_jieba=args.prefer_jieba
    )

    candidates = candidates[:args.candidate_topk]
    print(f"[INFO] 送入 reranker 的候选数: {len(candidates)}")

    reranker_tokenizer, reranker_model, reranker_device = load_reranker_model(
        model_name=args.reranker_model,
        device=args.device,
        local_files_only=args.local_files_only
    )

    print("[INFO] 开始 reranker 精排 ...")
    reranked_results = rerank_candidates(
        query=args.query,
        candidates=candidates,
        reranker_tokenizer=reranker_tokenizer,
        reranker_model=reranker_model,
        device=reranker_device,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length
    )

    if args.dedup_by_doc:
        reranked_results = deduplicate_by_doc(
            reranked_results,
            max_chunks_per_doc=args.max_chunks_per_doc
        )

    pretty_print_results(reranked_results, topk=args.final_topk)

    if args.save_json:
        save_path = os.path.join(args.index_dir, "rerank_results.json")
        serializable_results = []
        for item in reranked_results[:args.final_topk]:
            serializable_results.append(
                {
                    "idx": item["idx"],
                    "rerank_score": item["rerank_score"],
                    "rrf_score": item.get("rrf_score"),
                    "dense_rank": item.get("dense_rank"),
                    "dense_score": item.get("dense_score"),
                    "sparse_rank": item.get("sparse_rank"),
                    "sparse_score": item.get("sparse_score"),
                    "chunk": item["chunk"]
                }
            )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 精排结果已保存到: {save_path}")

    print("[DONE] reranker 完成。")


if __name__ == "__main__":
    main()