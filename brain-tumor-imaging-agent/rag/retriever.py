# retriever.py
# 作用：
# 1. 加载 FAISS 向量索引、chunks 元数据、BM25 索引
# 2. 对 query 同时做 dense 检索 + sparse 检索
# 3. 用 RRF 融合两路回结果
# 4. 暴露 hybrid_retrieve() 供 reranker.py / rag_kb.py 调用

import os
import re
import json
import pickle
import argparse
from collections import Counter
from typing import List, Dict, Any

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("布尔参数只支持 true/false")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenize(text: str) -> List[str]:
    """
    无 jieba 时的回退分词：
    - 中文按单字切
    - 英文/数字按连续串切
    """
    text = normalize_text(text).lower()
    if not text:
        return []
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", text)
    return [t for t in tokens if t.strip()]


def jieba_tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    if not text:
        return []

    import jieba
    tokens = jieba.lcut(text)

    cleaned = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        sub_tokens = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", t)
        cleaned.extend(sub_tokens)

    return cleaned


def get_tokenizer(index_dir: str, prefer_jieba: bool = True):
    """
    优先读取 bm25_config.json 中的 tokenizer 设置。
    如果没有配置，则根据 prefer_jieba 决定。
    """
    bm25_config_path = os.path.join(index_dir, "bm25_config.json")
    tokenizer_name = None

    if os.path.exists(bm25_config_path):
        try:
            with open(bm25_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            tokenizer_name = cfg.get("tokenizer", None)
        except Exception:
            tokenizer_name = None

    if tokenizer_name == "jieba":
        try:
            import jieba  # noqa: F401
            print("[INFO] BM25 tokenizer = jieba")
            return jieba_tokenize, "jieba"
        except Exception:
            print("[WARN] bm25_config 指定 jieba，但当前环境不可用，退回 simple_tokenize")
            return simple_tokenize, "simple"

    if tokenizer_name == "simple":
        print("[INFO] BM25 tokenizer = simple")
        return simple_tokenize, "simple"

    if prefer_jieba:
        try:
            import jieba  # noqa: F401
            print("[INFO] 未读到 BM25 tokenizer 配置，自动使用 jieba")
            return jieba_tokenize, "jieba"
        except Exception:
            print("[WARN] 未安装 jieba，自动退回 simple_tokenize")
            return simple_tokenize, "simple"

    print("[INFO] 使用 simple_tokenize")
    return simple_tokenize, "simple"


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


def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"chunks 文件不存在: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{chunks_path} 内容不是有效的非空 list")

    return chunks


def load_bm25_index(bm25_path: str) -> Dict[str, Any]:
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(f"BM25 索引不存在: {bm25_path}")

    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)

    required_keys = ["num_docs", "avgdl", "doc_len", "idf", "term_freqs", "chunk_ids", "k1", "b"]
    for k in required_keys:
        if k not in bm25_index:
            raise ValueError(f"BM25 索引缺少关键字段: {k}")

    return bm25_index


def load_faiss_index(faiss_path: str) -> faiss.Index:
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS 索引不存在: {faiss_path}")
    return faiss.read_index(faiss_path)


def encode_query(
    query: str,
    model_name: str,
    device: str = "cuda",
    local_files_only: bool = True
) -> np.ndarray:
    query = normalize_text(query)
    if not query:
        raise ValueError("query 不能为空")

    if ("/" in model_name or model_name.startswith(".")) and not os.path.exists(model_name):
        raise FileNotFoundError(f"模型路径不存在: {model_name}")

    device = resolve_device(device)

    print(f"[INFO] 加载 embedding 模型: {model_name}")
    print(f"[INFO] device = {device}")
    print(f"[INFO] local_files_only = {local_files_only}")

    model = SentenceTransformer(
        model_name_or_path=model_name,
        device=device,
        trust_remote_code=True,
        local_files_only=local_files_only
    )

    vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    vec = np.asarray(vec, dtype="float32")
    if vec.ndim != 2 or vec.shape[0] != 1:
        raise ValueError(f"query embedding 形状异常: {vec.shape}")

    return vec


def dense_search(
    query_vec: np.ndarray,
    faiss_index: faiss.Index,
    chunks: List[Dict[str, Any]],
    topk: int = 10
) -> List[Dict[str, Any]]:
    topk = min(topk, len(chunks), faiss_index.ntotal)

    scores, indices = faiss_index.search(query_vec, topk)
    scores = scores[0]
    indices = indices[0]

    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append(
            {
                "idx": int(idx),
                "rank": rank,
                "score": float(score),
                "chunk": chunks[idx]
            }
        )
    return results


def bm25_score_query(query_tokens: List[str], bm25_index: Dict[str, Any]) -> List[float]:
    num_docs = bm25_index["num_docs"]
    avgdl = bm25_index["avgdl"]
    doc_len = bm25_index["doc_len"]
    idf = bm25_index["idf"]
    term_freqs = bm25_index["term_freqs"]
    k1 = bm25_index["k1"]
    b = bm25_index["b"]

    scores = [0.0] * num_docs
    query_tf = Counter(query_tokens)

    for i in range(num_docs):
        score = 0.0
        tf_doc = term_freqs[i]
        dl = doc_len[i]

        for term, qf in query_tf.items():
            if term not in tf_doc:
                continue

            tf = tf_doc[term]
            term_idf = idf.get(term, 0.0)

            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))

            score += qf * term_idf * (numerator / max(denominator, 1e-9))

        scores[i] = score

    return scores


def sparse_search(
    query: str,
    bm25_index: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    tokenizer,
    topk: int = 10
) -> List[Dict[str, Any]]:
    query_tokens = tokenizer(query)
    if not query_tokens:
        return []

    scores = bm25_score_query(query_tokens, bm25_index)
    topk = min(topk, len(chunks))

    ranked = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )[:topk]

    results = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append(
            {
                "idx": int(idx),
                "rank": rank,
                "score": float(score),
                "chunk": chunks[idx]
            }
        )
    return results


def rrf_fuse(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    用 Reciprocal Rank Fusion 融合两路结果：
    fused_score += weight * 1 / (rrf_k + rank)
    """
    merged: Dict[int, Dict[str, Any]] = {}

    for item in dense_results:
        idx = item["idx"]
        if idx not in merged:
            merged[idx] = {
                "idx": idx,
                "chunk": chunks[idx],
                "dense_rank": None,
                "dense_score": None,
                "sparse_rank": None,
                "sparse_score": None,
                "rrf_score": 0.0
            }

        merged[idx]["dense_rank"] = item["rank"]
        merged[idx]["dense_score"] = item["score"]
        merged[idx]["rrf_score"] += dense_weight * (1.0 / (rrf_k + item["rank"]))

    for item in sparse_results:
        idx = item["idx"]
        if idx not in merged:
            merged[idx] = {
                "idx": idx,
                "chunk": chunks[idx],
                "dense_rank": None,
                "dense_score": None,
                "sparse_rank": None,
                "sparse_score": None,
                "rrf_score": 0.0
            }

        merged[idx]["sparse_rank"] = item["rank"]
        merged[idx]["sparse_score"] = item["score"]
        merged[idx]["rrf_score"] += sparse_weight * (1.0 / (rrf_k + item["rank"]))

    fused = list(merged.values())
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def validate_alignment(
    chunks: List[Dict[str, Any]],
    faiss_index: faiss.Index,
    bm25_index: Dict[str, Any]
):
    n_chunks = len(chunks)
    n_faiss = faiss_index.ntotal
    n_bm25 = bm25_index["num_docs"]

    if n_chunks != n_faiss:
        raise ValueError(f"chunks 数量({n_chunks}) 与 faiss 条目数({n_faiss}) 不一致")
    if n_chunks != n_bm25:
        raise ValueError(f"chunks 数量({n_chunks}) 与 bm25 文档数({n_bm25}) 不一致")

    print(f"[INFO] 对齐检查通过：chunks={n_chunks}, faiss={n_faiss}, bm25={n_bm25}")


def load_retrieval_resources(index_dir: str, prefer_jieba: bool = True) -> Dict[str, Any]:
    """
    统一加载 retriever 所需资源，供外部模块复用。
    """
    faiss_path = os.path.join(index_dir, "faiss.index")
    chunks_path = os.path.join(index_dir, "chunks.json")
    bm25_path = os.path.join(index_dir, "bm25_index.pkl")

    print("[INFO] 读取 chunks.json ...")
    chunks = load_chunks(chunks_path)
    print(f"[INFO] 共读取 {len(chunks)} 个 chunks")

    print("[INFO] 加载 FAISS 索引 ...")
    faiss_index = load_faiss_index(faiss_path)

    print("[INFO] 加载 BM25 索引 ...")
    bm25_index = load_bm25_index(bm25_path)

    validate_alignment(chunks, faiss_index, bm25_index)

    tokenizer, tokenizer_name = get_tokenizer(index_dir, prefer_jieba=prefer_jieba)

    return {
        "index_dir": index_dir,
        "chunks": chunks,
        "faiss_index": faiss_index,
        "bm25_index": bm25_index,
        "tokenizer": tokenizer,
        "tokenizer_name": tokenizer_name,
    }


def hybrid_retrieve_from_resources(
    query: str,
    resources: Dict[str, Any],
    model_name: str,
    device: str = "cuda",
    local_files_only: bool = True,
    dense_topk: int = 10,
    sparse_topk: int = 10,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    在已加载好的资源上执行 hybrid retrieval。
    这个函数适合被 reranker.py 调用，避免重复写检索逻辑。
    """
    query_vec = encode_query(
        query=query,
        model_name=model_name,
        device=device,
        local_files_only=local_files_only
    )

    dense_results = dense_search(
        query_vec=query_vec,
        faiss_index=resources["faiss_index"],
        chunks=resources["chunks"],
        topk=dense_topk
    )

    sparse_results = sparse_search(
        query=query,
        bm25_index=resources["bm25_index"],
        chunks=resources["chunks"],
        tokenizer=resources["tokenizer"],
        topk=sparse_topk
    )

    print(f"[INFO] dense 命中数: {len(dense_results)}")
    print(f"[INFO] sparse 命中数: {len(sparse_results)}")

    fused_results = rrf_fuse(
        dense_results=dense_results,
        sparse_results=sparse_results,
        chunks=resources["chunks"],
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k
    )
    return fused_results


def hybrid_retrieve(
    query: str,
    index_dir: str = "/root/code/agent1/rag/index",
    model_name: str = "/root/models/bge-m3",
    device: str = "cuda",
    local_files_only: bool = True,
    dense_topk: int = 10,
    sparse_topk: int = 10,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60,
    prefer_jieba: bool = True
) -> List[Dict[str, Any]]:
    """
    对外暴露的主函数。
    reranker.py 可以直接 import 这个函数来拿候选结果。
    """
    resources = load_retrieval_resources(index_dir=index_dir, prefer_jieba=prefer_jieba)
    return hybrid_retrieve_from_resources(
        query=query,
        resources=resources,
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
        dense_topk=dense_topk,
        sparse_topk=sparse_topk,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k
    )


def pretty_print_results(results: List[Dict[str, Any]], topk: int = 5, preview_chars: int = 160):
    print("\n[RESULT] 混合召回结果：")
    for rank, item in enumerate(results[:topk], start=1):
        chunk = item["chunk"]
        title = chunk.get("title", "")
        topic = chunk.get("topic", "")
        chunk_index = chunk.get("chunk_index", "")
        preview = normalize_text(chunk.get("text", ""))[:preview_chars]

        print(f"{rank:02d}. idx={item['idx']} | rrf={item['rrf_score']:.6f}")
        print(f"    title={title} | topic={topic} | chunk_index={chunk_index}")
        print(f"    dense_rank={item['dense_rank']} dense_score={item['dense_score']}")
        print(f"    sparse_rank={item['sparse_rank']} sparse_score={item['sparse_score']}")
        print(f"    preview={preview}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_dir",
        type=str,
        default="/root/code/agent1/rag/index",
        help="索引目录，默认 /root/code/agent1/rag/index"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/root/models/bge-m3",
        help="embedding 模型名或本地路径"
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
        default=10,
        help="FAISS 召回 topk"
    )
    parser.add_argument(
        "--sparse_topk",
        type=int,
        default=10,
        help="BM25 召回 topk"
    )
    parser.add_argument(
        "--final_topk",
        type=int,
        default=5,
        help="最终融合后返回 topk"
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
        "--save_json",
        type=str2bool,
        default=False,
        help="是否将结果保存为 retrieval_results.json"
    )
    args = parser.parse_args()

    fused_results = hybrid_retrieve(
        query=args.query,
        index_dir=args.index_dir,
        model_name=args.model_name,
        device=args.device,
        local_files_only=args.local_files_only,
        dense_topk=args.dense_topk,
        sparse_topk=args.sparse_topk,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        rrf_k=args.rrf_k,
        prefer_jieba=args.prefer_jieba
    )

    pretty_print_results(fused_results, topk=args.final_topk)

    if args.save_json:
        save_path = os.path.join(args.index_dir, "retrieval_results.json")
        serializable_results = []
        for item in fused_results[:args.final_topk]:
            serializable_results.append(
                {
                    "idx": item["idx"],
                    "rrf_score": item["rrf_score"],
                    "dense_rank": item["dense_rank"],
                    "dense_score": item["dense_score"],
                    "sparse_rank": item["sparse_rank"],
                    "sparse_score": item["sparse_score"],
                    "chunk": item["chunk"]
                }
            )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 检索结果已保存到: {save_path}")

    print("[DONE] 检索完成。")


if __name__ == "__main__":
    main()