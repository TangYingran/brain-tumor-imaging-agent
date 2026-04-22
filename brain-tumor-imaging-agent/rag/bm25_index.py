# bm25_index.py
# 作用：
# 1. 读取 build_index.py 生成的 chunks.json
# 2. 对每个 chunk 文本做分词
# 3. 构建 BM25 索引
# 4. 保存 BM25 索引与配置，供 retriever.py 加载使用

import os
import re
import json
import math
import pickle
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("布尔参数只支持 true/false")


def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"chunks 文件不存在: {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{chunks_path} 内容不是有效的非空 list")

    return chunks


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

    import jieba  # 延迟导入
    tokens = jieba.lcut(text)
    cleaned = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        # 保留中文词、英文词、数字词
        sub_tokens = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", t)
        cleaned.extend(sub_tokens)
    return cleaned


def get_tokenizer(use_jieba: bool):
    if use_jieba:
        try:
            import jieba  # noqa: F401
            print("[INFO] 检测到 jieba，使用 jieba 分词")
            return jieba_tokenize, "jieba"
        except Exception:
            print("[WARN] 未安装 jieba，自动退回 simple_tokenize")
            return simple_tokenize, "simple"
    else:
        print("[INFO] 使用 simple_tokenize 分词")
        return simple_tokenize, "simple"


def build_bm25_index(
    tokenized_corpus: List[List[str]],
    chunk_ids: List[str],
    k1: float = 1.5,
    b: float = 0.75
) -> Dict[str, Any]:
    """
    构建一个可序列化的 BM25 索引字典。
    """
    if not tokenized_corpus:
        raise ValueError("tokenized_corpus 为空，无法构建 BM25 索引")

    num_docs = len(tokenized_corpus)
    doc_len = [len(doc) for doc in tokenized_corpus]
    avgdl = sum(doc_len) / max(num_docs, 1)

    # 每个文档的词频
    term_freqs: List[Dict[str, int]] = []
    # 每个词的文档频次
    doc_freq: Counter = Counter()

    for doc_tokens in tokenized_corpus:
        tf = Counter(doc_tokens)
        term_freqs.append(dict(tf))
        for term in tf.keys():
            doc_freq[term] += 1

    # BM25 IDF
    idf = {}
    for term, df in doc_freq.items():
        idf[term] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

    bm25_index = {
        "num_docs": num_docs,
        "avgdl": avgdl,
        "doc_len": doc_len,
        "idf": idf,
        "term_freqs": term_freqs,
        "chunk_ids": chunk_ids,
        "k1": k1,
        "b": b
    }

    return bm25_index


def bm25_score_query(
    query_tokens: List[str],
    bm25_index: Dict[str, Any]
) -> List[float]:
    """
    对整个语料计算一个 query 的 BM25 分数。
    返回长度为 num_docs 的分数列表。
    """
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


def search_bm25(
    query: str,
    bm25_index: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    tokenizer,
    topk: int = 5
) -> List[Tuple[int, float, Dict[str, Any]]]:
    query_tokens = tokenizer(query)
    scores = bm25_score_query(query_tokens, bm25_index)

    ranked = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )[:topk]

    results = []
    for idx, score in ranked:
        results.append((idx, score, chunks[idx]))
    return results


def save_pickle(obj: Any, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks_path",
        type=str,
        default="/root/code/agent1/rag/index/chunks.json",
        help="build_index.py 生成的 chunks.json 路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/code/agent1/rag/index",
        help="BM25 索引输出目录"
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 的 k1 参数"
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 的 b 参数"
    )
    parser.add_argument(
        "--use_jieba",
        type=str2bool,
        default=True,
        help="是否优先使用 jieba 分词，默认 true"
    )
    parser.add_argument(
        "--save_tokenized_json",
        type=str2bool,
        default=True,
        help="是否保存 tokenized_corpus.json，默认 true"
    )
    parser.add_argument(
        "--test_query",
        type=str,
        default="",
        help="可选：建库完成后立即测试一个 query"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="测试检索时返回的 topk"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] 读取 chunks.json ...")
    chunks = load_chunks(args.chunks_path)
    print(f"[INFO] 共读取 {len(chunks)} 个 chunks")

    tokenizer, tokenizer_name = get_tokenizer(args.use_jieba)

    print("[INFO] 开始分词 ...")
    tokenized_corpus: List[List[str]] = []
    chunk_ids: List[str] = []

    for i, chunk in enumerate(chunks):
        text = normalize_text(chunk.get("text", ""))
        tokens = tokenizer(text)
        if not tokens:
            tokens = ["[empty]"]

        tokenized_corpus.append(tokens)
        chunk_ids.append(chunk.get("chunk_id", f"chunk_{i}"))

    print("[INFO] 构建 BM25 索引中 ...")
    bm25_index = build_bm25_index(
        tokenized_corpus=tokenized_corpus,
        chunk_ids=chunk_ids,
        k1=args.k1,
        b=args.b
    )

    bm25_pkl_path = os.path.join(args.output_dir, "bm25_index.pkl")
    bm25_config_path = os.path.join(args.output_dir, "bm25_config.json")
    tokenized_json_path = os.path.join(args.output_dir, "tokenized_corpus.json")

    print("[INFO] 保存 BM25 索引 ...")
    save_pickle(bm25_index, bm25_pkl_path)

    bm25_config = {
        "chunks_path": args.chunks_path,
        "output_dir": args.output_dir,
        "num_docs": bm25_index["num_docs"],
        "avgdl": bm25_index["avgdl"],
        "k1": args.k1,
        "b": args.b,
        "tokenizer": tokenizer_name
    }
    save_json(bm25_config, bm25_config_path)

    if args.save_tokenized_json:
        save_json(tokenized_corpus, tokenized_json_path)

    print(f"[INFO] BM25 索引已保存到: {bm25_pkl_path}")
    print(f"[INFO] 配置已保存到: {bm25_config_path}")
    if args.save_tokenized_json:
        print(f"[INFO] 分词结果已保存到: {tokenized_json_path}")

    if args.test_query.strip():
        print(f"[INFO] 开始测试查询: {args.test_query}")
        results = search_bm25(
            query=args.test_query,
            bm25_index=bm25_index,
            chunks=chunks,
            tokenizer=tokenizer,
            topk=args.topk
        )

        print("\n[TEST] BM25 检索结果：")
        for rank, (idx, score, chunk) in enumerate(results, start=1):
            title = chunk.get("title", "")
            text = chunk.get("text", "")
            preview = text[:120].replace("\n", " ")
            print(f"{rank:02d}. idx={idx}  score={score:.6f}  title={title}")
            print(f"    preview: {preview}")

    print("[DONE] BM25 建库完成。")


if __name__ == "__main__":
    main()