# build_index.py
# 作用：
# 1. 读取 rag/corpus/*.jsonl
# 2. 将每条知识拼成检索文本
# 3. 按字符长度切块
# 4. 使用本地 bge-m3 生成向量
# 5. 构建并保存 FAISS 索引与 chunk 元数据

import os
import json
import glob
import uuid
import argparse
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


def resolve_device(device: str) -> str:
    """
    当指定 cuda 但当前环境不可用或初始化失败时，自动切换到 cpu。
    """
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


def load_jsonl_files(corpus_dir: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"语料目录不存在: {corpus_dir}")

    records = []
    jsonl_files = sorted(glob.glob(os.path.join(corpus_dir, "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"在 {corpus_dir} 下未找到任何 .jsonl 文件")

    for fp in jsonl_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("每行 JSON 必须是对象(dict)")
                    item["_source_file"] = os.path.basename(fp)
                    records.append(item)
                except json.JSONDecodeError as e:
                    raise ValueError(f"文件 {fp} 第 {line_no} 行不是合法 JSON: {e}")

    if not records:
        raise ValueError(f"在 {corpus_dir} 下虽找到 .jsonl 文件，但没有读取到有效记录")

    return records


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def build_retrieval_text(record: Dict[str, Any]) -> str:
    """
    把结构化字段拼成适合 embedding 的文本。
    """
    title = str(record.get("title", "")).strip()
    topic = str(record.get("topic", "")).strip()
    entity = str(record.get("entity", "")).strip()
    keywords = "、".join(_ensure_list(record.get("keywords", [])))
    text = str(record.get("text", "")).strip()
    questions = "\n".join(_ensure_list(record.get("questions", [])))

    parts = []
    if title:
        parts.append(f"标题：{title}")
    if topic:
        parts.append(f"主题：{topic}")
    if entity:
        parts.append(f"实体：{entity}")
    if keywords:
        parts.append(f"关键词：{keywords}")
    if text:
        parts.append(f"正文：{text}")
    if questions:
        parts.append(f"常见问法：{questions}")

    result = "\n".join(parts).strip()
    if not result:
        result = json.dumps(record, ensure_ascii=False)

    return result


def split_text(text: str, max_chars: int = 300, overlap: int = 50) -> List[str]:
    """
    简单按字符切块。
    对现在这种短知识条目通常够用。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能小于 0")
    if overlap >= max_chars:
        raise ValueError("overlap 必须小于 max_chars，否则会死循环")

    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_len:
            break

        start = end - overlap

    return chunks


def build_chunks(records: List[Dict[str, Any]], max_chars: int, overlap: int) -> List[Dict[str, Any]]:
    chunk_items = []

    for record in records:
        base_text = build_retrieval_text(record)
        chunk_texts = split_text(base_text, max_chars=max_chars, overlap=overlap)

        for idx, chunk_text in enumerate(chunk_texts):
            chunk_items.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": str(record.get("id", "")),
                    "title": str(record.get("title", "")),
                    "topic": str(record.get("topic", "")),
                    "entity": str(record.get("entity", "")),
                    "language": str(record.get("language", "zh")),
                    "source": str(record.get("source", "")),
                    "source_type": str(record.get("source_type", "")),
                    "source_url": str(record.get("source_url", "")),
                    "keywords": _ensure_list(record.get("keywords", [])),
                    "questions": _ensure_list(record.get("questions", [])),
                    "chunk_index": idx,
                    "text": chunk_text,
                    "_source_file": record.get("_source_file", "")
                }
            )

    if not chunk_items:
        raise ValueError("没有成功生成任何 chunks，请检查语料内容是否为空")

    return chunk_items


def encode_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str = "cuda",
    local_files_only: bool = True
) -> np.ndarray:
    if not texts:
        raise ValueError("texts 为空，无法生成向量")

    # 如果传的是本地路径，先检查目录是否存在
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

    print(f"[INFO] 开始向量化，共 {len(texts)} 个 chunks")
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    vectors = np.asarray(vectors, dtype="float32")

    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(f"向量形状异常: {vectors.shape}")

    return vectors


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    使用余弦相似度：
    因为 encode_texts 中已经 normalize_embeddings=True，
    所以这里直接用 Inner Product 即可。
    """
    if vectors.ndim != 2:
        raise ValueError("vectors 必须是二维数组")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_outputs(output_dir: str, index: faiss.Index, chunks: List[Dict[str, Any]], config: Dict[str, Any]):
    os.makedirs(output_dir, exist_ok=True)

    faiss_path = os.path.join(output_dir, "faiss.index")
    chunks_path = os.path.join(output_dir, "chunks.json")
    config_path = os.path.join(output_dir, "build_config.json")

    faiss.write_index(index, faiss_path)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[INFO] FAISS 索引已保存到: {faiss_path}")
    print(f"[INFO] Chunk 元数据已保存到: {chunks_path}")
    print(f"[INFO] 配置已保存到: {config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="/root/code/agent1/rag/corpus",
        help="jsonl 语料目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/code/agent1/rag/index",
        help="索引输出目录"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/root/models/bge-m3",
        help="embedding 模型名或本地路径"
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=300,
        help="单个 chunk 最大字符数"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="chunk 重叠字符数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="embedding batch size"
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
    args = parser.parse_args()

    print("[INFO] 读取 JSONL 语料中...")
    records = load_jsonl_files(args.corpus_dir)
    print(f"[INFO] 共读取 {len(records)} 条原始知识")

    print("[INFO] 构建 chunks 中...")
    chunks = build_chunks(records, max_chars=args.max_chars, overlap=args.overlap)
    print(f"[INFO] 共生成 {len(chunks)} 个 chunks")

    texts = [c["text"] for c in chunks]

    vectors = encode_texts(
        texts=texts,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=args.local_files_only
    )

    print("[INFO] 构建 FAISS 索引中...")
    index = build_faiss_index(vectors)

    config = {
        "corpus_dir": args.corpus_dir,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "max_chars": args.max_chars,
        "overlap": args.overlap,
        "batch_size": args.batch_size,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "num_records": len(records),
        "num_chunks": len(chunks),
        "embedding_dim": int(vectors.shape[1]),
    }

    save_outputs(args.output_dir, index, chunks, config)

    print("[DONE] 建库完成。")


if __name__ == "__main__":
    main()