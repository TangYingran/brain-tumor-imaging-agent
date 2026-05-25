"""
联网搜索模块
用于在本地知识无法回答时进行网络搜索
"""

import requests
import json
import urllib.parse
from typing import List, Dict, Any, Optional


def web_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    执行联网搜索
    使用 DuckDuckGo API 进行搜索（无需 API Key）
    
    Args:
        query: 搜索关键词
        num_results: 返回结果数量
    
    Returns:
        搜索结果列表，每个结果包含 title, url, snippet
    """
    try:
        # 使用 DuckDuckGo 搜索 API
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        if "RelatedTopics" in data:
            for topic in data["RelatedTopics"][:num_results]:
                if "Text" in topic and "FirstURL" in topic:
                    results.append({
                        "title": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")[:200]
                    })
        
        return results
    
    except Exception as e:
        print(f"[WARN] 联网搜索失败：{str(e)}")
        return []


def is_out_of_knowledge_scope(answer: str) -> bool:
    """
    判断回答是否表明问题超出知识范围
    
    Args:
        answer: 本地知识返回的回答
        
    Returns:
        True 如果问题超出知识范围，False 否则
    """
    out_of_scope_keywords = [
        "未获得有效回答",
        "无法回答",
        "不知道",
        "不清楚",
        "不了解",
        "超出知识范围",
        "暂无相关知识",
        "未找到相关信息",
        "没有找到相关结果",
        "没有足够的信息",
        "无法提供更多信息"
    ]
    
    return any(keyword in answer for keyword in out_of_scope_keywords)


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    格式化搜索结果为可读文本
    
    Args:
        results: 搜索结果列表
        
    Returns:
        格式化后的文本
    """
    if not results:
        return ""
    
    lines = ["\n【网络搜索结果】"]
    for i, result in enumerate(results, start=1):
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        
        lines.append(f"{i}. {title}")
        if snippet:
            lines.append(f"   {snippet}")
        if url:
            lines.append(f"   来源: {url}")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试搜索功能
    test_query = "脑肿瘤最新治疗方法 2024"
    print(f"搜索测试: {test_query}")
    results = web_search(test_query)
    print(format_search_results(results))