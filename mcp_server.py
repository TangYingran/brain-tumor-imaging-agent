import uuid
from mcp.server.fastmcp import FastMCP

from brain_core import (
    run_segmentation_core,
    analyze_volume_core,
    analyze_result_core,
    query_knowledge_core,
)

mcp = FastMCP("brain-tumor-segmentation")


@mcp.tool()
def brain_tumor_segmentation(
    t1: str,
    flair: str,
    t1ce: str,
    t2: str,
    case_id: str | None = None,
) -> dict:
    """
    对四模态脑肿瘤 MRI（T1/FLAIR/T1CE/T2）执行 3D 分割。
    """
    try:
        case_id = case_id or str(uuid.uuid4())
        return run_segmentation_core(case_id, t1, flair, t1ce, t2)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def analyze_tumor_volume(case_id: str) -> dict:
    """
    基于已有分割结果，对 ET / TC / WT 体积及占比做定量分析。
    """
    try:
        return analyze_volume_core(case_id)
    except Exception as e:
        return {"error": str(e), "case_id": case_id}


@mcp.tool()
def analyze_segmentation_result(case_id: str) -> dict:
    """
    解释当前分割结果中 ET / TC / WT 的医学含义。
    """
    try:
        return analyze_result_core(case_id)
    except Exception as e:
        return {"error": str(e), "case_id": case_id}


@mcp.tool()
def knowledge_query(question: str) -> str | dict:
    """
    查询与脑肿瘤、MRI 模态、多模态影像与分割结果相关的医学知识。
    """
    try:
        return query_knowledge_core(question)
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")