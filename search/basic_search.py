"""
基础搜索模块
基于文本单元的简单搜索功能 - MVP版本
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

async def perform_basic_search(
    query: str,
    project_path: Path,
    max_tokens: int = 1500,
    response_type: str = "Single Paragraph",
    **kwargs
) -> Dict[str, Any]:
    """
    执行基础搜索 - MVP版本
    """
    try:
        logger.info(f"开始基础搜索: {query}")
        
        # 检查项目路径是否存在
        if not project_path.exists():
            raise FileNotFoundError(f"项目路径不存在: {project_path}")
        
        # 简单的基础搜索响应
        result = {
            "query": query,
            "response": f"基础搜索结果（基于文本单元）: {query}",
            "context_data": {
                "project_path": str(project_path)
            },
            "completion_time": 0.0,
            "llm_calls": 0,
            "prompt_tokens": len(query.split()) * 2,
            "search_type": "basic"
        }
        
        logger.info("基础搜索完成")
        return result
        
    except Exception as e:
        logger.error(f"基础搜索失败: {str(e)}")
        raise Exception(f"基础搜索错误: {str(e)}") 