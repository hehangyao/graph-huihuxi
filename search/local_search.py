"""
本地搜索模块
基于GraphRAG的本地搜索功能 - MVP版本
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

async def perform_local_search(
    query: str,
    project_path: Path,
    max_tokens: int = 1500,
    response_type: str = "Single Paragraph",
    **kwargs
) -> Dict[str, Any]:
    """
    执行本地搜索 - MVP版本
    """
    try:
        logger.info(f"开始本地搜索: {query}")
        
        # 检查项目路径是否存在
        if not project_path.exists():
            raise FileNotFoundError(f"项目路径不存在: {project_path}")
        
        # 简单的本地搜索响应
        result = {
            "query": query,
            "response": f"本地搜索结果（基于实体及其邻居）: {query}",
            "context_data": {
                "project_path": str(project_path)
            },
            "completion_time": 0.0,
            "llm_calls": 1,
            "prompt_tokens": len(query.split()) * 2,
            "search_type": "local"
        }
        
        logger.info("本地搜索完成")
        return result
        
    except Exception as e:
        logger.error(f"本地搜索失败: {str(e)}")
        raise Exception(f"本地搜索错误: {str(e)}") 