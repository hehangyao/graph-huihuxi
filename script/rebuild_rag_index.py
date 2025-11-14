#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG索引重建脚本
使用RAGService的rebuild_index方法重新构建向量索引
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入RAG服务
from rag.services.rag_service import rag_service
from rag.rag_config import rag_config as config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """主函数"""
    try:
        print("=" * 60)
        print("RAG索引重建脚本")
        print("=" * 60)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"文档路径: {config.DOCUMENT_STORAGE_PATH}")
        print(f"索引路径: {config.VECTOR_INDEX_PATH}")
        print()
        
        # 初始化RAG服务
        print("正在初始化RAG服务...")
        await rag_service.initialize()
        print("✓ RAG服务初始化完成")
        print()
        
        # 重建索引
        print("正在重建向量索引...")
        print("这可能需要几分钟时间，请耐心等待...")
        
        result = await rag_service.rebuild_index()
        
        print()
        print("=" * 60)
        print("重建结果:")
        print(f"✓ 成功: {result['success']}")
        if result['success']:
            print(f"✓ 总文档块: {result.get('total_chunks', 0)}")
            print(f"✓ 处理文档块: {result.get('processed_chunks', 0)}")
            print(f"✓ 重建耗时: {result.get('rebuild_time', 0)}秒")
        
        total_chunks = result.get('total_chunks', 0)
        processed_chunks = result.get('processed_chunks', 0)
        if total_chunks != processed_chunks:
            failed_count = total_chunks - processed_chunks
            print(f"⚠ 失败块数: {failed_count}")
            print("请检查日志了解失败原因")
        
        print("=" * 60)
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("索引重建完成！")
        
    except Exception as e:
        logger.error(f"重建索引失败: {str(e)}")
        print(f"\n❌ 错误: {str(e)}")
        print("请检查配置和日志文件")
        sys.exit(1)
    
    finally:
        # 关闭服务
        try:
            await rag_service.close()
            print("\n✓ RAG服务已关闭")
        except Exception as e:
            logger.error(f"关闭服务失败: {str(e)}")

if __name__ == "__main__":
    # 检查环境变量
    if not config.DASHSCOPE_API_KEY:
        print("❌ 错误: 未设置DASHSCOPE_API_KEY环境变量")
        print("请设置环境变量后重试")
        sys.exit(1)
    
    # 运行脚本
    asyncio.run(main())