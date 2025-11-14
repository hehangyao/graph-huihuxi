#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG文档处理脚本
用于处理documents目录下的文档文件并建立索引
"""

import os
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_documents():
    """
    处理documents目录下的所有文档文件
    """
    try:
        print("=" * 60)
        print("RAG文档处理脚本")
        print("=" * 60)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 文档目录路径
        documents_dir = Path("./rag/data/documents")
        if not documents_dir.exists():
            documents_dir = Path("./data/documents")
        
        print(f"文档目录: {documents_dir}")
        
        # 初始化RAG服务
        print("\n正在初始化RAG服务...")
        await rag_service.initialize()
        print("✓ RAG服务初始化完成")
        
        # 获取所有txt文件
        txt_files = list(documents_dir.glob("*.txt"))
        if not txt_files:
            print("❌ 未找到任何txt文件")
            return
        
        print(f"\n找到 {len(txt_files)} 个文档文件:")
        for file in txt_files:
            print(f"  - {file.name}")
        
        # 处理每个文档
        print("\n开始处理文档...")
        processed_count = 0
        failed_count = 0
        
        for file_path in txt_files:
            try:
                print(f"\n正在处理: {file_path.name}")
                
                # 添加文档到RAG系统
                result = await rag_service.add_document_from_file(str(file_path))
                
                if result and result.get('doc_id'):
                    processed_count += 1
                    print(f"✓ 处理成功: {file_path.name}")
                    print(f"  - 文档ID: {result.get('doc_id', 'N/A')}")
                    print(f"  - 文档块数: {result.get('total_chunks', 0)}")
                    print(f"  - 处理块数: {result.get('processed_chunks', 0)}")
                    print(f"  - 总tokens: {result.get('total_tokens', 0)}")
                    print(f"  - 处理时间: {result.get('processing_time', 0)}秒")
                else:
                    failed_count += 1
                    print(f"❌ 处理失败: {file_path.name}")
                    if result:
                        print(f"  - 返回结果: {result}")
                    
            except Exception as e:
                failed_count += 1
                print(f"❌ 处理文档失败 {file_path.name}: {str(e)}")
                logger.error(f"处理文档失败 {file_path.name}: {str(e)}")
        
        # 输出处理结果
        print("\n" + "=" * 60)
        print("处理结果:")
        print(f"✓ 总文件数: {len(txt_files)}")
        print(f"✓ 处理成功: {processed_count}")
        print(f"❌ 处理失败: {failed_count}")
        
        if processed_count > 0:
            print("\n正在获取系统统计信息...")
            try:
                stats = await rag_service.get_stats()
                db_stats = stats.get('database', {})
                vector_stats = stats.get('vector_store', {})
                
                print(f"✓ 数据库文档数: {db_stats.get('document_count', 0)}")
                print(f"✓ 数据库文档块数: {db_stats.get('chunk_count', 0)}")
                print(f"✓ 向量索引大小: {vector_stats.get('index_size', 0)}")
            except Exception as e:
                print(f"⚠️  获取统计信息失败: {str(e)}")
        
        print("=" * 60)
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("文档处理完成！")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请检查配置和日志文件")
        logger.error(f"文档处理失败: {str(e)}")
        raise
    
    finally:
        # 关闭RAG服务
        try:
            await rag_service.close()
            print("\n✓ RAG服务已关闭")
        except Exception as e:
            print(f"\n⚠️  关闭RAG服务时出错: {str(e)}")

if __name__ == "__main__":
    asyncio.run(process_documents())