#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试智能分块策略效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.services.document_processor import DocumentProcessor
from rag.rag_config import rag_config
from pathlib import Path
import json

def test_chunking_strategy():
    """测试新的分块策略"""
    
    # 创建文档处理器
    processor = DocumentProcessor()
    
    # 测试文档路径
    test_file = "d:/develop/ai/graph-huihuxi/input/成人阻塞性睡眠呼吸暂停多学科诊疗指南.txt"
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    print("=== 智能分块策略测试 ===")
    print(f"测试文件: {test_file}")
    print(f"分块大小: {rag_config.CHUNK_SIZE}")
    print(f"重叠大小: {rag_config.CHUNK_OVERLAP}")
    print()
    
    try:
        # 加载单个文档
        print("正在加载文档...")
        document = processor._load_single_document(Path(test_file))
        
        if not document:
            print("未能加载文档")
            return
        
        documents = [document]
        
        document = documents[0]
        print(f"文档加载成功: {document['metadata']['filename']}")
        print(f"文档大小: {len(document['content'])} 字符")
        print()
        
        # 执行分块
        print("正在执行智能分块...")
        chunks = processor._split_single_document(document, rag_config.CHUNK_SIZE, rag_config.CHUNK_OVERLAP)
        
        print(f"分块完成，共生成 {len(chunks)} 个分块")
        print()
        
        # 分析分块结果
        print("=== 分块分析 ===")
        
        # 统计信息
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        avg_size = total_chars / len(chunks) if chunks else 0
        
        chunk_types = {}
        section_levels = {}
        
        for chunk in chunks:
            metadata = chunk['metadata']
            chunk_type = metadata.get('chunk_type', 'unknown')
            section_level = metadata.get('section_level', 0)
            
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            if section_level > 0:
                section_levels[section_level] = section_levels.get(section_level, 0) + 1
        
        print(f"总字符数: {total_chars}")
        print(f"平均分块大小: {avg_size:.1f} 字符")
        print(f"分块类型分布: {chunk_types}")
        print(f"章节级别分布: {section_levels}")
        print()
        
        # 显示前几个分块的详细信息
        print("=== 前5个分块详情 ===")
        for i, chunk in enumerate(chunks[:5]):
            metadata = chunk['metadata']
            print(f"\n分块 {i+1}:")
            print(f"  ID: {chunk['chunk_id']}")
            print(f"  大小: {metadata['chunk_size']} 字符, {metadata.get('line_count', 0)} 行")
            print(f"  类型: {metadata.get('chunk_type', 'unknown')}")
            print(f"  章节级别: {metadata.get('section_level', 0)}")
            print(f"  章节标题: {metadata.get('section_title', 'N/A')}")
            print(f"  内容预览: {chunk['text'][:100]}...")
        
        # 检查分块质量
        print("\n=== 分块质量检查 ===")
        
        # 检查是否有过小的分块
        small_chunks = [c for c in chunks if len(c['text']) < rag_config.CHUNK_SIZE * 0.1]
        if small_chunks:
            print(f"警告: 发现 {len(small_chunks)} 个过小的分块 (<{rag_config.CHUNK_SIZE * 0.1:.0f} 字符)")
        
        # 检查是否有过大的分块
        large_chunks = [c for c in chunks if len(c['text']) > rag_config.CHUNK_SIZE * 1.5]
        if large_chunks:
            print(f"警告: 发现 {len(large_chunks)} 个过大的分块 (>{rag_config.CHUNK_SIZE * 1.5:.0f} 字符)")
        
        # 检查结构保持情况
        heading_chunks = [c for c in chunks if c['metadata'].get('chunk_type') == 'heading']
        print(f"包含标题的分块: {len(heading_chunks)} 个")
        
        # 保存详细结果到文件
        result_file = "d:/develop/ai/graph-huihuxi/script/chunking_test_result.json"
        result_data = {
            "test_file": test_file,
            "chunk_size": rag_config.CHUNK_SIZE,
            "chunk_overlap": rag_config.CHUNK_OVERLAP,
            "total_chunks": len(chunks),
            "total_chars": total_chars,
            "avg_chunk_size": avg_size,
            "chunk_types": chunk_types,
            "section_levels": section_levels,
            "chunks_preview": [
                {
                    "chunk_id": chunk['chunk_id'],
                    "size": chunk['metadata']['chunk_size'],
                    "type": chunk['metadata'].get('chunk_type', 'unknown'),
                    "section_level": chunk['metadata'].get('section_level', 0),
                    "section_title": chunk['metadata'].get('section_title', ''),
                    "preview": chunk['text'][:200]
                }
                for chunk in chunks[:10]  # 只保存前10个分块的预览
            ]
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {result_file}")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chunking_strategy()