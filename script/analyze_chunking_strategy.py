#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档分块策略分析和改进方案

当前问题分析：
1. 固定长度分块：当前使用固定的CHUNK_SIZE(1000)和CHUNK_OVERLAP(200)，不考虑内容语义
2. 简单边界检测：只在句号、问号、感叹号处分割，忽略了段落、章节等结构
3. 缺乏内容感知：不区分不同类型的内容（标题、正文、列表等）
4. 重叠策略单一：固定重叠长度可能导致信息冗余或丢失

改进方案：
1. 语义感知分块：基于内容结构和语义边界进行分块
2. 自适应长度：根据内容类型动态调整分块大小
3. 智能重叠：基于内容相关性决定重叠策略
4. 多层次分块：支持段落级、章节级等多层次分块
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

class SmartChunker:
    """智能文档分块器"""
    
    def __init__(self, 
                 min_chunk_size: int = 300,
                 max_chunk_size: int = 1500,
                 target_chunk_size: int = 800,
                 overlap_ratio: float = 0.1):
        """
        初始化智能分块器
        
        Args:
            min_chunk_size: 最小分块大小
            max_chunk_size: 最大分块大小
            target_chunk_size: 目标分块大小
            overlap_ratio: 重叠比例（0.0-0.3）
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_chunk_size = target_chunk_size
        self.overlap_ratio = overlap_ratio
        
        # 语义边界标识符
        self.strong_boundaries = [
            r'\n\s*#{1,6}\s+',  # Markdown标题
            r'\n\s*\d+\.\s+',   # 数字列表
            r'\n\s*[一二三四五六七八九十]+[、.]\s+',  # 中文数字列表
            r'\n\s*[（(]\d+[）)]\s+',  # 带括号的数字
            r'\n\s*[A-Za-z]\)\s+',  # 字母列表
        ]
        
        self.medium_boundaries = [
            r'\n\s*\n',  # 空行（段落分隔）
            r'[。！？]\s*\n',  # 句末换行
            r'[。！？]\s+',   # 句末空格
        ]
        
        self.weak_boundaries = [
            r'[，；,;]\s+',  # 逗号、分号
            r'\s+',  # 空格
        ]
    
    def analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """分析文档结构"""
        lines = content.split('\n')
        structure = {
            'total_lines': len(lines),
            'total_chars': len(content),
            'paragraphs': [],
            'headers': [],
            'lists': [],
            'avg_line_length': 0,
            'content_type': 'unknown'
        }
        
        # 分析每行内容
        current_paragraph = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if current_paragraph:
                    structure['paragraphs'].append({
                        'start_line': i - len(current_paragraph),
                        'end_line': i - 1,
                        'content': '\n'.join(current_paragraph),
                        'length': sum(len(l) for l in current_paragraph)
                    })
                    current_paragraph = []
                continue
            
            # 检测标题
            if re.match(r'^#{1,6}\s+', line) or re.match(r'^\d+\.\s+', line):
                structure['headers'].append({
                    'line': i,
                    'content': line,
                    'level': self._get_header_level(line)
                })
            
            # 检测列表
            if re.match(r'^[\s]*[-*+]\s+', line) or re.match(r'^[\s]*\d+\.\s+', line):
                structure['lists'].append({
                    'line': i,
                    'content': line
                })
            
            current_paragraph.append(line)
        
        # 处理最后一个段落
        if current_paragraph:
            structure['paragraphs'].append({
                'start_line': len(lines) - len(current_paragraph),
                'end_line': len(lines) - 1,
                'content': '\n'.join(current_paragraph),
                'length': sum(len(l) for l in current_paragraph)
            })
        
        # 计算平均行长度
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            structure['avg_line_length'] = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        
        # 判断内容类型
        structure['content_type'] = self._determine_content_type(structure)
        
        return structure
    
    def _get_header_level(self, line: str) -> int:
        """获取标题级别"""
        if line.startswith('#'):
            return len(line) - len(line.lstrip('#'))
        elif re.match(r'^\d+\.', line):
            return 1
        return 0
    
    def _determine_content_type(self, structure: Dict[str, Any]) -> str:
        """判断内容类型"""
        total_chars = structure['total_chars']
        headers_count = len(structure['headers'])
        lists_count = len(structure['lists'])
        paragraphs_count = len(structure['paragraphs'])
        
        if headers_count > paragraphs_count * 0.3:
            return 'structured_document'  # 结构化文档
        elif lists_count > paragraphs_count * 0.2:
            return 'list_heavy'  # 列表较多
        elif structure['avg_line_length'] > 50:
            return 'narrative'  # 叙述性文本
        else:
            return 'mixed'  # 混合类型
    
    def find_best_split_points(self, content: str, start: int, target_end: int) -> List[int]:
        """寻找最佳分割点"""
        candidates = []
        
        # 强边界（优先级最高）
        for pattern in self.strong_boundaries:
            for match in re.finditer(pattern, content[start:target_end]):
                candidates.append({
                    'position': start + match.start(),
                    'priority': 3,
                    'type': 'strong'
                })
        
        # 中等边界
        for pattern in self.medium_boundaries:
            for match in re.finditer(pattern, content[start:target_end]):
                candidates.append({
                    'position': start + match.start(),
                    'priority': 2,
                    'type': 'medium'
                })
        
        # 弱边界
        for pattern in self.weak_boundaries:
            for match in re.finditer(pattern, content[start:target_end]):
                candidates.append({
                    'position': start + match.start(),
                    'priority': 1,
                    'type': 'weak'
                })
        
        # 按优先级和位置排序
        candidates.sort(key=lambda x: (-x['priority'], abs(x['position'] - (start + self.target_chunk_size))))
        
        return [c['position'] for c in candidates[:10]]  # 返回前10个候选点
    
    def smart_chunk(self, content: str, doc_id: str = "doc") -> List[Dict[str, Any]]:
        """智能分块"""
        if len(content) <= self.min_chunk_size:
            return [{
                'chunk_id': f"{doc_id}_chunk_0",
                'content': content,
                'start_pos': 0,
                'end_pos': len(content),
                'chunk_index': 0,
                'chunk_type': 'complete',
                'metadata': {
                    'length': len(content),
                    'is_complete_document': True
                }
            }]
        
        # 分析文档结构
        structure = self.analyze_document_structure(content)
        
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(content):
            # 计算目标结束位置
            target_end = min(current_pos + self.target_chunk_size, len(content))
            
            # 如果剩余内容很少，直接包含在当前块中
            if len(content) - current_pos <= self.max_chunk_size:
                chunk_content = content[current_pos:].strip()
                if chunk_content:
                    chunks.append({
                        'chunk_id': f"{doc_id}_chunk_{chunk_index}",
                        'content': chunk_content,
                        'start_pos': current_pos,
                        'end_pos': len(content),
                        'chunk_index': chunk_index,
                        'chunk_type': 'final',
                        'metadata': {
                            'length': len(chunk_content),
                            'content_type': structure['content_type']
                        }
                    })
                break
            
            # 寻找最佳分割点
            split_points = self.find_best_split_points(content, current_pos, target_end)
            
            if split_points:
                # 选择最佳分割点
                best_split = split_points[0]
                for point in split_points:
                    if self.min_chunk_size <= point - current_pos <= self.max_chunk_size:
                        best_split = point
                        break
                
                chunk_end = best_split
            else:
                # 没有找到合适的分割点，使用目标长度
                chunk_end = target_end
            
            # 确保块大小在合理范围内
            chunk_size = chunk_end - current_pos
            if chunk_size < self.min_chunk_size and current_pos + self.max_chunk_size < len(content):
                chunk_end = current_pos + self.min_chunk_size
            elif chunk_size > self.max_chunk_size:
                chunk_end = current_pos + self.max_chunk_size
            
            chunk_content = content[current_pos:chunk_end].strip()
            
            if chunk_content:
                chunks.append({
                    'chunk_id': f"{doc_id}_chunk_{chunk_index}",
                    'content': chunk_content,
                    'start_pos': current_pos,
                    'end_pos': chunk_end,
                    'chunk_index': chunk_index,
                    'chunk_type': 'normal',
                    'metadata': {
                        'length': len(chunk_content),
                        'content_type': structure['content_type']
                    }
                })
                
                chunk_index += 1
            
            # 计算下一个块的起始位置（考虑重叠）
            overlap_size = int(chunk_size * self.overlap_ratio)
            current_pos = max(current_pos + 1, chunk_end - overlap_size)
        
        # 添加总块数信息
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    def analyze_current_strategy(self, content: str) -> Dict[str, Any]:
        """分析当前分块策略的效果"""
        # 当前策略（固定长度）
        current_chunks = self._fixed_length_chunk(content, 1000, 200)
        
        # 智能策略
        smart_chunks = self.smart_chunk(content)
        
        analysis = {
            'document_length': len(content),
            'current_strategy': {
                'chunk_count': len(current_chunks),
                'avg_chunk_size': sum(len(c['content']) for c in current_chunks) / len(current_chunks) if current_chunks else 0,
                'size_variance': self._calculate_variance([len(c['content']) for c in current_chunks]),
                'boundary_quality': self._assess_boundary_quality(current_chunks)
            },
            'smart_strategy': {
                'chunk_count': len(smart_chunks),
                'avg_chunk_size': sum(len(c['content']) for c in smart_chunks) / len(smart_chunks) if smart_chunks else 0,
                'size_variance': self._calculate_variance([len(c['content']) for c in smart_chunks]),
                'boundary_quality': self._assess_boundary_quality(smart_chunks)
            }
        }
        
        return analysis
    
    def _fixed_length_chunk(self, content: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """当前的固定长度分块策略"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + chunk_size
            
            if end < len(content):
                # 寻找句子边界
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if content[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'chunk_id': f"chunk_{chunk_index}",
                    'content': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_index': chunk_index
                })
                chunk_index += 1
            
            start = max(start + 1, end - overlap)
            
            if start >= len(content):
                break
        
        return chunks
    
    def _calculate_variance(self, sizes: List[int]) -> float:
        """计算大小方差"""
        if not sizes:
            return 0
        mean = sum(sizes) / len(sizes)
        return sum((x - mean) ** 2 for x in sizes) / len(sizes)
    
    def _assess_boundary_quality(self, chunks: List[Dict[str, Any]]) -> float:
        """评估边界质量（0-1，越高越好）"""
        if len(chunks) <= 1:
            return 1.0
        
        good_boundaries = 0
        total_boundaries = len(chunks) - 1
        
        for i in range(len(chunks) - 1):
            chunk_end = chunks[i]['content'][-50:] if len(chunks[i]['content']) > 50 else chunks[i]['content']
            
            # 检查是否在句子边界结束
            if re.search(r'[。！？]\s*$', chunk_end) or re.search(r'\n\s*$', chunk_end):
                good_boundaries += 1
        
        return good_boundaries / total_boundaries if total_boundaries > 0 else 1.0


def demo_analysis():
    """演示分析功能"""
    # 示例文档内容
    sample_content = """
# 睡眠呼吸暂停综合征治疗指南

## 1. 疾病概述

睡眠呼吸暂停综合征（Sleep Apnea Syndrome, SAS）是一种常见的睡眠障碍疾病。患者在睡眠过程中反复出现呼吸暂停和低通气事件，导致睡眠质量下降和日间嗜睡。

### 1.1 病因分析

主要病因包括：
1. 上气道解剖异常
2. 肌肉张力降低
3. 神经调节异常
4. 肥胖等危险因素

## 2. 诊断标准

### 2.1 临床症状

患者常见症状包括：
- 夜间打鼾
- 呼吸暂停
- 日间嗜睡
- 注意力不集中
- 记忆力减退

### 2.2 检查方法

确诊需要进行多导睡眠监测（PSG），主要指标包括：
- 呼吸暂停低通气指数（AHI）
- 最低血氧饱和度
- 睡眠结构分析

## 3. 治疗方案

### 3.1 保守治疗

1. 生活方式调整
   - 减重
   - 戒烟戒酒
   - 侧卧睡眠
   - 规律作息

2. 口腔矫治器
   - 下颌前移器
   - 舌保持器

### 3.2 CPAP治疗

持续正压通气（CPAP）是目前最有效的治疗方法：
- 适应症：中重度OSA患者
- 治疗原理：通过正压气流保持气道开放
- 使用注意事项：需要适应期，定期清洁设备

### 3.3 手术治疗

适用于保守治疗无效的患者：
1. 鼻腔手术
2. 软腭手术
3. 舌根手术
4. 下颌前移术

## 4. 预后评估

经过规范治疗后，大多数患者症状可明显改善。定期随访很重要，包括：
- 症状评估
- 睡眠质量评价
- 设备使用情况检查
- 并发症监测

治疗效果的评估指标包括AHI改善程度、日间嗜睡评分变化、生活质量提升等。
    """
    
    chunker = SmartChunker()
    
    print("=== 文档分块策略分析报告 ===")
    print()
    
    # 分析文档结构
    structure = chunker.analyze_document_structure(sample_content)
    print(f"文档结构分析：")
    print(f"- 总字符数: {structure['total_chars']}")
    print(f"- 总行数: {structure['total_lines']}")
    print(f"- 段落数: {len(structure['paragraphs'])}")
    print(f"- 标题数: {len(structure['headers'])}")
    print(f"- 列表项数: {len(structure['lists'])}")
    print(f"- 内容类型: {structure['content_type']}")
    print(f"- 平均行长度: {structure['avg_line_length']:.1f}")
    print()
    
    # 比较分块策略
    analysis = chunker.analyze_current_strategy(sample_content)
    
    print("分块策略对比：")
    print()
    print("当前策略（固定长度）：")
    current = analysis['current_strategy']
    print(f"- 分块数量: {current['chunk_count']}")
    print(f"- 平均块大小: {current['avg_chunk_size']:.1f} 字符")
    print(f"- 大小方差: {current['size_variance']:.1f}")
    print(f"- 边界质量: {current['boundary_quality']:.2f}")
    print()
    
    print("智能策略（语义感知）：")
    smart = analysis['smart_strategy']
    print(f"- 分块数量: {smart['chunk_count']}")
    print(f"- 平均块大小: {smart['avg_chunk_size']:.1f} 字符")
    print(f"- 大小方差: {smart['size_variance']:.1f}")
    print(f"- 边界质量: {smart['boundary_quality']:.2f}")
    print()
    
    # 显示智能分块结果
    smart_chunks = chunker.smart_chunk(sample_content, "demo")
    print("智能分块结果预览：")
    for i, chunk in enumerate(smart_chunks[:3]):
        print(f"\n--- 块 {i+1} ({len(chunk['content'])} 字符) ---")
        preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
        print(preview)
    
    print(f"\n总共生成 {len(smart_chunks)} 个分块")
    
    return analysis


if __name__ == "__main__":
    demo_analysis()