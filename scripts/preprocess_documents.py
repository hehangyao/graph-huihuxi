#!/usr/bin/env python3
"""
医疗文档预处理脚本
清理可能导致GraphRAG格式字符串错误的特殊字符
基于GitHub Issue和社区解决方案改进
"""

import os
import re
import json
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text_for_graphrag(text: str) -> str:
    """
    清理文本中可能导致GraphRAG格式字符串错误的特殊字符
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本
    """
    # 1. 处理花括号 - 这是最常见的格式字符串错误原因
    text = re.sub(r'(?<!\{)\}(?!\})', '右括号', text)  # 单独的 } 替换为"右括号"
    text = re.sub(r'(?<!\{)\{(?!\{)', '左括号', text)  # 单独的 { 替换为"左括号"
    
    # 2. 处理医学文档中的特殊模式
    text = re.sub(r'(\d+)分呗', r'\1分贝', text)  # 修正"分呗"为"分贝"
    text = re.sub(r'(\d+)次/h', r'\1次每小时', text)  # 处理医学单位
    text = re.sub(r'(\d+)cm\s*H2?O', r'\1厘米水柱', text)  # 处理压力单位
    text = re.sub(r'(\d+)mm\s*Hg', r'\1毫米汞柱', text)  # 处理血压单位
    
    # 3. 处理可能的格式字符串问题字符
    text = re.sub(r'\$\{([^}]+)\}', r'变量\1', text)  # ${variable} 模式
    text = re.sub(r'%\{([^}]+)\}', r'参数\1', text)  # %{parameter} 模式
    text = re.sub(r'#\{([^}]+)\}', r'标识\1', text)  # #{identifier} 模式
    
    # 4. 处理医学文档中的特殊符号组合
    text = re.sub(r'［(\d+)］', r'[\1]', text)  # 全角方括号改为半角
    text = re.sub(r'（([^）]+)）', r'(\1)', text)  # 全角圆括号改为半角
    text = re.sub(r'【([^】]+)】', r'[\1]', text)  # 全角方括号改为半角
    
    # 5. 处理医学术语中的特殊字符
    text = re.sub(r'≥', '大于等于', text)
    text = re.sub(r'≤', '小于等于', text)
    text = re.sub(r'±', '正负', text)
    text = re.sub(r'×', '乘以', text)
    text = re.sub(r'÷', '除以', text)
    
    # 6. 处理可能导致编码问题的字符
    # 保留基本的ASCII、中文、常用标点符号
    text = re.sub(r'[^\u0000-\u007F\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u2000-\u206f]', '', text)
    
    # 7. 处理医学文档中的特殊格式
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1点\2', text)  # 处理小数点
    text = re.sub(r'(\d+)%', r'\1百分比', text)  # 处理百分号
    
    # 8. 清理多余的空白字符
    text = re.sub(r'\s+', ' ', text)  # 多个空白字符合并为一个
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 清理多余的空行
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # 清理行首行尾空白
    
    # 9. 处理可能导致GraphRAG解析问题的模式
    # 移除或替换可能被误解为格式字符串的模式
    text = re.sub(r'([^{])\{([^}]+)\}([^}])', r'\1(\2)\3', text)  # 将单独的{}改为()
    
    return text.strip()

def validate_cleaned_text(text: str) -> bool:
    """
    验证清理后的文本是否安全
    
    Args:
        text: 清理后的文本
    
    Returns:
        是否通过验证
    """
    # 检查是否还有单独的花括号
    if re.search(r'(?<!\{)\}(?!\})', text) or re.search(r'(?<!\{)\{(?!\{)', text):
        logger.warning("文本中仍包含单独的花括号")
        return False
    
    # 检查是否有其他可能的格式字符串问题
    if re.search(r'[^\\]\{[^}]*$', text) or re.search(r'^[^{]*\}', text):
        logger.warning("文本中可能包含未配对的花括号")
        return False
    
    return True

def preprocess_document_file(input_path: Path, output_path: Path) -> bool:
    """
    预处理单个文档文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    
    Returns:
        是否成功处理
    """
    try:
        logger.info(f"处理文档: {input_path}")
        
        # 尝试多种编码方式读取文档
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        content = None
        
        for encoding in encodings:
            try:
                with open(input_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.info(f"成功使用 {encoding} 编码读取文档")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            logger.error(f"无法读取文档 {input_path}，尝试了所有编码方式")
            return False
        
        # 清理文本
        cleaned_content = clean_text_for_graphrag(content)
        
        # 验证清理结果
        if not validate_cleaned_text(cleaned_content):
            logger.warning(f"文档 {input_path} 清理后仍可能存在问题")
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入清理后的内容
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        logger.info(f"成功处理并保存到: {output_path}")
        
        # 生成处理报告
        stats = {
            "original_length": len(content),
            "cleaned_length": len(cleaned_content),
            "reduction_ratio": (len(content) - len(cleaned_content)) / len(content) * 100
        }
        
        report_path = output_path.with_suffix('.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"处理文档失败 {input_path}: {str(e)}")
        return False

def preprocess_all_documents(input_dir: str = "input", output_dir: str = "input_cleaned"):
    """
    预处理所有文档
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_path}")
        return
    
    logger.info(f"开始预处理文档，输入目录: {input_path}, 输出目录: {output_path}")
    
    # 处理所有.txt文件
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        logger.warning(f"在 {input_path} 中没有找到.txt文件")
        return
    
    success_count = 0
    total_count = len(txt_files)
    
    for txt_file in txt_files:
        output_file = output_path / txt_file.name
        if preprocess_document_file(txt_file, output_file):
            success_count += 1
    
    logger.info(f"文档预处理完成: {success_count}/{total_count} 成功")
    
    # 生成总体报告
    summary_report = {
        "total_files": total_count,
        "successful_files": success_count,
        "failed_files": total_count - success_count,
        "success_rate": success_count / total_count * 100
    }
    
    summary_path = output_path / "preprocessing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"预处理总结已保存到: {summary_path}")

if __name__ == "__main__":
    preprocess_all_documents() 