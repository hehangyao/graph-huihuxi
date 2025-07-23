import logging
import hashlib
import re
import os
import mimetypes
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time
from datetime import datetime
import json

logger = logging.getLogger(__name__)

def generate_doc_id(content: str, prefix: str = "doc") -> str:
    """生成文档ID"""
    # 使用内容的MD5哈希作为ID的一部分
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}_{content_hash}"

def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """生成文档块ID"""
    return f"{doc_id}_chunk_{chunk_index}"

def clean_text(text: str) -> str:
    """清理文本内容"""
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留中文、英文、数字、基本标点）
    text = re.sub(r'[^\u4e00-\u9fff\w\s.,;:!?()\[\]{}"\'-]', '', text)
    
    # 去除首尾空白
    text = text.strip()
    
    return text

def estimate_tokens(text: str) -> int:
    """估算文本的token数量"""
    if not text:
        return 0
    
    # 简单估算：中文字符按1个token计算，英文单词按1个token计算
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    
    # 其他字符按0.5个token计算
    other_chars = len(text) - chinese_chars - sum(len(word) for word in re.findall(r'\b[a-zA-Z]+\b', text))
    
    total_tokens = chinese_chars + english_words + int(other_chars * 0.5)
    return max(1, total_tokens)  # 至少1个token

def split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int = 0) -> List[str]:
    """按token数量分割文本"""
    if not text or max_tokens <= 0:
        return []
    
    # 按句子分割
    sentences = re.split(r'[.!?。！？]', text)
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_tokens = estimate_tokens(sentence)
        
        # 如果单个句子就超过最大token数，需要进一步分割
        if sentence_tokens > max_tokens:
            # 保存当前块
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_tokens = 0
            
            # 分割长句子
            sub_chunks = split_long_sentence(sentence, max_tokens)
            chunks.extend(sub_chunks)
            continue
        
        # 检查是否需要开始新块
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            
            # 处理重叠
            if overlap_tokens > 0 and chunks:
                overlap_text = get_text_tail(current_chunk, overlap_tokens)
                current_chunk = overlap_text + " " + sentence
                current_tokens = estimate_tokens(current_chunk)
            else:
                current_chunk = sentence
                current_tokens = sentence_tokens
        else:
            current_chunk += (" " if current_chunk else "") + sentence
            current_tokens += sentence_tokens
    
    # 添加最后一块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]

def split_long_sentence(sentence: str, max_tokens: int) -> List[str]:
    """分割长句子"""
    if estimate_tokens(sentence) <= max_tokens:
        return [sentence]
    
    # 按逗号分割
    parts = re.split(r'[,，;；]', sentence)
    
    chunks = []
    current_chunk = ""
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        part_tokens = estimate_tokens(part)
        current_tokens = estimate_tokens(current_chunk)
        
        if current_tokens + part_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = part
        else:
            current_chunk += ("，" if current_chunk else "") + part
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # 如果还有超长的部分，按字符强制分割
    final_chunks = []
    for chunk in chunks:
        if estimate_tokens(chunk) > max_tokens:
            final_chunks.extend(force_split_by_chars(chunk, max_tokens))
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def force_split_by_chars(text: str, max_tokens: int) -> List[str]:
    """按字符强制分割文本"""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    
    # 估算每个token对应的字符数
    total_chars = len(text)
    total_tokens = estimate_tokens(text)
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 1
    
    # 计算每块的大概字符数
    chars_per_chunk = int(max_tokens * chars_per_token)
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chars_per_chunk, len(text))
        
        # 尝试在合适的位置断开（避免在单词中间断开）
        if end < len(text):
            # 向前查找合适的断点
            for i in range(end, max(start, end - 50), -1):
                if text[i] in ' \n\t，。！？；：':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end
    
    return chunks

def get_text_tail(text: str, max_tokens: int) -> str:
    """获取文本的尾部（用于重叠）"""
    if not text or max_tokens <= 0:
        return ""
    
    if estimate_tokens(text) <= max_tokens:
        return text
    
    # 按句子从后往前取
    sentences = re.split(r'[.!?。！？]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    result = ""
    tokens = 0
    
    for sentence in reversed(sentences):
        sentence_tokens = estimate_tokens(sentence)
        if tokens + sentence_tokens <= max_tokens:
            result = sentence + ("。" if result else "") + result
            tokens += sentence_tokens
        else:
            break
    
    return result

def validate_file_type(file_path: str, allowed_types: List[str] = None) -> bool:
    """验证文件类型"""
    if allowed_types is None:
        allowed_types = ['.txt', '.md', '.pdf', '.docx', '.doc']
    
    file_ext = Path(file_path).suffix.lower()
    return file_ext in allowed_types

def get_file_info(file_path: str) -> Dict[str, Any]:
    """获取文件信息"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    stat = path.stat()
    mime_type, _ = mimetypes.guess_type(str(path))
    
    return {
        "name": path.name,
        "size": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "extension": path.suffix.lower(),
        "mime_type": mime_type,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "absolute_path": str(path.absolute())
    }

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def sanitize_filename(filename: str) -> str:
    """清理文件名"""
    # 移除或替换不安全的字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # 移除控制字符
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # 限制长度
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        max_name_len = 255 - len(ext)
        filename = name[:max_name_len] + ext
    
    return filename.strip()

def create_directory(dir_path: str) -> bool:
    """创建目录"""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"创建目录失败 {dir_path}: {str(e)}")
        return False

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """安全的JSON解析"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """安全的JSON序列化"""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return default

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """截断文本"""
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """提取关键词（简单实现）"""
    if not text:
        return []
    
    # 移除标点符号并转换为小写
    clean_text_content = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text.lower())
    
    # 分词
    words = clean_text_content.split()
    
    # 过滤停用词和短词
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    
    filtered_words = []
    for word in words:
        if len(word) > 1 and word not in stop_words:
            filtered_words.append(word)
    
    # 统计词频
    word_count = {}
    for word in filtered_words:
        word_count[word] = word_count.get(word, 0) + 1
    
    # 按频率排序并返回前N个
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, count in sorted_words[:max_keywords]]

def calculate_text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（简单实现）"""
    if not text1 or not text2:
        return 0.0
    
    # 提取关键词
    keywords1 = set(extract_keywords(text1, 20))
    keywords2 = set(extract_keywords(text2, 20))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # 计算Jaccard相似度
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    similarity = len(intersection) / len(union) if union else 0.0
    return similarity

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    import math
    
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def format_duration(seconds: float) -> str:
    """格式化时间间隔"""
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}m"

def validate_config_value(value: Any, expected_type: type, default: Any = None) -> Any:
    """验证配置值"""
    if value is None:
        return default
    
    try:
        if expected_type == bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        elif expected_type == int:
            return int(value)
        elif expected_type == float:
            return float(value)
        elif expected_type == str:
            return str(value)
        else:
            return value
    except (ValueError, TypeError):
        return default

class Timer:
    """简单的计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self) -> float:
        """停止计时并返回耗时（秒）"""
        if self.start_time is None:
            return 0.0
        
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def elapsed(self) -> float:
        """获取已经过的时间（秒）"""
        if self.start_time is None:
            return 0.0
        
        current_time = self.end_time or time.time()
        return current_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def batch_process(items: List[Any], batch_size: int, process_func, *args, **kwargs) -> List[Any]:
    """批量处理数据"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            batch_result = process_func(batch, *args, **kwargs)
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        except Exception as e:
            logger.error(f"批量处理失败 (batch {i//batch_size + 1}): {str(e)}")
            continue
    
    return results