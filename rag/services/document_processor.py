import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import hashlib
from pathlib import Path
from rag.rag_config import rag_config as config
from rag.utils import generate_doc_id, clean_text, estimate_tokens, split_text_by_tokens

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, documents_path: str = "documents"):
        self.documents_path = Path(documents_path)
        self.supported_extensions = {".txt", ".md", ".rst"}
        
    def load_documents(self) -> List[Dict[str, Any]]:
        """加载文档目录中的所有文档"""
        documents = []
        
        if not self.documents_path.exists():
            logger.warning(f"文档目录不存在: {self.documents_path}")
            return documents
        
        for file_path in self.documents_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self._load_single_document(file_path)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"加载文档失败 {file_path}: {str(e)}")
        
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def _load_single_document(self, file_path: Path) -> Dict[str, Any]:
        """加载单个文档"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 生成文档ID
            doc_id = self._generate_doc_id(file_path, content)
            
            # 计算基本统计信息
            char_count = len(content)
            word_count = len(content.split())
            token_count = self._estimate_token_count(content)
            
            return {
                "doc_id": doc_id,
                "content": content,
                "metadata": {
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "char_count": char_count,
                    "word_count": word_count,
                    "token_count": token_count,
                    "source": "file"
                }
            }
            
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {str(e)}")
            return None
    
    def _generate_doc_id(self, file_path: Path, content: str) -> str:
        """生成文档ID"""
        # 使用导入的generate_doc_id函数
        return generate_doc_id(content, "doc")
    
    def _estimate_token_count(self, text: str) -> int:
        """估算token数量（简单估算：1个token约等于4个字符）"""
        return len(text) // 4
    
    def split_documents(self, documents: List[Dict[str, Any]], 
                       chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """分割文档为块"""
        chunks = []
        
        for doc in documents:
            doc_chunks = self._split_single_document(
                doc, chunk_size, chunk_overlap
            )
            chunks.extend(doc_chunks)
        
        logger.info(f"文档分割完成，共生成 {len(chunks)} 个块")
        return chunks
    
    def _split_single_document(self, document: Dict[str, Any], 
                              chunk_size: int, 
                              chunk_overlap: int) -> List[Dict[str, Any]]:
        """基于文档结构的智能分割单个文档"""
        content = document["content"]
        doc_id = document["doc_id"]
        
        if len(content) <= chunk_size:
            # 文档太小，不需要分割
            return [{
                "chunk_id": f"{doc_id}_chunk_0",
                "text": content,
                "metadata": {
                    **document["metadata"],
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "chunk_size": len(content),
                    "chunk_type": "complete_document"
                }
            }]
        
        # 使用结构感知分块
        chunks = self._structure_aware_split(content, doc_id, document["metadata"], chunk_size, chunk_overlap)
        
        # 更新总块数信息
        for chunk in chunks:
            chunk["metadata"]["chunk_count"] = len(chunks)
        
        return chunks
    
    def _structure_aware_split(self, content: str, doc_id: str, metadata: Dict[str, Any], 
                              chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """基于文档结构的智能分块"""
        import re
        
        chunks = []
        chunk_index = 0
        
        # 按行分割内容
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        current_section_info = {"level": 0, "title": ""}
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_size = len(line) + 1  # +1 for newline
            
            # 检测结构元素
            structure_info = self._detect_structure(line)
            
            # 如果遇到新的主要结构（标题、表格等）且当前块不为空
            if (structure_info["is_major_break"] and current_chunk and 
                current_size > chunk_size * 0.3):  # 至少30%的目标大小才分块
                
                # 保存当前块
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(self._create_chunk(
                        chunk_text, doc_id, metadata, chunk_index, 
                        current_section_info, len(current_chunk)
                    ))
                    chunk_index += 1
                
                # 开始新块
                current_chunk = [line]
                current_size = line_size
                current_section_info = structure_info
                
            # 如果当前行会使块超过大小限制
            elif current_size + line_size > chunk_size and current_chunk:
                
                # 尝试找到合适的分割点
                split_point = self._find_split_point(current_chunk, chunk_size)
                
                if split_point > 0:
                    # 在找到的分割点分块
                    chunk_text = '\n'.join(current_chunk[:split_point]).strip()
                    if chunk_text:
                        chunks.append(self._create_chunk(
                            chunk_text, doc_id, metadata, chunk_index,
                            current_section_info, split_point
                        ))
                        chunk_index += 1
                    
                    # 保留重叠内容
                    overlap_start = max(0, split_point - self._calculate_overlap_lines(current_chunk[:split_point], chunk_overlap))
                    current_chunk = current_chunk[overlap_start:] + [line]
                    current_size = sum(len(l) + 1 for l in current_chunk)
                else:
                    # 无法找到合适分割点，强制分割
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(self._create_chunk(
                            chunk_text, doc_id, metadata, chunk_index,
                            current_section_info, len(current_chunk)
                        ))
                        chunk_index += 1
                    
                    current_chunk = [line]
                    current_size = line_size
            else:
                # 正常添加行
                current_chunk.append(line)
                current_size += line_size
                
                # 更新章节信息
                if structure_info["level"] > 0:
                    current_section_info = structure_info
            
            i += 1
        
        # 处理最后一个块
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(self._create_chunk(
                    chunk_text, doc_id, metadata, chunk_index,
                    current_section_info, len(current_chunk)
                ))
        
        return chunks
    
    def _detect_structure(self, line: str) -> Dict[str, Any]:
        """检测行的结构类型"""
        import re
        
        line_stripped = line.strip()
        
        # 检测标题（Markdown格式）
        if re.match(r'^#{1,6}\s+', line_stripped):
            level = len(re.match(r'^#+', line_stripped).group())
            title = re.sub(r'^#+\s*', '', line_stripped)
            return {
                "type": "heading",
                "level": level,
                "title": title,
                "is_major_break": level <= 3  # h1-h3为主要分割点
            }
        
        # 检测表格
        if re.match(r'^\|.*\|\s*$', line_stripped) or re.match(r'^\s*[-|:]+\s*$', line_stripped):
            return {
                "type": "table",
                "level": 0,
                "title": "表格",
                "is_major_break": False  # 表格内容不分割
            }
        
        # 检测列表
        if re.match(r'^\s*[-*+]\s+', line_stripped) or re.match(r'^\s*\d+\.\s+', line_stripped):
            return {
                "type": "list",
                "level": 0,
                "title": "列表项",
                "is_major_break": False
            }
        
        # 检测空行（段落分隔）
        if not line_stripped:
            return {
                "type": "empty",
                "level": 0,
                "title": "",
                "is_major_break": False
            }
        
        # 普通文本
        return {
            "type": "text",
            "level": 0,
            "title": "",
            "is_major_break": False
        }
    
    def _find_split_point(self, lines: List[str], target_size: int) -> int:
        """在合适的位置找到分割点"""
        import re
        
        # 优先级：段落边界 > 句子边界 > 强制分割
        best_split = 0
        current_size = 0
        
        for i, line in enumerate(lines):
            current_size += len(line) + 1
            
            # 如果超过目标大小，停止搜索
            if current_size > target_size:
                break
            
            line_stripped = line.strip()
            
            # 段落边界（空行后的非空行）
            if (i > 0 and not lines[i-1].strip() and line_stripped):
                best_split = i
            # 句子边界（以句号、问号、感叹号结尾）
            elif re.search(r'[.!?。！？]\s*$', line_stripped):
                best_split = i + 1
            # 列表项结束
            elif (i < len(lines) - 1 and 
                  re.match(r'^\s*[-*+\d]', line_stripped) and 
                  not re.match(r'^\s*[-*+\d]', lines[i+1].strip())):
                best_split = i + 1
        
        return best_split
    
    def _calculate_overlap_lines(self, lines: List[str], target_overlap: int) -> int:
        """计算重叠的行数"""
        overlap_size = 0
        overlap_lines = 0
        
        # 从后往前计算
        for i in range(len(lines) - 1, -1, -1):
            line_size = len(lines[i]) + 1
            if overlap_size + line_size <= target_overlap:
                overlap_size += line_size
                overlap_lines += 1
            else:
                break
        
        return overlap_lines
    
    def _create_chunk(self, text: str, doc_id: str, metadata: Dict[str, Any], 
                     chunk_index: int, section_info: Dict[str, Any], line_count: int) -> Dict[str, Any]:
        """创建分块对象"""
        return {
            "chunk_id": f"{doc_id}_chunk_{chunk_index}",
            "text": text,
            "metadata": {
                **metadata,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "chunk_size": len(text),
                "line_count": line_count,
                "section_level": section_info.get("level", 0),
                "section_title": section_info.get("title", ""),
                "chunk_type": section_info.get("type", "text")
            }
        }
    
    def add_document_from_text(self, filename: str, content: str, 
                              metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """从文本内容添加文档"""
        # 生成临时文件路径用于ID生成
        temp_path = Path(filename)
        doc_id = self._generate_doc_id(temp_path, content)
        
        # 计算统计信息
        char_count = len(content)
        word_count = len(content.split())
        token_count = self._estimate_token_count(content)
        
        doc_metadata = {
            "filename": filename,
            "file_path": filename,
            "file_size": len(content.encode('utf-8')),
            "char_count": char_count,
            "word_count": word_count,
            "token_count": token_count,
            "source": "text"
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        return {
            "doc_id": doc_id,
            "content": content,
            "metadata": doc_metadata
        }
    
    async def process_file(self, file_path: str, doc_id: str = None) -> Dict[str, Any]:
        """处理文件并返回文档结构"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"不支持的文件类型: {file_path.suffix}")
        
        # 加载文档
        document = self._load_single_document(file_path)
        if not document:
            raise ValueError(f"无法加载文档: {file_path}")
        
        # 如果指定了doc_id，使用指定的ID
        if doc_id:
            document["doc_id"] = doc_id
        
        # 分割文档
        chunks = self._split_single_document(
            document, 
            config.CHUNK_SIZE, 
            config.CHUNK_OVERLAP
        )
        
        # 构建最终文档结构
        return {
            "doc_id": document["doc_id"],
            "title": document["metadata"]["filename"],
            "file_path": str(file_path),
            "content": document["content"],
            "total_tokens": document["metadata"]["token_count"],
            "created_at": datetime.now().isoformat(),
            "chunks": [{
                "chunk_id": chunk["chunk_id"],
                "doc_id": document["doc_id"],
                "content": chunk["text"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "tokens": estimate_tokens(chunk["text"]),
                "metadata": chunk["metadata"]
            } for chunk in chunks]
        }
    
    async def process_text(self, content: str, doc_id: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理文本内容并返回文档结构"""
        if not content.strip():
            raise ValueError("文档内容不能为空")
        
        # 生成文档ID
        if not doc_id:
            doc_id = generate_doc_id(content, "doc")
        
        # 创建文档
        filename = metadata.get("filename", "text_document.txt") if metadata else "text_document.txt"
        document = self.add_document_from_text(filename, content, metadata)
        document["doc_id"] = doc_id
        
        # 分割文档
        chunks = self._split_single_document(
            document,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP
        )
        
        # 构建最终文档结构
        return {
            "doc_id": doc_id,
            "title": filename,
            "file_path": filename,
            "content": content,
            "total_tokens": document["metadata"]["token_count"],
            "created_at": datetime.now().isoformat(),
            "chunks": [{
                "chunk_id": chunk["chunk_id"],
                "doc_id": doc_id,
                "content": chunk["text"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "tokens": estimate_tokens(chunk["text"]),
                "metadata": chunk["metadata"]
            } for chunk in chunks]
        }
    
    def validate_document(self, document: Dict[str, Any]) -> bool:
        """验证文档格式"""
        required_fields = ["doc_id", "content", "metadata"]
        
        for field in required_fields:
            if field not in document:
                logger.error(f"文档缺少必需字段: {field}")
                return False
        
        if not isinstance(document["content"], str):
            logger.error("文档内容必须是字符串")
            return False
        
        if not isinstance(document["metadata"], dict):
            logger.error("文档元数据必须是字典")
            return False
        
        return True