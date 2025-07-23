import os
import logging
from typing import List, Dict, Any
import hashlib
from pathlib import Path
from rag_config import rag_config as config
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
        """分割单个文档"""
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
                    "chunk_size": len(content)
                }
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # 如果不是最后一块，尝试在句子边界分割
            if end < len(content):
                # 寻找最近的句号、问号或感叹号
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if content[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                    "text": chunk_text,
                    "metadata": {
                        **document["metadata"],
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_text),
                        "start_pos": start,
                        "end_pos": end
                    }
                })
                
                chunk_index += 1
            
            # 计算下一个块的起始位置（考虑重叠）
            start = max(start + 1, end - chunk_overlap)
            
            # 避免无限循环
            if start >= len(content):
                break
        
        # 更新总块数信息
        for chunk in chunks:
            chunk["metadata"]["chunk_count"] = len(chunks)
        
        return chunks
    
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