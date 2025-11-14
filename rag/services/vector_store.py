import logging
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from rag.rag_config import rag_config as config

logger = logging.getLogger(__name__)

class VectorStore:
    """简单的内存向量存储"""
    
    def __init__(self, index_path: str = None):
        self.index_path = Path(index_path or config.INDEX_PATH)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.doc_id_to_index: Dict[str, int] = {}
        self.initialized = False
        
    def create_index(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """创建向量索引"""
        if len(documents) != len(embeddings):
            raise ValueError("文档数量与嵌入向量数量不匹配")
        
        self.documents = documents.copy()
        self.embeddings = embeddings.copy()
        
        # 创建文档ID到索引的映射
        self.doc_id_to_index = {}
        for i, doc in enumerate(documents):
            chunk_id = doc.get("chunk_id") or doc.get("doc_id", f"doc_{i}")
            self.doc_id_to_index[chunk_id] = i
        
        self.initialized = True
        logger.info(f"向量索引创建完成，包含 {len(documents)} 个文档")
    
    def add_document(self, document: Dict[str, Any], embedding: List[float]):
        """添加单个文档到索引"""
        chunk_id = document.get("chunk_id") or document.get("doc_id", f"doc_{len(self.documents)}")
        
        # 检查是否已存在
        if chunk_id in self.doc_id_to_index:
            # 更新现有文档
            index = self.doc_id_to_index[chunk_id]
            self.documents[index] = document
            self.embeddings[index] = embedding
            logger.debug(f"更新文档: {chunk_id}")
        else:
            # 添加新文档
            index = len(self.documents)
            self.documents.append(document)
            self.embeddings.append(embedding)
            self.doc_id_to_index[chunk_id] = index
            logger.debug(f"添加文档: {chunk_id}")
        
        self.initialized = True
    
    def search(self, query_embedding: List[float], top_k: int = 10, 
              similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        if not self.initialized or not self.embeddings:
            logger.warning("向量索引未初始化或为空")
            return []
        
        if similarity_threshold is None:
            similarity_threshold = 0.0  # 默认不过滤
        
        # 计算相似度
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 获取top_k结果
        results = []
        for similarity, index in similarities[:top_k]:
            if similarity >= similarity_threshold:
                doc = self.documents[index].copy()
                doc["vector_score"] = float(similarity)
                results.append(doc)
        
        logger.debug(f"搜索返回 {len(results)} 个结果")
        return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            logger.error(f"向量维度不匹配: {len(vec1)} vs {len(vec2)}")
            return 0.0
        
        # 转换为numpy数组以提高计算效率
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # 计算余弦相似度
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(np.clip(similarity, -1.0, 1.0))
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        if doc_id in self.doc_id_to_index:
            index = self.doc_id_to_index[doc_id]
            return self.documents[index].copy()
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        if doc_id not in self.doc_id_to_index:
            return False
        
        index = self.doc_id_to_index[doc_id]
        
        # 删除文档和嵌入
        del self.documents[index]
        del self.embeddings[index]
        del self.doc_id_to_index[doc_id]
        
        # 更新索引映射
        for chunk_id, idx in self.doc_id_to_index.items():
            if idx > index:
                self.doc_id_to_index[chunk_id] = idx - 1
        
        logger.info(f"删除文档: {doc_id}")
        return True
    
    def save_index(self, file_path: str = None):
        """保存索引到文件"""
        if not self.initialized:
            logger.warning("索引未初始化，无法保存")
            return
        
        save_path = Path(file_path) if file_path else self.index_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "doc_id_to_index": self.doc_id_to_index,
            "metadata": {
                "document_count": len(self.documents),
                "embedding_dimensions": len(self.embeddings[0]) if self.embeddings else 0,
                "version": "1.0"
            }
        }
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(index_data, f)
            logger.info(f"索引已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存索引失败: {str(e)}")
            raise
    
    def load_index(self, file_path: str = None) -> bool:
        """从文件加载索引"""
        load_path = Path(file_path) if file_path else self.index_path
        
        if not load_path.exists():
            logger.info(f"索引文件不存在: {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data["documents"]
            self.embeddings = index_data["embeddings"]
            self.doc_id_to_index = index_data["doc_id_to_index"]
            self.initialized = True
            
            metadata = index_data.get("metadata", {})
            logger.info(f"索引加载成功: {metadata.get('document_count', len(self.documents))} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"加载索引失败: {str(e)}")
            return False
    
    def delete_index(self, file_path: str = None):
        """删除索引文件和内存数据"""
        # 清空内存数据
        self.documents = []
        self.embeddings = []
        self.doc_id_to_index = {}
        self.initialized = False
        
        # 删除文件
        delete_path = Path(file_path) if file_path else self.index_path
        if delete_path.exists():
            try:
                delete_path.unlink()
                logger.info(f"索引文件已删除: {delete_path}")
            except Exception as e:
                logger.error(f"删除索引文件失败: {str(e)}")
        
        logger.info("向量索引已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self.initialized:
            return {
                "initialized": False,
                "document_count": 0,
                "embedding_dimensions": 0
            }
        
        return {
            "initialized": True,
            "document_count": len(self.documents),
            "embedding_dimensions": len(self.embeddings[0]) if self.embeddings else 0,
            "index_size_mb": self._calculate_index_size(),
            "unique_doc_ids": len(self.doc_id_to_index)
        }
    
    def _calculate_index_size(self) -> float:
        """计算索引大小（MB）"""
        try:
            # 估算内存使用量
            docs_size = len(str(self.documents).encode('utf-8'))
            embeddings_size = len(self.embeddings) * len(self.embeddings[0]) * 8 if self.embeddings else 0  # float64
            mapping_size = len(str(self.doc_id_to_index).encode('utf-8'))
            
            total_size = docs_size + embeddings_size + mapping_size
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0.0
    
    def is_initialized(self) -> bool:
        """检查索引是否已初始化"""
        return self.initialized and len(self.documents) > 0
    
    def get_similar_documents(self, doc_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """获取与指定文档相似的文档"""
        if doc_id not in self.doc_id_to_index:
            return []
        
        index = self.doc_id_to_index[doc_id]
        doc_embedding = self.embeddings[index]
        
        # 搜索相似文档（排除自身）
        results = self.search(doc_embedding, top_k + 1)
        
        # 过滤掉自身
        filtered_results = []
        for result in results:
            result_id = result.get("chunk_id") or result.get("doc_id")
            if result_id != doc_id:
                filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break
        
        return filtered_results