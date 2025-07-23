import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from datetime import datetime

from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.rerank_service import RerankService
from database import DatabaseManager
from rag_config import rag_config as config

logger = logging.getLogger(__name__)

class RAGService:
    """RAG核心服务"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.rerank_service = RerankService()
        self.db_manager = DatabaseManager()
        self.initialized = False
    
    async def initialize(self):
        """初始化RAG服务"""
        try:
            # 初始化数据库
            await self.db_manager.initialize()
            
            # 测试嵌入服务连接
            await self.embedding_service.test_connection()
            
            # 尝试加载现有索引
            if self.vector_store.load_index():
                logger.info("已加载现有向量索引")
            else:
                logger.info("未找到现有索引，将在添加文档时创建")
            
            self.initialized = True
            logger.info("RAG服务初始化完成")
            
        except Exception as e:
            logger.error(f"RAG服务初始化失败: {str(e)}")
            raise
    
    async def add_document_from_file(self, file_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """从文件添加文档"""
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # 处理文档
            document = await self.document_processor.process_file(file_path, doc_id)
            
            # 添加到RAG系统
            result = await self._add_document_internal(document)
            
            processing_time = time.time() - start_time
            result["processing_time"] = round(processing_time, 2)
            
            logger.info(f"文档添加完成: {document['doc_id']}, 耗时: {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise
    
    async def add_document_from_text(self, content: str, doc_id: Optional[str] = None, 
                                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """从文本内容添加文档"""
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # 处理文档
            document = await self.document_processor.process_text(content, doc_id, metadata)
            
            # 添加到RAG系统
            result = await self._add_document_internal(document)
            
            processing_time = time.time() - start_time
            result["processing_time"] = round(processing_time, 2)
            
            logger.info(f"文档添加完成: {document['doc_id']}, 耗时: {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise
    
    async def _add_document_internal(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """内部文档添加逻辑"""
        doc_id = document["doc_id"]
        chunks = document["chunks"]
        
        # 保存文档到数据库
        await self.db_manager.save_document(document)
        
        # 生成嵌入向量并保存文档块
        chunk_embeddings = []
        saved_chunks = []
        
        for chunk in chunks:
            try:
                # 生成嵌入向量
                embedding = await self.embedding_service.get_embedding(chunk["content"])
                if embedding:
                    chunk["embedding"] = embedding
                    chunk_embeddings.append(embedding)
                    
                    # 保存到数据库
                    await self.db_manager.save_document_chunk(chunk)
                    
                    # 添加到向量存储
                    self.vector_store.add_document(chunk, embedding)
                    
                    saved_chunks.append(chunk)
                else:
                    logger.warning(f"无法为文档块生成嵌入向量: {chunk['chunk_id']}")
                    
            except Exception as e:
                logger.error(f"处理文档块失败 {chunk['chunk_id']}: {str(e)}")
                continue
        
        # 保存向量索引
        if saved_chunks:
            self.vector_store.save_index()
        
        return {
            "doc_id": doc_id,
            "title": document.get("title", ""),
            "total_chunks": len(chunks),
            "processed_chunks": len(saved_chunks),
            "total_tokens": document.get("total_tokens", 0),
            "file_path": document.get("file_path", ""),
            "created_at": document.get("created_at", datetime.now().isoformat())
        }
    
    async def search(self, query: str, top_k: int = 10, 
                    similarity_threshold: float = None, 
                    enable_rerank: bool = None) -> Dict[str, Any]:
        """搜索文档"""
        if not self.initialized:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # 生成查询嵌入向量
            query_embedding = await self.embedding_service.get_embedding(query)
            if not query_embedding:
                raise ValueError("无法生成查询的嵌入向量")
            
            # 向量搜索
            if similarity_threshold is None:
                similarity_threshold = config.SIMILARITY_THRESHOLD
            
            search_results = self.vector_store.search(
                query_embedding, 
                top_k=top_k * 2,  # 获取更多结果用于重排序
                similarity_threshold=similarity_threshold
            )
            
            # 重排序（如果启用）
            if enable_rerank is None:
                enable_rerank = config.ENABLE_RERANK
            
            if enable_rerank and search_results:
                search_results = await self.rerank_service.rerank_documents(
                    query, search_results, top_k
                )
            else:
                search_results = search_results[:top_k]
            
            # 记录搜索历史
            await self._save_search_history(query, len(search_results))
            
            search_time = time.time() - start_time
            
            return {
                "query": query,
                "results": search_results,
                "total_results": len(search_results),
                "search_time": round(search_time, 3),
                "similarity_threshold": similarity_threshold,
                "rerank_enabled": enable_rerank
            }
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise
    
    async def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档信息"""
        try:
            # 从数据库获取文档信息
            document = await self.db_manager.get_document(doc_id)
            if not document:
                return None
            
            # 获取文档块信息
            chunks = await self.db_manager.get_document_chunks(doc_id)
            
            # 获取向量存储中的统计信息
            vector_doc = self.vector_store.get_document_by_id(doc_id)
            
            return {
                "doc_id": document["doc_id"],
                "title": document["title"],
                "file_path": document["file_path"],
                "total_tokens": document["total_tokens"],
                "created_at": document["created_at"],
                "chunk_count": len(chunks),
                "chunks": chunks,
                "in_vector_store": vector_doc is not None
            }
            
        except Exception as e:
            logger.error(f"获取文档信息失败: {str(e)}")
            return None
    
    async def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        try:
            # 从数据库删除
            db_deleted = await self.db_manager.delete_document(doc_id)
            
            # 从向量存储删除
            vector_deleted = self.vector_store.delete_document(doc_id)
            
            # 保存更新后的索引
            if vector_deleted:
                self.vector_store.save_index()
            
            success = db_deleted or vector_deleted
            if success:
                logger.info(f"文档删除成功: {doc_id}")
            else:
                logger.warning(f"文档不存在: {doc_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False
    
    async def list_documents(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """列出文档"""
        try:
            documents = await self.db_manager.list_documents(limit, offset)
            total_count = await self.db_manager.get_document_count()
            
            return {
                "documents": documents,
                "total_count": total_count,
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            logger.error(f"列出文档失败: {str(e)}")
            raise
    
    async def get_similar_documents(self, doc_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """获取相似文档"""
        try:
            return self.vector_store.get_similar_documents(doc_id, top_k)
        except Exception as e:
            logger.error(f"获取相似文档失败: {str(e)}")
            return []
    
    async def rebuild_index(self) -> Dict[str, Any]:
        """重建向量索引"""
        try:
            start_time = time.time()
            
            # 清空现有索引
            self.vector_store.delete_index()
            
            # 从数据库获取所有文档块
            all_chunks = await self.db_manager.get_all_document_chunks()
            
            if not all_chunks:
                logger.info("没有文档块需要重建索引")
                return {
                    "success": True,
                    "processed_chunks": 0,
                    "rebuild_time": 0
                }
            
            # 重新生成嵌入向量
            documents = []
            embeddings = []
            
            for chunk in all_chunks:
                try:
                    content = chunk["content"]
                    embedding = await self.embedding_service.get_embedding(content)
                    
                    if embedding:
                        documents.append(chunk)
                        embeddings.append(embedding)
                    else:
                        logger.warning(f"无法为文档块生成嵌入向量: {chunk['chunk_id']}")
                        
                except Exception as e:
                    logger.error(f"处理文档块失败 {chunk.get('chunk_id', 'unknown')}: {str(e)}")
                    continue
            
            # 创建新索引
            if documents and embeddings:
                self.vector_store.create_index(documents, embeddings)
                self.vector_store.save_index()
            
            rebuild_time = time.time() - start_time
            
            result = {
                "success": True,
                "total_chunks": len(all_chunks),
                "processed_chunks": len(documents),
                "rebuild_time": round(rebuild_time, 2)
            }
            
            logger.info(f"索引重建完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"重建索引失败: {str(e)}")
            raise
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            # 数据库统计
            db_stats = await self.db_manager.get_stats()
            
            # 向量存储统计
            vector_stats = self.vector_store.get_stats()
            
            # 嵌入服务统计
            embedding_info = await self.embedding_service.get_model_info()
            
            return {
                "database": db_stats,
                "vector_store": vector_stats,
                "embedding_service": embedding_info,
                "config": {
                    "embedding_model": config.EMBEDDING_MODEL,
                    "rerank_model": config.RERANK_MODEL,
                    "chunk_size": config.CHUNK_SIZE,
                    "chunk_overlap": config.CHUNK_OVERLAP,
                    "similarity_threshold": config.SIMILARITY_THRESHOLD,
                    "enable_rerank": config.ENABLE_RERANK
                }
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            raise
    
    async def _save_search_history(self, query: str, result_count: int):
        """保存搜索历史"""
        try:
            await self.db_manager.save_search_history(query, result_count)
        except Exception as e:
            logger.error(f"保存搜索历史失败: {str(e)}")
    
    async def get_search_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取搜索历史"""
        try:
            return await self.db_manager.get_search_history(limit)
        except Exception as e:
            logger.error(f"获取搜索历史失败: {str(e)}")
            return []
    
    async def clear_all_data(self) -> Dict[str, Any]:
        """清空所有数据"""
        try:
            # 清空数据库
            await self.db_manager.clear_all_data()
            
            # 清空向量索引
            self.vector_store.delete_index()
            
            logger.info("所有数据已清空")
            return {
                "success": True,
                "message": "所有数据已清空"
            }
            
        except Exception as e:
            logger.error(f"清空数据失败: {str(e)}")
            raise
    
    async def close(self):
        """关闭服务"""
        try:
            await self.embedding_service.close()
            await self.rerank_service.close()
            await self.db_manager.close()
            logger.info("RAG服务已关闭")
        except Exception as e:
            logger.error(f"关闭RAG服务失败: {str(e)}")

# 全局RAG服务实例
rag_service = RAGService()