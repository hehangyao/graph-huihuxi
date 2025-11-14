import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from rag.rag_config import rag_config as config

logger = logging.getLogger(__name__)

class RerankService:
    """重排序服务"""
    
    def __init__(self):
        self.api_key = config.DASHSCOPE_API_KEY
        self.model = config.RERANK_MODEL
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-embedding/text-embedding"
        self.session: Optional[aiohttp.ClientSession] = None
        self.max_retries = 3
        self.timeout = 30
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self.session
    
    async def rerank_documents(self, query: str, documents: List[Dict[str, Any]], 
                             top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """重排序文档"""
        if not documents:
            return []
        
        if not self.api_key or not config.ENABLE_RERANK:
            logger.info("重排序功能未启用，返回原始结果")
            return documents[:top_k] if top_k else documents
        
        try:
            # 准备重排序数据
            texts = []
            for doc in documents:
                content = doc.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                else:
                    texts.append(str(content))
            
            if not texts:
                logger.warning("没有有效的文档内容进行重排序")
                return documents
            
            # 调用重排序API
            rerank_scores = await self._call_rerank_api(query, texts)
            
            if not rerank_scores:
                logger.warning("重排序API返回空结果，使用原始排序")
                return documents[:top_k] if top_k else documents
            
            # 合并重排序分数
            reranked_docs = []
            for i, doc in enumerate(documents):
                if i < len(rerank_scores):
                    doc_copy = doc.copy()
                    doc_copy["rerank_score"] = rerank_scores[i]
                    # 计算综合分数（向量相似度 + 重排序分数）
                    vector_score = doc.get("vector_score", 0.0)
                    rerank_score = rerank_scores[i]
                    doc_copy["combined_score"] = self._calculate_combined_score(vector_score, rerank_score)
                    reranked_docs.append(doc_copy)
                else:
                    reranked_docs.append(doc)
            
            # 按综合分数重新排序
            reranked_docs.sort(key=lambda x: x.get("combined_score", x.get("rerank_score", x.get("vector_score", 0))), reverse=True)
            
            result = reranked_docs[:top_k] if top_k else reranked_docs
            logger.info(f"重排序完成，返回 {len(result)} 个文档")
            return result
            
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            # 重排序失败时返回原始结果
            return documents[:top_k] if top_k else documents
    
    async def _call_rerank_api(self, query: str, texts: List[str]) -> List[float]:
        """调用重排序API"""
        # 注意：这里使用的是一个简化的实现
        # 实际的DashScope重排序API可能有不同的接口
        # 这里我们使用文本相似度作为重排序的替代方案
        
        try:
            # 获取查询的嵌入向量
            from services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            
            query_embedding = await embedding_service.embed_query(query)
            if not query_embedding:
                return []
            
            # 获取文档的嵌入向量并计算相似度
            scores = []
            for text in texts:
                if len(text.strip()) == 0:
                    scores.append(0.0)
                    continue
                
                text_embedding = await embedding_service.embed_query(text[:1000])  # 限制长度
                if text_embedding:
                    similarity = embedding_service.calculate_similarity(query_embedding, text_embedding)
                    # 应用非线性变换增强区分度
                    enhanced_score = self._enhance_score(similarity)
                    scores.append(enhanced_score)
                else:
                    scores.append(0.0)
            
            return scores
            
        except Exception as e:
            logger.error(f"调用重排序API失败: {str(e)}")
            return []
    
    def _enhance_score(self, similarity: float) -> float:
        """增强分数的区分度"""
        import math
        
        # 应用sigmoid函数增强高分区间的区分度
        enhanced = 1 / (1 + math.exp(-10 * (similarity - 0.5)))
        return float(enhanced)
    
    def _calculate_combined_score(self, vector_score: float, rerank_score: float) -> float:
        """计算综合分数"""
        # 可以根据需要调整权重
        vector_weight = 0.3
        rerank_weight = 0.7
        
        combined = vector_weight * vector_score + rerank_weight * rerank_score
        return float(combined)
    
    async def batch_rerank(self, queries: List[str], documents_list: List[List[Dict[str, Any]]], 
                          top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """批量重排序"""
        if len(queries) != len(documents_list):
            raise ValueError("查询数量与文档列表数量不匹配")
        
        tasks = []
        for query, documents in zip(queries, documents_list):
            task = self.rerank_documents(query, documents, top_k)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"批量重排序第{i}个查询失败: {str(result)}")
                    # 返回原始文档
                    original_docs = documents_list[i]
                    final_results.append(original_docs[:top_k] if top_k else original_docs)
                else:
                    final_results.append(result)
            
            return final_results
            
        except Exception as e:
            logger.error(f"批量重排序失败: {str(e)}")
            # 返回原始文档列表
            return [docs[:top_k] if top_k else docs for docs in documents_list]
    
    def calculate_relevance_score(self, query: str, document: Dict[str, Any]) -> float:
        """计算文档与查询的相关性分数（同步版本）"""
        try:
            content = document.get("content", "")
            if not content or not query:
                return 0.0
            
            # 简单的关键词匹配分数
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words:
                return 0.0
            
            # 计算交集比例
            intersection = query_words.intersection(content_words)
            relevance = len(intersection) / len(query_words)
            
            # 考虑文档长度因素
            length_factor = min(1.0, len(content) / 500)  # 500字符为基准
            
            final_score = relevance * length_factor
            return float(final_score)
            
        except Exception as e:
            logger.error(f"计算相关性分数失败: {str(e)}")
            return 0.0
    
    async def close(self):
        """关闭HTTP会话"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("重排序服务HTTP会话已关闭")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'session') and self.session and not self.session.closed:
            # 在事件循环中关闭会话
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.session.close())
                else:
                    loop.run_until_complete(self.session.close())
            except:
                pass

# 全局重排序服务实例
rerank_service = RerankService()

# 便捷函数
async def rerank_search_results(query: str, documents: List[Dict[str, Any]], 
                               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """重排序搜索结果的便捷函数"""
    return await rerank_service.rerank_documents(query, documents, top_k)

def calculate_document_relevance(query: str, document: Dict[str, Any]) -> float:
    """计算文档相关性的便捷函数"""
    return rerank_service.calculate_relevance_score(query, document)