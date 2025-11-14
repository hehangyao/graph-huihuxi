import asyncio
import logging
from typing import List, Dict, Any, Optional
import aiohttp
import json
import time
from rag.rag_config import rag_config as config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """嵌入服务 - 使用DashScope API进行文本向量化"""
    
    def __init__(self):
        self.api_key = config.DASHSCOPE_API_KEY
        self.base_url = config.DASHSCOPE_BASE_URL
        self.model = config.EMBEDDING_MODEL
        self.dimensions = config.EMBEDDING_DIMENSIONS
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def embed_query(self, query: str) -> List[float]:
        """向量化单个查询"""
        try:
            embeddings = await self.embed_texts([query])
            return embeddings[0] if embeddings else []
        except Exception as e:
            logger.error(f"查询向量化失败: {str(e)}")
            raise
    
    async def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量（embed_query的别名）"""
        return await self.embed_query(text)
    
    async def embed_texts(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """批量向量化文本"""
        if not texts:
            return []
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = await self._embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                # 添加延迟避免API限流
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"批次 {i//batch_size + 1} 向量化失败: {str(e)}")
                # 为失败的批次添加零向量
                zero_embedding = [0.0] * self.dimensions
                all_embeddings.extend([zero_embedding] * len(batch))
        
        logger.info(f"完成 {len(texts)} 个文本的向量化")
        return all_embeddings
    
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """向量化一个批次的文本"""
        session = await self._get_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建请求数据
        data = {
            "model": self.model,
            "input": {
                "texts": texts
            },
            "parameters": {
                "text_type": "document"
            }
        }
        
        url = f"{self.base_url}/services/embeddings/text-embedding/text-embedding"
        
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._extract_embeddings(result)
                else:
                    error_text = await response.text()
                    logger.error(f"DashScope API错误 {response.status}: {error_text}")
                    raise Exception(f"API请求失败: {response.status}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP请求失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"向量化请求失败: {str(e)}")
            raise
    
    def _extract_embeddings(self, response_data: Dict[str, Any]) -> List[List[float]]:
        """从API响应中提取嵌入向量"""
        try:
            if "output" in response_data and "embeddings" in response_data["output"]:
                embeddings_data = response_data["output"]["embeddings"]
                
                embeddings = []
                for item in embeddings_data:
                    if "embedding" in item:
                        embeddings.append(item["embedding"])
                    else:
                        logger.warning("嵌入项缺少embedding字段")
                        embeddings.append([0.0] * self.dimensions)
                
                return embeddings
            else:
                logger.error(f"API响应格式错误: {response_data}")
                raise Exception("API响应格式错误")
                
        except Exception as e:
            logger.error(f"提取嵌入向量失败: {str(e)}")
            raise
    
    async def test_connection(self) -> bool:
        """测试API连接"""
        try:
            test_embedding = await self.embed_query("测试连接")
            return len(test_embedding) == self.dimensions
        except Exception as e:
            logger.error(f"连接测试失败: {str(e)}")
            return False
    
    async def get_embedding_info(self) -> Dict[str, Any]:
        """获取嵌入模型信息"""
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "api_available": await self.test_connection()
        }
    
    async def close(self):
        """关闭HTTP会话"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("嵌入服务HTTP会话已关闭")
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        if len(embedding1) != len(embedding2):
            raise ValueError("向量维度不匹配")
        
        # 计算点积
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # 计算向量长度
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        
        # 避免除零
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 余弦相似度
        similarity = dot_product / (norm1 * norm2)
        return max(-1.0, min(1.0, similarity))  # 确保结果在[-1, 1]范围内
    
    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """归一化嵌入向量"""
        norm = sum(x * x for x in embedding) ** 0.5
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]