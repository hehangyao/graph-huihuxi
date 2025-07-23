import asyncio
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

from rag_config import rag_config as config
from database import db_manager
from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.rerank_service import RerankService

# 路由将在后续版本中添加

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局服务实例
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_store = VectorStore()
rerank_service = RerankService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    try:
        # 验证配置
        config.validate()
        logger.info("配置验证成功")
        
        # 初始化数据库连接
        try:
            await db_manager.initialize()
            logger.info("数据库连接初始化成功")
        except Exception as e:
            logger.error(f"数据库连接初始化失败: {str(e)}")
            raise RuntimeError("数据库连接初始化失败")
        
        # 尝试加载现有索引
        if vector_store.load_index():
            logger.info("成功加载现有向量索引")
        else:
            logger.info("未找到现有索引，需要先建立索引")
        
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        raise
    
    yield  # 应用运行期间
    
    # 应用关闭时的清理工作
    logger.info("应用正在关闭")
    await db_manager.close()
    logger.info("数据库连接已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="RAG System API",
    description="基于向量检索和重排序的RAG系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册将在后续版本中添加

# Pydantic模型
class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = None
    rerank_top_n: Optional[int] = None
    use_rerank: bool = True

class IndexResponse(BaseModel):
    message: str
    document_count: int
    chunk_count: int
    total_tokens: int

class SearchResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    vector_score: Optional[float] = None
    relevance_score: Optional[float] = None
    rank: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
    search_time: float
    used_rerank: bool

@app.get("/rag/")
async def root():
    """根路径"""
    # 触发重新加载
    return {"message": "RAG System API", "version": "1.0.0"}

@app.get("/rag/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store.is_initialized(),
        "config_valid": bool(config.DASHSCOPE_API_KEY)
    }

@app.get("/rag/stats")
async def get_stats():
    """获取系统统计信息"""
    try:
        vector_stats = vector_store.get_stats()
        return {
            "vector_store": vector_stats,
            "config": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "top_k": config.TOP_K,
                "rerank_top_n": config.RERANK_TOP_N,
                "embedding_model": config.EMBEDDING_MODEL,
                "rerank_model": config.RERANK_MODEL
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@app.post("/rag/index", response_model=IndexResponse)
async def create_index(background_tasks: BackgroundTasks):
    """创建或重建向量索引"""
    try:
        logger.info("开始创建向量索引")
        
        # 1. 加载文档
        documents = document_processor.load_documents()
        if not documents:
            raise HTTPException(status_code=400, detail="未找到可处理的文档")
        
        # 2. 分割文档
        chunks = document_processor.split_documents(documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="文档分割失败")
        
        # 3. 提取文本用于向量化
        texts = [chunk["text"] for chunk in chunks]
        
        # 4. 批量向量化
        embeddings = await embedding_service.embed_texts(texts)
        
        # 5. 创建向量索引
        vector_store.create_index(chunks, embeddings)
        
        # 6. 保存索引
        vector_store.save_index()
        
        # 计算统计信息
        total_tokens = sum(doc["metadata"]["token_count"] for doc in documents)
        
        logger.info("向量索引创建完成")
        
        return IndexResponse(
            message="向量索引创建成功",
            document_count=len(documents),
            chunk_count=len(chunks),
            total_tokens=total_tokens
        )
        
    except Exception as e:
        logger.error(f"创建索引失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建索引失败: {str(e)}")

@app.post("/rag/search", response_model=SearchResponse)
async def search_documents(query: SearchQuery):
    """搜索文档"""
    import time
    
    start_time = time.time()
    
    try:
        # 检查向量存储是否已初始化
        if not vector_store.is_initialized():
            raise HTTPException(status_code=400, detail="向量索引未初始化，请先创建索引")
        
        logger.info(f"开始搜索查询: {query.query}")
        
        # 1. 向量化查询
        query_vector = await embedding_service.embed_query(query.query)
        
        # 2. 向量搜索
        top_k = query.top_k or config.TOP_K
        search_results = vector_store.search(query_vector, top_k)
        
        # 3. 重排序（如果启用）
        if query.use_rerank and search_results:
            reranked_results = await rerank_service.rerank_documents(
                query.query,
                search_results,
                query.rerank_top_n
            )
            
            # 构建最终结果
            final_results = []
            for i, doc in enumerate(reranked_results):
                result = SearchResult(
                    text=doc["text"],
                    metadata=doc["metadata"],
                    vector_score=doc.get("vector_score"),
                    relevance_score=doc.get("relevance_score"),
                    rank=i + 1
                )
                final_results.append(result)
        else:
            # 不使用重排序
            final_results = []
            for i, doc in enumerate(search_results):
                result = SearchResult(
                    text=doc["text"],
                    metadata=doc["metadata"],
                    vector_score=doc.get("vector_score"),
                    rank=i + 1
                )
                final_results.append(result)
        
        search_time = time.time() - start_time
        
        logger.info(f"搜索完成，返回 {len(final_results)} 个结果，耗时 {search_time:.2f}s")
        
        return SearchResponse(
            query=query.query,
            results=final_results,
            total_found=len(final_results),
            search_time=search_time,
            used_rerank=query.use_rerank and len(search_results) > 0
        )
        
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.delete("/rag/index")
async def delete_index():
    """删除向量索引"""
    try:
        vector_store.delete_index()
        return {"message": "向量索引已删除"}
    except Exception as e:
        logger.error(f"删除索引失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除索引失败: {str(e)}")

# GraphRAG兼容接口
class GraphRagQuery(BaseModel):
    query: str

@app.post("/rag/query")
async def graphrag_query(request: GraphRagQuery):
    """GraphRAG兼容接口 - 重定向到向量RAG搜索"""
    try:
        # 使用默认参数进行向量搜索
        search_query = SearchQuery(
            query=request.query,
            top_k=config.TOP_K,
            rerank_top_n=config.RERANK_TOP_N,
            use_rerank=True
        )
        
        # 调用现有的搜索功能
        search_response = await search_documents(search_query)
        
        # 转换为GraphRAG兼容格式
        if search_response.results:
            # 合并所有结果文本
            combined_text = "\n\n".join([result.text for result in search_response.results[:3]])
            return {"result": combined_text}
        else:
            return {"result": "未找到相关信息"}
            
    except Exception as e:
        logger.error(f"GraphRAG查询失败: {str(e)}")
        return {"result": "查询失败，请稍后重试"}

# 向量RAG专用接口
@app.post("/rag/ragquery", response_model=SearchResponse)
async def vector_rag_query(query: SearchQuery):
    """向量RAG专用查询接口"""
    return await search_documents(query)

@app.get("/rag/documents")
async def list_documents():
    """列出所有文档"""
    try:
        documents = document_processor.load_documents()
        doc_info = []
        for doc in documents:
            doc_info.append({
                "filename": doc["metadata"]["filename"],
                "file_size": doc["metadata"]["file_size"],
                "token_count": doc["metadata"]["token_count"],
                "source": doc["metadata"]["source"]
            })
        
        return {
            "documents": doc_info,
            "total_count": len(doc_info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"列出文档失败: {str(e)}")

if __name__ == "__main__":
    # 运行服务器
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )