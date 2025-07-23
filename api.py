from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn
import logging
import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
import time

from utils import process_context_data

from pathlib import Path
import graphrag.api as api
from graphrag.config.load_config import load_config
import pandas as pd
from config import PROJECT_DIRECTORY, COMMUNITY_LEVEL, CLAIM_EXTRACTION_ENABLED, RESPONSE_TYPE

# 添加RAG模块路径
rag_path = os.path.join(os.path.dirname(__file__), 'rag')
sys.path.insert(0, rag_path)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入RAG服务模块
try:
    # 临时修改sys.path以避免模块名冲突
    original_path = sys.path.copy()
    rag_module_path = os.path.join(os.path.dirname(__file__), 'rag')
    if rag_module_path not in sys.path:
        sys.path.insert(0, rag_module_path)
    
    # 导入RAG模块
    import importlib.util
    
    # 导入config模块
    config_spec = importlib.util.spec_from_file_location("rag_config", os.path.join(rag_module_path, "rag_config.py"))
    rag_config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(rag_config_module)
    rag_config = rag_config_module.rag_config
    
    # 导入其他模块
    from database import db_manager
    from services.document_processor import DocumentProcessor
    from services.embedding_service import EmbeddingService
    from services.vector_store import VectorStore
    from services.rerank_service import RerankService
    
    # 恢复原始路径
    sys.path = original_path
    
    rag_available = True
    logger.info("RAG服务模块导入成功")
except ImportError as e:
    logger.warning(f"RAG服务模块导入失败: {e}")
    rag_available = False
except Exception as e:
    logger.warning(f"RAG服务模块导入异常: {e}")
    rag_available = False

class SearchRequest(BaseModel):
    query: str

# RAG相关的Pydantic模型
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

class GraphRagQuery(BaseModel):
    query: str

# RAG服务实例（如果可用）
if rag_available:
    document_processor = DocumentProcessor(rag_config.DOCUMENT_STORAGE_PATH)
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    rerank_service = RerankService()
else:
    document_processor = None
    embedding_service = None
    vector_store = None
    rerank_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("正在加载配置和数据文件...")
        app.state.config = load_config(Path(PROJECT_DIRECTORY))
        app.state.entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
        app.state.communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
        app.state.community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/community_reports.parquet")
        app.state.text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")
        app.state.relationships = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")
        
        # 只有在启用声明提取时才加载 covariates
        if CLAIM_EXTRACTION_ENABLED:
            try:
                app.state.covariates = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/covariates.parquet")
                logger.info("已加载 covariates.parquet")
            except FileNotFoundError:
                logger.warning("covariates.parquet 文件不存在，设置为 None")
                app.state.covariates = None
        else:
            app.state.covariates = None
            logger.info("声明提取已禁用，covariates 设置为 None")
        
        logger.info("所有数据文件加载完成")
        
        # 初始化RAG服务（如果可用）
        if rag_available:
            try:
                # 验证RAG配置
                rag_config.validate()
                logger.info("RAG配置验证成功")
                
                # 初始化数据库连接
                try:
                    await db_manager.initialize()
                    logger.info("RAG数据库连接初始化成功")
                except Exception as e:
                    logger.error(f"RAG数据库连接初始化失败: {str(e)}")
                    raise RuntimeError("RAG数据库连接初始化失败")
                
                # 尝试加载现有索引
                if vector_store.load_index():
                    logger.info("成功加载现有RAG向量索引")
                else:
                    logger.info("未找到现有RAG索引，需要先建立索引")
                    
                app.state.rag_initialized = True
                logger.info("RAG服务初始化完成")
            except Exception as e:
                logger.error(f"RAG服务初始化失败: {str(e)}")
                app.state.rag_initialized = False
        else:
            app.state.rag_initialized = False
            logger.info("RAG服务不可用")
        
        yield
        
        # 应用关闭时的清理工作
        if rag_available and app.state.rag_initialized:
            logger.info("正在关闭RAG服务")
            await db_manager.close()
            logger.info("RAG数据库连接已关闭")
            
    except Exception as e:
        logger.error(f"启动时发生错误: {str(e)}")
        raise

app = FastAPI(lifespan=lifespan, title="GraphRAG API", description="GraphRAG搜索API服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 根路径
@app.get("/")
async def root():
    return {"message": "GraphRAG API 服务运行中", "docs": "/docs"}

@app.get("/search/global")
async def global_search_get(query: str = Query(..., description="Global Search")):
    try:
        logger.info(f"执行全局搜索: {query}")
        response, context = await api.global_search(
                                config=app.state.config,
                                entities=app.state.entities,
                                communities=app.state.communities,
                                community_reports=app.state.community_reports,                                
                                community_level=COMMUNITY_LEVEL,
                                dynamic_community_selection=False,
                                response_type=RESPONSE_TYPE,
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        logger.error(f"全局搜索错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"全局搜索失败: {str(e)}")

@app.post("/global_search")
async def global_search_post(request: SearchRequest):
    return await global_search_get(request.query)

@app.get("/search/local")
async def local_search_get(query: str = Query(..., description="Local Search")):
    try:
        logger.info(f"执行本地搜索: {query}")
        response, context = await api.local_search(
                                config=app.state.config,
                                entities=app.state.entities,
                                communities=app.state.communities,
                                community_reports=app.state.community_reports,
                                text_units=app.state.text_units,
                                relationships=app.state.relationships,
                                covariates=app.state.covariates,
                                community_level=COMMUNITY_LEVEL,                                
                                response_type=RESPONSE_TYPE,
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }        
        logger.info(f"本地搜索响应: {response_dict}")
        return JSONResponse(content=response_dict)
    except Exception as e:
        logger.error(f"本地搜索错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"本地搜索失败: {str(e)}")

@app.post("/local_search")
async def local_search_post(request: SearchRequest):
    return await local_search_get(request.query)

@app.get("/search/drift")
async def drift_search_get(query: str = Query(..., description="DRIFT Search")):
    try:
        logger.info(f"执行DRIFT搜索: {query}")
        response, context = await api.drift_search(
                                config=app.state.config,
                                entities=app.state.entities,
                                communities=app.state.communities,
                                community_reports=app.state.community_reports,
                                text_units=app.state.text_units,
                                relationships=app.state.relationships,
                                community_level=COMMUNITY_LEVEL,                                
                                response_type=RESPONSE_TYPE,
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        logger.error(f"DRIFT搜索错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DRIFT搜索失败: {str(e)}")

@app.post("/drift_search")
async def drift_search_post(request: SearchRequest):
    return await drift_search_get(request.query)

@app.get("/search/basic")
async def basic_search_get(query: str = Query(..., description="Basic Search")):
    try:
        logger.info(f"执行基础搜索: {query}")
        response, context = await api.basic_search(
                                config=app.state.config,
                                text_units=app.state.text_units,                                
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        logger.error(f"基础搜索错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"基础搜索失败: {str(e)}")

@app.post("/basic_search")
async def basic_search_post(request: SearchRequest):
    return await basic_search_get(request.query)

@app.get("/status")
async def status():
    return JSONResponse(content={
        "status": "Server is up and running",
        "claim_extraction_enabled": CLAIM_EXTRACTION_ENABLED,
        "community_level": COMMUNITY_LEVEL,
        "response_type": RESPONSE_TYPE
    })

@app.get("/health")
async def health_check():
    try:
        # 检查关键数据是否加载
        data_status = {
            "entities_loaded": hasattr(app.state, 'entities') and len(app.state.entities) > 0,
            "communities_loaded": hasattr(app.state, 'communities') and len(app.state.communities) > 0,
            "text_units_loaded": hasattr(app.state, 'text_units') and len(app.state.text_units) > 0,
            "covariates_enabled": CLAIM_EXTRACTION_ENABLED,
            "covariates_loaded": app.state.covariates is not None if hasattr(app.state, 'covariates') else False
        }
        
        # 添加RAG服务状态
        rag_status = {
            "rag_available": rag_available,
            "rag_initialized": getattr(app.state, 'rag_initialized', False),
            "vector_store_initialized": vector_store.is_initialized() if rag_available and vector_store else False,
            "rag_config_valid": bool(rag_config.DASHSCOPE_API_KEY) if rag_available else False
        }
        
        return JSONResponse(content={
            "status": "healthy",
            "data_status": data_status,
            "rag_status": rag_status
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

# RAG服务路由
@app.get("/rag/")
async def rag_root():
    """RAG根路径"""
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG服务不可用")
    return {"message": "RAG System API", "version": "1.0.0"}

@app.get("/rag/health")
async def rag_health_check():
    """RAG健康检查"""
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG服务不可用")
    return {
        "status": "healthy",
        "vector_store_initialized": vector_store.is_initialized(),
        "config_valid": bool(rag_config.DASHSCOPE_API_KEY)
    }

@app.get("/rag/stats")
async def rag_get_stats():
    """获取RAG系统统计信息"""
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG服务不可用")
    try:
        vector_stats = vector_store.get_stats()
        return {
            "vector_store": vector_stats,
            "config": {
                "chunk_size": rag_config.CHUNK_SIZE,
                "chunk_overlap": rag_config.CHUNK_OVERLAP,
                "top_k": rag_config.TOP_K,
                "rerank_top_n": rag_config.RERANK_TOP_N,
                "embedding_model": rag_config.EMBEDDING_MODEL,
                "rerank_model": rag_config.RERANK_MODEL
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@app.post("/rag/index", response_model=IndexResponse)
async def rag_create_index(background_tasks: BackgroundTasks):
    """创建或重建RAG向量索引"""
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG服务不可用")
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
async def rag_search_documents(query: SearchQuery):
    """RAG搜索文档"""
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG服务不可用")
    
    start_time = time.time()
    
    try:
        # 检查向量存储是否已初始化
        if not vector_store.is_initialized():
            raise HTTPException(status_code=400, detail="向量索引未初始化，请先创建索引")
        
        logger.info(f"开始搜索查询: {query.query}")
        
        # 1. 向量化查询
        query_vector = await embedding_service.embed_query(query.query)
        
        # 2. 向量搜索
        top_k = query.top_k or rag_config.TOP_K
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
async def rag_delete_index():
    """删除RAG向量索引"""
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG服务不可用")
    try:
        vector_store.delete_index()
        return {"message": "向量索引已删除"}
    except Exception as e:
        logger.error(f"删除索引失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除索引失败: {str(e)}")

@app.post("/rag/query")
async def rag_graphrag_query(request: GraphRagQuery):
    """GraphRAG兼容接口 - 重定向到向量RAG搜索"""
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG服务不可用")
    try:
        # 使用默认参数进行向量搜索
        search_query = SearchQuery(
            query=request.query,
            top_k=rag_config.TOP_K,
            rerank_top_n=rag_config.RERANK_TOP_N,
            use_rerank=True
        )
        
        # 调用现有的搜索功能
        search_response = await rag_search_documents(search_query)
        
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

@app.post("/rag/ragquery", response_model=SearchResponse)
async def rag_vector_rag_query(query: SearchQuery):
    """向量RAG专用查询接口"""
    return await rag_search_documents(query)

@app.get("/rag/documents")
async def rag_list_documents():
    """列出所有RAG文档"""
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG服务不可用")
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
