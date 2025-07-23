import os
from typing import Optional

class Config:
    """RAG系统配置类"""
    
    def __init__(self):
        # API配置
        self.API_HOST = os.getenv("API_HOST", "localhost")
        self.API_PORT = int(os.getenv("API_PORT", "8001"))
        
        # DashScope配置
        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
        self.DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1")
        
        # 嵌入模型配置
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        self.EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        
        # 重排序模型配置
        self.RERANK_MODEL = os.getenv("RERANK_MODEL", "gte-rerank")
        
        # 文档处理配置
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # 搜索配置
        self.TOP_K = int(os.getenv("TOP_K", "5"))
        self.RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "2"))
        self.ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
        
        # 数据库配置
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///rag.db")
        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = int(os.getenv("DB_PORT", "3306"))
        self.DB_USER = os.getenv("DB_USER", "root")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "")
        self.DB_NAME = os.getenv("DB_NAME", "rag_db")
        
        # 文档存储路径
        self.DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "documents")
        self.DOCUMENT_STORAGE_PATH = os.getenv("DOCUMENT_STORAGE_PATH", "./rag/data/documents")
        self.TEMP_STORAGE_PATH = os.getenv("TEMP_STORAGE_PATH", "./data/temp")
        
        # 向量索引存储路径
        self.INDEX_PATH = os.getenv("INDEX_PATH", "index")
        self.VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "./data/vector_index")
        
        # 日志配置
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE_PATH", "./logs/rag.log")
        self.LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", "10485760"))
        self.LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        
    def validate(self):
        """验证配置"""
        if not self.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY环境变量未设置")
        
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE必须大于0")
            
        if self.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP不能小于0")
            
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP不能大于等于CHUNK_SIZE")
            
        if self.TOP_K <= 0:
            raise ValueError("TOP_K必须大于0")
            
        if self.RERANK_TOP_N <= 0:
            raise ValueError("RERANK_TOP_N必须大于0")
            
        if self.RERANK_TOP_N > self.TOP_K:
            raise ValueError("RERANK_TOP_N不能大于TOP_K")

# 全局配置实例
rag_config = Config()