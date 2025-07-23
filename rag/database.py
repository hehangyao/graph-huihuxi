import asyncio
import logging
from typing import Optional, Dict, Any, List
import sqlite3
import json
from datetime import datetime
from rag_config import rag_config as config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "rag.db"):
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        
    async def initialize(self):
        """初始化数据库连接和表结构"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            # 创建表结构
            await self._create_tables()
            logger.info(f"数据库初始化成功: {self.db_path}")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
            raise
    
    async def _create_tables(self):
        """创建数据库表结构"""
        cursor = self.connection.cursor()
        
        # 文档表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 文档块表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding_vector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        """)
        
        # 搜索历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                results_count INTEGER,
                search_time REAL,
                used_rerank BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON document_chunks (doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON document_chunks (chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_query ON search_history (query)")
        
        self.connection.commit()
        logger.info("数据库表结构创建完成")
    
    async def save_document(self, doc_id: str, filename: str, content: str, metadata: Dict[str, Any] = None):
        """保存文档"""
        cursor = self.connection.cursor()
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents (doc_id, filename, content, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, filename, content, metadata_json, datetime.now()))
        
        self.connection.commit()
        logger.debug(f"文档已保存: {doc_id}")
    
    async def save_document_chunk(self, chunk_id: str, doc_id: str, chunk_index: int, 
                                content: str, metadata: Dict[str, Any] = None, 
                                embedding_vector: List[float] = None):
        """保存文档块"""
        cursor = self.connection.cursor()
        metadata_json = json.dumps(metadata) if metadata else None
        embedding_json = json.dumps(embedding_vector) if embedding_vector else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO document_chunks 
            (chunk_id, doc_id, chunk_index, content, metadata, embedding_vector)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (chunk_id, doc_id, chunk_index, content, metadata_json, embedding_json))
        
        self.connection.commit()
        logger.debug(f"文档块已保存: {chunk_id}")
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        return None
    
    async def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """获取文档的所有块"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM document_chunks 
            WHERE doc_id = ? 
            ORDER BY chunk_index
        """, (doc_id,))
        
        chunks = []
        for row in cursor.fetchall():
            chunks.append({
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "embedding_vector": json.loads(row["embedding_vector"]) if row["embedding_vector"] else None,
                "created_at": row["created_at"]
            })
        
        return chunks
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT doc_id, filename, metadata, created_at, updated_at FROM documents")
        
        documents = []
        for row in cursor.fetchall():
            documents.append({
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            })
        
        return documents
    
    async def delete_document(self, doc_id: str):
        """删除文档及其所有块"""
        cursor = self.connection.cursor()
        
        # 删除文档块
        cursor.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
        
        # 删除文档
        cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        
        self.connection.commit()
        logger.info(f"文档已删除: {doc_id}")
    
    async def save_search_history(self, query: str, results_count: int, 
                                search_time: float, used_rerank: bool):
        """保存搜索历史"""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO search_history (query, results_count, search_time, used_rerank)
            VALUES (?, ?, ?, ?)
        """, (query, results_count, search_time, used_rerank))
        
        self.connection.commit()
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        cursor = self.connection.cursor()
        
        # 总搜索次数
        cursor.execute("SELECT COUNT(*) as total_searches FROM search_history")
        total_searches = cursor.fetchone()["total_searches"]
        
        # 平均搜索时间
        cursor.execute("SELECT AVG(search_time) as avg_search_time FROM search_history")
        avg_search_time = cursor.fetchone()["avg_search_time"] or 0
        
        # 重排序使用率
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN used_rerank = 1 THEN 1 END) as rerank_used,
                COUNT(*) as total
            FROM search_history
        """)
        rerank_stats = cursor.fetchone()
        rerank_usage_rate = (rerank_stats["rerank_used"] / rerank_stats["total"]) * 100 if rerank_stats["total"] > 0 else 0
        
        return {
            "total_searches": total_searches,
            "avg_search_time": round(avg_search_time, 3),
            "rerank_usage_rate": round(rerank_usage_rate, 2)
        }
    
    async def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("数据库连接已关闭")

# 全局数据库管理器实例
db_manager = DatabaseManager()