# RAG (Retrieval-Augmented Generation) 系统

基于 FastAPI 和阿里云 DashScope 的向量检索增强生成系统。

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  向量存储        │    │  阿里云DashScope │
│   Web服务       │◄──►│  VectorStore    │◄──►│  嵌入模型        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档处理       │    │   MySQL         │    │   重排序服务     │
│   DocumentProc  │    │   数据库        │    │   RerankService │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 核心功能

- **文档管理**: 支持 TXT、PDF、DOCX、Markdown 等格式
- **向量检索**: 基于语义相似度的文档搜索
- **智能分块**: 自动将长文档分割为合适的块
- **重排序**: 提升搜索结果的相关性
- **RESTful API**: 完整的 HTTP 接口
- **数据持久化**: MySQL 数据库存储

## 安装依赖

### 方式一：使用 pip
```bash
cd rag
pip install -r requirements.txt
```

### 方式二：使用项目根目录的 pyproject.toml
```bash
cd ..
pip install -e .
```

## 配置

### 1. 环境变量配置

创建 `.env` 文件：
```bash
# 阿里云 DashScope API
DASHSCOPE_API_KEY=your_dashscope_api_key

# 数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=huihuxi_rag

# 可选配置
EMBEDDING_MODEL=text-embedding-v3
RERANK_MODEL=gte-rerank
CHUNK_SIZE=500
CHUNK_OVERLAP=50
SIMILARITY_THRESHOLD=0.3
ENABLE_RERANK=true
```

### 2. 数据库准备

确保 MySQL 服务运行，并创建数据库：
```sql
CREATE DATABASE huihuxi_rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

## 启动服务

```bash
cd rag
python main.py
```

或使用 uvicorn：
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## API 接口

### 1. 创建索引
```http
POST /api/v1/index/create
Content-Type: multipart/form-data

file: [文件]
doc_id: [可选，文档ID]
```

### 2. 搜索文档
```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "搜索关键词",
  "top_k": 10,
  "similarity_threshold": 0.3,
  "enable_rerank": true
}
```

### 3. 列出文档
```http
GET /api/v1/documents?limit=20&offset=0
```

### 4. 删除文档
```http
DELETE /api/v1/documents/{doc_id}
```

### 5. 获取统计信息
```http
GET /api/v1/stats
```

### 6. 删除索引
```http
DELETE /api/v1/index
```

## 使用示例

### Python 客户端示例

```python
import requests
import json

# 服务地址
BASE_URL = "http://localhost:8000/api/v1"

# 1. 上传文档
with open("document.txt", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/index/create", files=files)
    print("上传结果:", response.json())

# 2. 搜索文档
search_data = {
    "query": "睡眠呼吸暂停治疗",
    "top_k": 5,
    "enable_rerank": True
}
response = requests.post(
    f"{BASE_URL}/search", 
    headers={"Content-Type": "application/json"},
    data=json.dumps(search_data)
)
print("搜索结果:", response.json())

# 3. 获取文档列表
response = requests.get(f"{BASE_URL}/documents")
print("文档列表:", response.json())
```

### curl 示例

```bash
# 上传文档
curl -X POST "http://localhost:8000/api/v1/index/create" \
  -F "file=@document.txt"

# 搜索文档
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "睡眠呼吸暂停治疗",
    "top_k": 5
  }'

# 获取文档列表
curl "http://localhost:8000/api/v1/documents"
```

## 目录结构

```
rag/
├── main.py                 # FastAPI 应用入口
├── config.py              # 配置管理
├── database.py            # 数据库操作
├── utils.py               # 工具函数
├── requirements.txt       # Python 依赖
├── README.md             # 说明文档
├── .env.example          # 环境变量示例
└── services/             # 核心服务
    ├── __init__.py
    ├── document_processor.py  # 文档处理
    ├── embedding_service.py   # 嵌入服务
    ├── vector_store.py       # 向量存储
    ├── rerank_service.py     # 重排序服务
    └── rag_service.py        # RAG 核心服务
```

## 核心组件说明

### DocumentProcessor
- 支持多种文档格式解析
- 智能文本分块
- Token 数量估算
- 文档元数据提取

### EmbeddingService
- 阿里云 DashScope 集成
- 批量向量化处理
- 连接池管理
- 错误重试机制

### VectorStore
- 内存向量索引
- 余弦相似度计算
- 索引持久化
- 增量更新支持

### RerankService
- 搜索结果重排序
- 多种相关性算法
- 批量处理支持
- 性能优化

### RAGService
- 统一的服务接口
- 完整的文档生命周期管理
- 搜索历史记录
- 系统统计信息

## 性能优化

1. **批量处理**: 支持批量文档上传和向量化
2. **连接池**: HTTP 连接复用，减少延迟
3. **缓存机制**: 向量索引内存缓存
4. **异步处理**: 全异步 I/O 操作
5. **分块策略**: 智能文档分块，平衡检索精度和性能

## 监控和日志

- 详细的操作日志
- 性能指标统计
- 错误追踪和报告
- 搜索历史记录

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查数据库服务是否运行
   - 验证连接参数
   - 确认数据库权限

2. **DashScope API 错误**
   - 验证 API Key 是否正确
   - 检查网络连接
   - 确认 API 配额

3. **文档上传失败**
   - 检查文件格式是否支持
   - 验证文件大小限制
   - 确认磁盘空间

4. **搜索结果为空**
   - 检查索引是否创建
   - 调整相似度阈值
   - 验证查询内容

### 日志级别

```python
# 在 config.py 中调整日志级别
LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

## 扩展开发

### 添加新的文档格式支持

在 `DocumentProcessor` 中添加新的解析器：

```python
def _extract_text_from_custom_format(self, file_path: str) -> str:
    # 实现自定义格式解析
    pass
```

### 集成其他嵌入模型

继承 `EmbeddingService` 并实现新的 API 接口：

```python
class CustomEmbeddingService(EmbeddingService):
    async def get_embedding(self, text: str) -> List[float]:
        # 实现自定义嵌入服务
        pass
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！