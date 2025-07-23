# GraphRAG医疗知识库技术栈

## 核心技术

### GraphRAG框架
- **主框架**: Microsoft GraphRAG - 结合图数据结构与检索增强生成的先进框架
- **版本**: 最新稳定版（0.3.x+）
- **特点**: 支持大规模文档处理、知识图谱构建、语义检索和智能问答

### 医学自然语言处理
- **医学实体识别**: BioBERT, ClinicalBERT
- **医学关系抽取**: SciBERT + 医学关系分类器
- **医学文本嵌入**: BioSentVec, ClinicalBERT embeddings
- **中文医学NLP**: Chinese-BERT-wwm + 医学领域微调

## 构建系统

### 环境管理
```bash
# Python环境
Python 3.9+
uv (包管理器)

# 虚拟环境创建
uv venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# GraphRAG安装
uv add graphrag
uv add python-docx pymupdf  # 文档处理
uv add neo4j milvus-pymilvus  # 数据库
```

### 配置文件
- **pyproject.toml**: Python依赖管理
- **.env**: 环境变量（API密钥等）
- **settings.yaml**: GraphRAG配置文件
- **docker-compose.yml**: 容器编排配置

## 常用命令

### GraphRAG基础操作
```bash
# 初始化GraphRAG工作区
python -m graphrag.index --init --root ./medical-kb

# 配置API密钥
export GRAPHRAG_API_KEY="your-api-key"

# 运行文档索引
python -m graphrag.index --root ./medical-kb

# 执行查询
python -m graphrag.query --root ./medical-kb \
    --method global \
    --query "OSAS的治疗方案有哪些？"

# 局部搜索
python -m graphrag.query --root ./medical-kb \
    --method local \
    --query "呼吸机故障代码E02"
```

### 文档处理
```bash
# 批量处理医疗文档
python scripts/process_medical_docs.py \
    --input input/ \
    --output processed/ \
    --doc-types docx,pdf

# 医学实体识别
python scripts/extract_medical_entities.py \
    --input processed/ \
    --model biobert \
    --output entities/
```

### 开发环境
```bash
# 启动开发服务器
uvicorn medical_kb.api:app --reload --port 8000

# 运行测试
python -m pytest tests/ -v

# 构建Docker镜像
docker build -t medical-graphrag:latest .

# 启动完整服务栈
docker-compose up -d
```

### 生产部署
```bash
# 构建生产镜像
docker build -f Dockerfile.prod -t medical-graphrag:prod .

# 数据备份
docker exec graphrag-neo4j neo4j-admin backup \
    --backup-dir=/backups/$(date +%Y%m%d)

# 监控检查
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/status
```

## 库和框架

### 核心依赖
- **graphrag**: Microsoft GraphRAG框架
- **fastapi**: 高性能Web API框架
- **neo4j**: 图数据库Python驱动
- **pymilvus**: 向量数据库Python客户端
- **transformers**: Hugging Face模型库
- **sentence-transformers**: 文本嵌入模型

### 医学NLP库
- **scispacy**: 科学/医学文本处理
- **biobert**: 生物医学BERT模型
- **medspacy**: 医学文本处理扩展
- **clinical-bert**: 临床文本BERT模型

### 文档处理
- **python-docx**: DOCX文档解析
- **pymupdf**: PDF文档处理
- **pandas**: 数据处理和分析
- **openpyxl**: Excel文件处理

### 部署运维
- **docker**: 容器化技术
- **docker-compose**: 多容器编排
- **gunicorn**: WSGI服务器
- **nginx**: 反向代理服务器
- **prometheus**: 监控和指标收集
- **grafana**: 监控数据可视化

## 代码风格和规范

### Python代码规范
```bash
# 代码格式化
uv add black isort
black medical_kb/
isort medical_kb/

# 代码检查
uv add flake8 mypy
flake8 medical_kb/
mypy medical_kb/

# 预提交钩子
uv add pre-commit
pre-commit install
```

### 提交规范
- **feat**: 新功能 (如: feat: 添加医学实体识别功能)
- **fix**: 错误修复 (如: fix: 修复GraphRAG查询超时问题)
- **docs**: 文档更新 (如: docs: 更新API使用说明)
- **refactor**: 代码重构 (如: refactor: 优化医疗问答生成逻辑)
- **test**: 测试相关 (如: test: 添加医学准确性测试用例)

## 版本控制

### Git工作流
```bash
# 创建功能分支
git checkout -b feature/medical-entity-extraction

# 提交代码
git add .
git commit -m "feat: 实现医学实体识别功能"

# 推送分支
git push origin feature/medical-entity-extraction

# 合并到主分支
git checkout main
git merge feature/medical-entity-extraction
```

### 分支策略
- **main**: 生产环境代码
- **develop**: 开发环境代码
- **feature/***: 新功能开发分支
- **hotfix/***: 紧急修复分支

## 测试策略

### 测试框架和工具
```bash
# 安装测试依赖
uv add pytest pytest-asyncio pytest-cov

# 运行所有测试
python -m pytest

# 生成覆盖率报告
python -m pytest --cov=medical_kb --cov-report=html

# 运行特定测试
python -m pytest tests/test_medical_entities.py -v
```

### 测试类型
- **单元测试**: 测试单个函数和类的功能
- **集成测试**: 测试组件间的协作
- **端到端测试**: 测试完整的用户场景
- **性能测试**: 测试系统响应时间和吞吐量
- **医学准确性测试**: 验证医学信息的正确性

### 测试覆盖率目标
- **核心功能**: > 90%
- **API接口**: > 85% 
- **医学模块**: > 95%
- **整体覆盖率**: > 80%

## 数据库配置

### Neo4j图数据库
```yaml
# docker-compose.yml
neo4j:
  image: neo4j:5.0
  environment:
    NEO4J_AUTH: neo4j/medical-kb-2024
    NEO4J_PLUGINS: '["apoc"]'
  ports:
    - "7474:7474"
    - "7687:7687"
  volumes:
    - neo4j_data:/data
```

### Milvus向量数据库
```yaml
milvus:
  image: milvusdb/milvus:latest
  environment:
    ETCD_ENDPOINTS: etcd:2379
    MINIO_ADDRESS: minio:9000
  ports:
    - "19530:19530"
  volumes:
    - milvus_data:/var/lib/milvus
```

## 环境变量配置

### .env文件示例
```bash
# GraphRAG配置
GRAPHRAG_API_KEY=your-openai-api-key
GRAPHRAG_API_BASE=https://api.openai.com/v1
GRAPHRAG_MODEL=gpt-4

# 数据库连接
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=medical-kb-2024

MILVUS_HOST=localhost
MILVUS_PORT=19530

# 医学模型配置
BIOBERT_MODEL_PATH=./models/biobert
CLINICAL_BERT_MODEL_PATH=./models/clinical-bert

# 日志和监控
LOG_LEVEL=INFO
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

## 性能优化

### 缓存策略
- **Redis**: API响应缓存
- **内存缓存**: 模型和嵌入缓存
- **图缓存**: Neo4j查询结果缓存

### 并发处理
- **异步处理**: FastAPI + asyncio
- **队列系统**: Celery + Redis
- **批处理**: 文档和查询批量处理

### 监控指标
- **响应时间**: API请求响应时间
- **准确率**: 医学信息准确率
- **吞吐量**: 每秒处理查询数
- **资源使用**: CPU、内存、存储使用率

## 安全配置

### API安全
- **认证**: JWT token认证
- **授权**: 基于角色的访问控制
- **限流**: 请求频率限制
- **HTTPS**: SSL/TLS加密传输

### 数据安全
- **加密**: 敏感数据加密存储
- **脱敏**: 医疗数据匿名化
- **审计**: 操作日志记录
- **备份**: 定期数据备份

## 医疗合规

### 医疗标准
- **ICD-10**: 国际疾病分类编码
- **SNOMED CT**: 系统化医学术语
- **UMLS**: 统一医学语言系统
- **HL7 FHIR**: 医疗数据交换标准

### 质量保证
- **专家评审**: 医学专家内容审核
- **版本控制**: 医学指南版本管理
- **更新机制**: 定期内容更新
- **免责声明**: 医疗建议免责条款