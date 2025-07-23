# GraphRAG重新索引操作指南

## 概述
本文档记录了在Windows环境下使用uv包管理器重新索引GraphRAG的完整流程，适用于医疗知识库项目。

## 环境信息
- **操作系统**: Windows 10/11
- **Python版本**: 3.11.11
- **包管理器**: uv
- **GraphRAG版本**: 2.2.1
- **项目路径**: `D:\develop\ai\graph-huihuxi\ragtest`

## 前置条件
- 已安装uv包管理器
- 已配置API密钥（阿里云通义千问）
- 医疗文档已预处理完成

## 操作步骤

### 1. 清理环境
```powershell
# 进入项目目录
cd D:\develop\ai\graph-huihuxi\ragtest

# 删除旧的虚拟环境（如果存在权限问题）
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue

# 清理旧的索引输出
Remove-Item -Recurse -Force output -ErrorAction SilentlyContinue
```

### 2. 创建虚拟环境
```powershell
# 创建新的虚拟环境
uv venv

# 输出示例：
# Using CPython 3.11.11
# Activate with: .venv\Scripts\activate
```

### 3. 安装依赖
```powershell
# 安装GraphRAG及相关依赖
uv add graphrag

# 这将安装以下主要包：
# - graphrag~=2.2.1
# - fastapi~=0.112.4
# - pandas>=2.3.1
# - tiktoken>=0.9.0
# - 以及其他139个依赖包
```

### 4. 配置文件检查

#### 4.1 检查.env文件
```bash
# 文件路径: ragtest/.env
GRAPHRAG_API_KEY=sk-2a92cf1c6d7940e5a91ae51954bf9be2
```

#### 4.2 修复settings.yaml配置
**问题**: `file_pattern`中的`$`符号被误认为环境变量占位符

**解决方案**:
```yaml
# 修改前（有问题）
file_pattern: ".*\\.txt$"

# 修改后（正确）
file_pattern: ".*\\.txt"
```

**完整配置示例**:
```yaml
input:
  type: file
  file_type: text
  base_dir: "input_cleaned"
  file_encoding: utf-8
  file_pattern: ".*\\.txt"

models:
  default_chat_model:
    type: openai_chat
    api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: ${GRAPHRAG_API_KEY}
    model: qwen-plus
    encoding_model: cl100k_base
    concurrent_requests: 5
    
  default_embedding_model:
    type: openai_embedding
    api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: ${GRAPHRAG_API_KEY}
    model: text-embedding-v4
    encoding_model: cl100k_base
    concurrent_requests: 3
```

### 5. 执行重新索引
```powershell
# 运行GraphRAG索引
uv run graphrag index --root .

# 预期输出：
# Logging enabled at D:\develop\ai\graph-huihuxi\ragtest\logs\indexing-engine.log
# 🚀 LLM Config Params Validated
# 🚀 Embedding LLM Config Params Validated
# Running standard indexing.
# 🚀 create_base_text_units
```

### 6. 验证索引结果
```powershell
# 检查输出目录结构
ls output/

# 预期生成的文件：
# - artifacts/
# - reports/
# - 各种parquet文件
```

## 常见问题及解决方案

### 问题1: 虚拟环境创建失败
**错误**: `Failed to create virtualenv: 拒绝访问。 (os error 5)`

**解决方案**:
```powershell
# 删除残留的.venv文件夹
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
# 重新创建
uv venv
```

### 问题2: 环境变量占位符错误
**错误**: `ValueError: Invalid placeholder in string: line 52, col 26`

**解决方案**: 检查settings.yaml中的正则表达式，移除或转义`$`符号

### 问题3: 模块导入错误
**错误**: `No module named graphrag.index.__main__`

**解决方案**: 使用正确的命令格式：
```powershell
# 错误命令
uv run python -m graphrag.index --root .

# 正确命令
uv run graphrag index --root .
```

## 文件结构说明

### 输入文件 (input_cleaned/)
```
input_cleaned/
├── 三种治疗方案的介绍.txt (7.7KB)
├── 呼吸机使用常见问题AI解答.txt (3.9KB)
├── 成人阻塞性睡眠呼吸暂停低通气综合征无创通气治疗最佳总结.txt (32KB)
├── *.json (元数据文件)
└── preprocessing_summary.json
```

### 输出文件 (output/)
```
output/
├── artifacts/ (GraphRAG生成的知识图谱文件)
├── reports/ (索引报告)
└── logs/ (日志文件)
```

## 性能优化建议

1. **并发请求限制**: 
   - Chat模型: 5个并发请求
   - Embedding模型: 3个并发请求

2. **速率限制**:
   - Chat模型: 60请求/分钟
   - Embedding模型: 30请求/分钟

3. **重试策略**: 
   - 最大重试次数: 3次
   - 重试策略: native

## 后续操作

### 测试索引结果
```powershell
# 全局搜索测试
uv run graphrag query --root . --method global "OSAS的治疗方案有哪些？"

# 局部搜索测试
uv run graphrag query --root . --method local "呼吸机故障代码E02"
```

### 启动API服务
```powershell
# 返回项目根目录
cd ..

# 启动GraphRAG API服务
python api.py
```

## 注意事项

1. **API密钥安全**: 确保.env文件不被提交到版本控制系统
2. **磁盘空间**: 索引过程会生成大量临时文件，确保有足够磁盘空间
3. **网络连接**: 确保能够正常访问阿里云API服务
4. **日志监控**: 关注logs/indexing-engine.log中的错误信息

## 版本信息
- **文档版本**: 1.0
- **最后更新**: 2025-01-27
- **适用GraphRAG版本**: 2.2.1+
- **测试环境**: Windows 10 + Python 3.11.11 + uv 