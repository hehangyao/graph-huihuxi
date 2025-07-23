#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统启动脚本
提供便捷的启动选项和环境检查
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from rag_config import rag_config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'aiohttp',
        'aiofiles',
        'numpy',
        'mysql',
        'pymysql'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('.', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"缺少依赖包: {', '.join(missing_packages)}")
        logger.info("请运行以下命令安装依赖:")
        logger.info("pip install -r requirements.txt")
        return False
    
    logger.info("依赖检查通过")
    return True

def check_environment():
    """检查环境配置"""
    issues = []
    
    # 检查必需的环境变量
    if not config.DASHSCOPE_API_KEY or config.DASHSCOPE_API_KEY == "your_dashscope_api_key":
        issues.append("DASHSCOPE_API_KEY 未配置或使用默认值")
    
    if not config.DB_PASSWORD or config.DB_PASSWORD == "your_password":
        issues.append("DB_PASSWORD 未配置或使用默认值")
    
    # 检查目录是否存在
    required_dirs = [
        config.DOCUMENT_STORAGE_PATH,
        Path(config.INDEX_PATH).parent,
        Path(config.LOG_FILE).parent if config.LOG_FILE else None
    ]
    
    for dir_path in required_dirs:
        if dir_path and not Path(dir_path).exists():
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"创建目录: {dir_path}")
            except Exception as e:
                issues.append(f"无法创建目录 {dir_path}: {str(e)}")
    
    if issues:
        logger.warning("环境配置问题:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.info("请检查 .env 文件配置")
        return False
    
    logger.info("环境配置检查通过")
    return True

def test_database_connection():
    """测试数据库连接"""
    try:
        import mysql.connector
        
        connection = mysql.connector.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME
        )
        
        if connection.is_connected():
            logger.info("数据库连接测试成功")
            connection.close()
            return True
        else:
            logger.error("数据库连接失败")
            return False
            
    except Exception as e:
        logger.error(f"数据库连接测试失败: {str(e)}")
        logger.info("请检查数据库配置和服务状态")
        return False

def test_dashscope_connection():
    """测试DashScope API连接"""
    try:
        import asyncio
        from services.embedding_service import EmbeddingService
        
        async def test_api():
            service = EmbeddingService()
            return await service.test_connection()
        
        result = asyncio.run(test_api())
        if result:
            logger.info("DashScope API连接测试成功")
            return True
        else:
            logger.error("DashScope API连接测试失败")
            return False
            
    except Exception as e:
        logger.error(f"DashScope API连接测试失败: {str(e)}")
        logger.info("请检查API Key配置和网络连接")
        return False

def initialize_database():
    """初始化数据库"""
    try:
        import asyncio
        from database import DatabaseManager
        
        async def init_db():
            db_manager = DatabaseManager()
            await db_manager.initialize()
            await db_manager.close()
        
        asyncio.run(init_db())
        logger.info("数据库初始化完成")
        return True
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        return False

def start_server(host=None, port=None, reload=False, workers=1):
    """启动FastAPI服务器"""
    host = host or config.API_HOST
    port = port or config.API_PORT
    
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "main:app",
        "--host", str(host),
        "--port", str(port),
        "--workers", str(workers)
    ]
    
    if reload:
        cmd.append("--reload")
    
    logger.info(f"启动RAG服务器: http://{host}:{port}")
    logger.info(f"API文档: http://{host}:{port}/docs")
    logger.info(f"健康检查: http://{host}:{port}/health")
    
    try:
        subprocess.run(cmd, cwd=current_dir)
    except KeyboardInterrupt:
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"启动服务器失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="RAG系统启动脚本")
    parser.add_argument("--host", default=None, help="服务器主机地址")
    parser.add_argument("--port", type=int, default=None, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用自动重载（开发模式）")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数量")
    parser.add_argument("--skip-checks", action="store_true", help="跳过环境检查")
    parser.add_argument("--check-only", action="store_true", help="仅执行环境检查")
    parser.add_argument("--init-db", action="store_true", help="初始化数据库")
    
    args = parser.parse_args()
    
    logger.info("RAG系统启动脚本")
    logger.info("=" * 50)
    
    # 仅初始化数据库
    if args.init_db:
        logger.info("初始化数据库...")
        if initialize_database():
            logger.info("数据库初始化成功")
        else:
            logger.error("数据库初始化失败")
            sys.exit(1)
        return
    
    # 环境检查
    if not args.skip_checks:
        logger.info("执行环境检查...")
        
        checks = [
            ("依赖包检查", check_dependencies),
            ("环境配置检查", check_environment),
            ("数据库连接检查", test_database_connection),
            ("DashScope API检查", test_dashscope_connection)
        ]
        
        failed_checks = []
        for check_name, check_func in checks:
            logger.info(f"执行 {check_name}...")
            try:
                if not check_func():
                    failed_checks.append(check_name)
            except Exception as e:
                logger.error(f"{check_name} 执行失败: {str(e)}")
                failed_checks.append(check_name)
        
        if failed_checks:
            logger.error(f"以下检查失败: {', '.join(failed_checks)}")
            if not args.check_only:
                logger.warning("建议修复问题后再启动服务")
                response = input("是否继续启动服务? (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    sys.exit(1)
        else:
            logger.info("所有检查通过")
    
    # 仅执行检查
    if args.check_only:
        return
    
    # 初始化数据库
    logger.info("初始化数据库...")
    if not initialize_database():
        logger.error("数据库初始化失败，无法启动服务")
        sys.exit(1)
    
    # 启动服务器
    logger.info("启动RAG服务器...")
    start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )

if __name__ == "__main__":
    main()