#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试RAG服务修复后的功能
临时脚本，用完即删
"""

import requests
import json
import time

def test_rag_services():
    """测试RAG服务功能"""
    base_url = "http://localhost:8000"
    
    print("=== 测试RAG服务修复 ===")
    print(f"基础URL: {base_url}")
    print()
    
    # 测试基础健康检查
    print("1. 测试基础健康检查")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   健康检查: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            rag_status = health_data.get('rag_status', {})
            print(f"   RAG可用性: {rag_status.get('rag_available', False)}")
            print(f"   RAG初始化: {rag_status.get('rag_initialized', False)}")
            print(f"   向量存储: {rag_status.get('vector_store_initialized', False)}")
            print(f"   配置有效: {rag_status.get('rag_config_valid', False)}")
    except Exception as e:
        print(f"   健康检查失败: {e}")
    print()
    
    # 测试RAG根路径
    print("2. 测试RAG根路径")
    try:
        response = requests.get(f"{base_url}/rag/")
        print(f"   RAG根路径 (/rag/): {response.status_code}")
        if response.status_code == 200:
            print(f"   响应: {response.json()}")
        elif response.status_code == 503:
            print("   RAG服务仍不可用")
    except Exception as e:
        print(f"   RAG根路径测试失败: {e}")
    print()
    
    # 测试RAG健康检查
    print("3. 测试RAG健康检查")
    try:
        response = requests.get(f"{base_url}/rag/health")
        print(f"   RAG健康检查 (/rag/health): {response.status_code}")
        if response.status_code == 200:
            print(f"   响应: {response.json()}")
        elif response.status_code == 503:
            print("   RAG服务仍不可用")
    except Exception as e:
        print(f"   RAG健康检查失败: {e}")
    print()
    
    # 测试RAG统计信息
    print("4. 测试RAG统计信息")
    try:
        response = requests.get(f"{base_url}/rag/stats")
        print(f"   RAG统计 (/rag/stats): {response.status_code}")
        if response.status_code == 200:
            print(f"   响应: {response.json()}")
        elif response.status_code == 503:
            print("   RAG服务仍不可用")
    except Exception as e:
        print(f"   RAG统计测试失败: {e}")
    print()
    
    print("=== 测试完成 ===")
    print()
    
    # 总结
    print("总结:")
    print("- 如果RAG接口返回200状态码，说明修复成功")
    print("- 如果仍返回503状态码，说明还有其他问题需要解决")
    print("- GraphRAG服务应该继续正常工作")

if __name__ == "__main__":
    test_rag_services()