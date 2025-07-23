import requests
import json

def test_rag_query():
    """测试RAG查询功能"""
    url = "http://localhost:8000/rag/ragquery"
    
    # 测试查询
    query_data = {
        "query": "呼吸机异响的定义",
        "top_k": 200
    }
    
    try:
        print(f"正在测试RAG查询: {query_data['query']}")
        print(f"请求的top_k: {query_data['top_k']}")
        
        response = requests.post(url, json=query_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n查询成功!")
            print(f"查询: {result.get('query', '')}")
            print(f"返回结果数量: {len(result.get('results', []))}")
            print(f"总找到结果: {result.get('total_found', 0)}")
            print(f"搜索时间: {result.get('search_time', 0)}秒")
            print(f"使用重排序: {result.get('used_rerank', False)}")
            
            # 显示前3个结果的摘要
            results = result.get('results', [])
            if results:
                print("\n前3个结果摘要:")
                for i, res in enumerate(results[:3]):
                    print(f"  {i+1}. 相似度: {res.get('similarity', 0):.4f}")
                    print(f"     内容: {res.get('content', '')[:100]}...")
                    print()
        else:
            print(f"查询失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    test_rag_query()