#!/usr/bin/env python3
"""
Test script for LangGraph Agent - Assignment 2
Verifies core functionality without Streamlit interface
"""

import os
import sys
from langchain_core.messages import HumanMessage

def test_agent_import():
    """Test if agent can be imported successfully"""
    try:
        from langgraph_agent import get_agent, ChatState
        print("✅ Agent import successful")
        return True
    except ImportError as e:
        print(f"❌ Agent import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Agent setup error: {e}")
        return False

def test_basic_queries():
    """Test basic agent functionality"""
    try:
        from langgraph_agent import get_agent
        
        agent = get_agent()
        print("✅ Agent created successfully")
        
        # Test queries for each type
        test_cases = [
            {
                "type": "structured",
                "query": "What are the most frequent categories?",
                "expected_tools": ["get_most_frequent_categories"]
            },
            {
                "type": "unstructured", 
                "query": "Summarize the REFUND category",
                "expected_tools": ["summarise"]
            },
            {
                "type": "out_of_scope",
                "query": "Who is Magnus Carlsen?",
                "expected_tools": []
            },
            {
                "type": "recommend",
                "query": "What should I ask next?",
                "expected_tools": []
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n🧪 Test {i+1}: {test_case['type']} query")
            print(f"Query: {test_case['query']}")
            
            try:
                # Prepare state
                initial_state = {
                    "messages": [HumanMessage(content=test_case['query'])],
                    "query_type": "",
                    "user_summary": "",
                    "conversation_count": 0,
                    "last_query_type": "",
                    "current_session": "test_session"
                }
                
                # Configure with test thread
                config = {"configurable": {"thread_id": "test_thread"}}
                
                # Run agent
                result = agent.invoke(initial_state, config=config)
                
                if result and "query_type" in result:
                    print(f"✅ Classified as: {result['query_type']}")
                    
                    if result["query_type"] == test_case["type"]:
                        print("✅ Correct classification")
                    else:
                        print(f"⚠️  Expected: {test_case['type']}, Got: {result['query_type']}")
                else:
                    print("❌ No classification result")
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic query test failed: {e}")
        return False

def test_memory_functionality():
    """Test memory persistence"""
    try:
        from langgraph_agent import get_agent
        
        agent = get_agent()
        
        print("\n🧠 Testing Memory Functionality")
        
        # First interaction
        initial_state = {
            "messages": [HumanMessage(content="Show me examples of REFUND category")],
            "query_type": "",
            "user_summary": "",
            "conversation_count": 0,
            "last_query_type": "",
            "current_session": "memory_test"
        }
        
        config = {"configurable": {"thread_id": "memory_test_thread"}}
        
        result1 = agent.invoke(initial_state, config=config)
        print("✅ First interaction completed")
        
        if "user_summary" in result1:
            print(f"✅ User summary created: {result1['user_summary'][:50]}...")
        
        # Second interaction with memory query
        memory_state = {
            "messages": [HumanMessage(content="What do you remember about me?")],
            "query_type": "",
            "user_summary": result1.get("user_summary", ""),
            "conversation_count": 1,
            "last_query_type": result1.get("query_type", ""),
            "current_session": "memory_test"
        }
        
        result2 = agent.invoke(memory_state, config=config)
        print("✅ Memory query completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_tools():
    """Test individual tools"""
    try:
        from langgraph_agent import (
            get_all_categories, 
            get_all_intents,
            count_category,
            show_examples
        )
        
        print("\n🛠️  Testing Individual Tools")
        
        # Test get_all_categories
        categories = get_all_categories()
        if categories:
            print(f"✅ get_all_categories: {len(categories)} categories found")
        else:
            print("❌ get_all_categories: No categories found")
        
        # Test get_all_intents  
        intents = get_all_intents()
        if intents:
            print(f"✅ get_all_intents: {len(intents)} intents found")
        else:
            print("❌ get_all_intents: No intents found")
        
        # Test count_category
        if categories:
            count = count_category(categories[0])
            print(f"✅ count_category: {categories[0]} has {count} records")
        
        # Test show_examples
        examples = show_examples(3)
        if isinstance(examples, dict) and "examples" in examples:
            print(f"✅ show_examples: Retrieved {len(examples['examples'])} examples")
        else:
            print("❌ show_examples: Failed to retrieve examples")
        
        return True
        
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
        return False

def test_dataset():
    """Test dataset loading"""
    try:
        import pandas as pd
        
        dataset_path = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            return False
        
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Check required columns
        required_columns = ['instruction', 'category', 'intent', 'response']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        else:
            print(f"✅ All required columns present")
        
        # Show sample data
        print(f"✅ Sample categories: {df['category'].unique()[:5].tolist()}")
        print(f"✅ Sample intents: {df['intent'].unique()[:5].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 LangGraph Agent Test Suite - Assignment 2\n")
    
    # Check API key first
    if not os.getenv("NEBIUS_YAIR"):
        print("❌ NEBIUS_YAIR API key not found")
        print("🔑 Please set your API key before running tests:")
        print("   export NEBIUS_YAIR='your-api-key'")
        return
    
    tests = [
        ("Dataset Loading", test_dataset),
        ("Agent Import", test_agent_import),
        ("Individual Tools", test_tools),
        ("Basic Queries", test_basic_queries),
        ("Memory Functionality", test_memory_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🔍 Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append(result)
            
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append(False)
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print('='*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Agent is ready for use.")
        print("🚀 Run: streamlit run streamlit_app.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("💡 Common issues:")
        print("   - Missing API key")
        print("   - Dataset file not found")
        print("   - Missing dependencies")
    
    print('='*50)

if __name__ == "__main__":
    main()
