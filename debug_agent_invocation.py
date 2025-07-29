#!/usr/bin/env python3
"""
Debug script to identify where the agent invocation is failing
"""

from langgraph_agent import get_agent, ChatState
from langchain_core.messages import HumanMessage, AIMessage

def test_agent_step_by_step():
    """Test agent invocation step by step to isolate the error"""
    
    print("🔍 Debugging Agent Invocation...")
    print("="*50)
    
    # Step 1: Test agent creation
    print("\n1️⃣ Testing agent creation...")
    try:
        agent = get_agent()
        print("✅ Agent created successfully")
        print(f"   Agent type: {type(agent)}")
        print(f"   Has checkpointer: {hasattr(agent, 'checkpointer')}")
        if hasattr(agent, 'checkpointer'):
            print(f"   Checkpointer type: {type(agent.checkpointer)}")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return
    
    # Step 2: Test simple state creation
    print("\n2️⃣ Testing state creation...")
    try:
        test_state = {
            "messages": [HumanMessage(content="What are the most frequent categories?")],
            "query_type": "",
            "user_summary": "",
            "conversation_count": 0,
            "last_query_type": "",
            "current_session": "test_session"
        }
        print("✅ State created successfully")
        print(f"   State keys: {list(test_state.keys())}")
        print(f"   Message type: {type(test_state['messages'][0])}")
    except Exception as e:
        print(f"❌ State creation failed: {e}")
        return
    
    # Step 3: Test config creation
    print("\n3️⃣ Testing config creation...")
    try:
        config = {"configurable": {"thread_id": "test_thread"}}
        print("✅ Config created successfully")
        print(f"   Config: {config}")
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return
    
    # Step 4: Test agent invocation without checkpointer
    print("\n4️⃣ Testing agent invocation WITHOUT checkpointer...")
    try:
        # Create agent without checkpointer
        from langgraph_agent import create_graph
        simple_agent = create_graph().compile()
        
        result = simple_agent.invoke(test_state)
        print("✅ Simple agent invocation works")
        print(f"   Result type: {type(result)}")
        print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"❌ Simple agent invocation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Test agent invocation with checkpointer
    print("\n5️⃣ Testing agent invocation WITH checkpointer...")
    try:
        result = agent.invoke(test_state, config=config)
        print("✅ Checkpointed agent invocation works")
        print(f"   Result type: {type(result)}")
        print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"❌ Checkpointed agent invocation failed: {e}")
        print(f"   Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Test individual node functions
    print("\n6️⃣ Testing individual node functions...")
    try:
        from langgraph_agent import classify_query, structured_agent
        
        # Test classify_query
        classify_result = classify_query(test_state)
        print("✅ classify_query works")
        print(f"   Query type: {classify_result.get('query_type', 'None')}")
        
        # Test structured_agent
        test_state["query_type"] = "structured"
        structured_result = structured_agent(test_state)
        print("✅ structured_agent works")
        
    except Exception as e:
        print(f"❌ Node function testing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("🎯 Analysis Complete!")

def test_minimal_working_example():
    """Test with minimal working agent"""
    
    print("\n🧪 Testing Minimal Working Example...")
    print("="*50)
    
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.sqlite import SqliteSaver
        from typing import TypedDict, List
        from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
        
        # Simple state
        class MinimalState(TypedDict):
            messages: List[BaseMessage]
            query_type: str
        
        def simple_classifier(state):
            print(f"   Classifier received: {len(state['messages'])} messages")
            return {"query_type": "test"}
        
        def simple_responder(state):
            print(f"   Responder received: {state['query_type']}")
            return {"messages": [AIMessage(content="Test response")]}
        
        # Create workflow
        workflow = StateGraph(MinimalState)
        workflow.add_node("classify", simple_classifier)
        workflow.add_node("respond", simple_responder)
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "respond")
        workflow.add_edge("respond", END)
        
        # Test without checkpointer
        simple_agent = workflow.compile()
        test_state = {
            "messages": [HumanMessage(content="test")],
            "query_type": ""
        }
        
        result1 = simple_agent.invoke(test_state)
        print("✅ Minimal agent without checkpointer works")
        
        # Test with checkpointer
        memory_saver = SqliteSaver.from_conn_string(":memory:")
        checkpointed_agent = workflow.compile(checkpointer=memory_saver)
        
        config = {"configurable": {"thread_id": "minimal_test"}}
        result2 = checkpointed_agent.invoke(test_state, config=config)
        print("✅ Minimal agent with checkpointer works")
        
    except Exception as e:
        print(f"❌ Minimal example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_step_by_step()
    test_minimal_working_example()
