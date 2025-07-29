#!/usr/bin/env python3
"""
Test script to identify working checkpointer method
"""

def test_checkpointers():
    """Test different checkpointer options"""
    
    print("🔍 Testing Checkpointer Options...")
    print("="*50)
    
    # Test 1: SqliteSaver from_conn_string
    print("\n1️⃣ Testing SqliteSaver.from_conn_string...")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        memory_saver = SqliteSaver.from_conn_string(":memory:")
        print("✅ SqliteSaver.from_conn_string works")
    except Exception as e:
        print(f"❌ SqliteSaver.from_conn_string failed: {e}")
    
    # Test 2: SqliteSaver with manual connection
    print("\n2️⃣ Testing SqliteSaver with manual connection...")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        memory_saver = SqliteSaver(conn)
        print("✅ SqliteSaver with manual connection works")
    except Exception as e:
        print(f"❌ SqliteSaver with manual connection failed: {e}")
    
    # Test 3: MemorySaver
    print("\n3️⃣ Testing MemorySaver...")
    try:
        from langgraph.checkpoint.memory import MemorySaver
        memory_saver = MemorySaver()
        print("✅ MemorySaver works")
    except Exception as e:
        print(f"❌ MemorySaver failed: {e}")
    
    # Test 4: Check available checkpoint modules
    print("\n4️⃣ Checking available checkpoint modules...")
    try:
        import langgraph.checkpoint
        print("✅ langgraph.checkpoint available")
        
        # List available checkpointers
        import pkgutil
        checkpoint_modules = []
        for importer, modname, ispkg in pkgutil.iter_modules(langgraph.checkpoint.__path__):
            checkpoint_modules.append(modname)
        
        print(f"Available checkpoint modules: {checkpoint_modules}")
        
    except Exception as e:
        print(f"❌ Error checking checkpoint modules: {e}")
    
    # Test 5: Try creating a simple graph
    print("\n5️⃣ Testing simple graph compilation...")
    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        
        class SimpleState(TypedDict):
            message: str
        
        def simple_node(state):
            return {"message": "test"}
        
        workflow = StateGraph(SimpleState)
        workflow.add_node("test", simple_node)
        workflow.set_entry_point("test")
        workflow.add_edge("test", END)
        
        # Compile without checkpointer
        graph = workflow.compile()
        print("✅ Simple graph compilation works")
        
    except Exception as e:
        print(f"❌ Simple graph compilation failed: {e}")
    
    print("\n" + "="*50)
    print("🎯 Recommendations:")
    print("• If #1 failed but #2 worked: Use manual SqliteSaver connection")
    print("• If #1 and #2 failed but #3 worked: Use MemorySaver")
    print("• If all failed: Use simple compilation without checkpointer")
    print("• Check your langgraph version: pip show langgraph")

if __name__ == "__main__":
    test_checkpointers()
