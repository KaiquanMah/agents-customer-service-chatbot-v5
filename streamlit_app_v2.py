import streamlit as st
import os
import sqlite3
import json
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_agent import get_agent, ChatState
from dotenv import load_dotenv

load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Data Analyst Chatbot - Assignment 2",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Data Analyst Chatbot - LangGraph Implementation")
st.markdown("*Powered by LangGraph • Customer Service Dataset Analysis*")


# SessionManager class integrated directly into the app
class SessionManager:
    """Utility class to manage and visualize LangGraph session memory"""

    def __init__(self, db_path=":memory:"):
        self.db_path = db_path

    def get_all_sessions(self):
        """Get all available session threads"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if checkpoints table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='checkpoints'
            """)
            table_exists = cursor.fetchone()

            sessions = []
            if table_exists:
                cursor.execute("""
                    SELECT DISTINCT thread_id, 
                           COUNT(*) as checkpoint_count,
                           MAX(checkpoint_id) as latest_checkpoint
                    FROM checkpoints 
                    GROUP BY thread_id
                    ORDER BY latest_checkpoint DESC
                """)
                sessions = cursor.fetchall()

            conn.close()
            return sessions

        except Exception as e:
            st.error(f"Error accessing session data: {e}")
            return []

    def get_session_messages(self, thread_id):
        """Get messages for a specific session"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT checkpoint, metadata
                FROM checkpoints 
                WHERE thread_id = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
            """, (thread_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                checkpoint_data = json.loads(result[0]) if result[0] else {}
                return checkpoint_data.get('messages', [])

            return []

        except Exception as e:
            st.error(f"Error retrieving session messages: {e}")
            return []


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "default_session"

if "user_summary" not in st.session_state:
    st.session_state.user_summary = ""

if "example_query" not in st.session_state:
    st.session_state.example_query = None

# Create SessionManager instance
session_manager = SessionManager(":memory:")  # Use same as langgraph_agent.py

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Session Management
    st.subheader("🔗 Session Management")
    session_id = st.text_input(
        "Session ID",
        value=st.session_state.get("session_id", "default_session"),
        help="Enter a unique session ID to maintain conversation history"
    )

    if session_id != st.session_state.get("session_id"):
        st.session_state.session_id = session_id
        st.session_state.messages = []
        st.rerun()

    # Planning Mode Toggle
    st.subheader("🧠 Planning Mode")
    planning_mode = st.radio(
        "Agent Mode:",
        options=["ReAct", "Planning"],
        index=0,
        help="""
        • **ReAct**: Dynamic planning and execution (default)
        • **Planning**: Pre-plan tool calls before execution
        """
    )

    # Memory Management
    st.subheader("💾 Memory")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Session"):
            st.session_state.messages = []
            st.session_state.user_summary = ""
            st.success("Session cleared!")
            st.rerun()

    with col2:
        if st.button("New Session"):
            # GENERATE SESSION_ID BASED ON UTC+0 TIMESTAMP
            new_session_id = f"session_{datetime.now().strftime('%H%M%S')}"
            st.session_state.session_id = new_session_id
            st.session_state.messages = []
            st.session_state.user_summary = ""
            st.success(f"New session: {new_session_id}")
            st.rerun()

    # Enhanced Session Info
    st.subheader("📊 Current Session Info")
    st.write(f"**Session ID:** `{session_id}`")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    st.write(f"**Has Profile:** {'✅' if st.session_state.user_summary else '❌'}")

    if st.session_state.user_summary:
        st.write("**User Profile:**")
        st.write(f"_{st.session_state.user_summary}_")

    # Quick Actions
    st.subheader("⚡ Quick Actions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Test Memory"):
            st.session_state.example_query = "What do you remember about me?"
            st.rerun()

    with col2:
        if st.button("Get Advice"):
            st.session_state.example_query = "Advise me what to query next"
            st.rerun()

    # Session Explorer - HERE'S WHERE SessionManager IS USED!
    st.subheader("🔍 Session Explorer")

    # Get all sessions using SessionManager
    all_sessions = session_manager.get_all_sessions()

    if all_sessions:
        st.write(f"**Total Sessions:** {len(all_sessions)}")

        # Display sessions
        for thread_id, checkpoint_count, latest_checkpoint in all_sessions:
            with st.expander(f"📋 {thread_id} ({checkpoint_count} checkpoints)"):
                st.write(f"**Thread ID:** {thread_id}")
                st.write(f"**Checkpoints:** {checkpoint_count}")
                st.write(f"**Latest:** {latest_checkpoint}")

                # Button to switch to this session
                if st.button(f"Switch to {thread_id}", key=f"switch_{thread_id}"):
                    st.session_state.session_id = thread_id
                    st.session_state.messages = []  # Will reload from checkpoint
                    st.rerun()

                # Show some messages from this session
                messages = session_manager.get_session_messages(thread_id)
                if messages:
                    st.write("**Recent Messages:**")
                    for msg in messages[-3:]:  # Show last 3 messages
                        if hasattr(msg, 'content'):
                            msg_type = "User" if isinstance(msg, HumanMessage) else "Assistant"
                            st.write(f"- {msg_type}: {msg.content[:50]}...")
    else:
        st.write("**No sessions found**")
        st.info("Start chatting to create your first session!")

    # Memory Testing Interface
    st.subheader("🧪 Memory Testing")

    with st.expander("Memory Test Guide"):
        st.write("**How to test memory:**")
        st.write("1. Use session 'billing_test'")
        st.write("2. Ask about billing categories")
        st.write("3. Switch to 'refund_test'")
        st.write("4. Ask about refunds")
        st.write("5. Switch back to 'billing_test'")
        st.write("6. Ask 'What do you remember?'")

        test_sessions = ['billing_test', 'refund_test', 'account_test']
        selected_test = st.selectbox("Quick Test Session:", test_sessions)

        if st.button("Switch to Test Session"):
            st.session_state.session_id = selected_test
            st.session_state.messages = []
            st.rerun()

# Example queries in expander
with st.expander("💡 Example Questions", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**📊 Structured Questions:**")
        structured_examples = [
            "What are the most frequent categories?",
            "Show 5 examples of REFUND category",
            "How many get_refund intents are there?",
            "What categories exist?",
            "Show intent distribution for ACCOUNT category"
        ]
        for i, example in enumerate(structured_examples):
            if st.button(example, key=f"struct_{i}"):
                st.session_state.example_query = example
                st.rerun()

    with col2:
        st.write("**📝 Unstructured Questions:**")
        unstructured_examples = [
            "Summarize the REFUND category",
            "What trends do you see in customer complaints?",
            "Summarize how agents respond to billing issues",
            "Analyze the most common customer problems"
        ]
        for i, example in enumerate(unstructured_examples):
            if st.button(example, key=f"unstruct_{i}"):
                st.session_state.example_query = example
                st.rerun()

    with col3:
        st.write("**🎯 Special Queries:**")
        special_examples = [
            "What do you remember about me?",
            "Advise me what to query next",
            "Who is Magnus Carlsen?",
            "Help me file a claim"
        ]
        for i, example in enumerate(special_examples):
            if st.button(example, key=f"special_{i}"):
                st.session_state.example_query = example
                st.rerun()


# Function to process queries (both from chat input and example buttons)
def process_query(prompt):
    """Process a query through the LangGraph agent"""

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with LangGraph agent
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Get the agent
                agent = get_agent()

                # Prepare state
                config = {"configurable": {"thread_id": session_id}}

                # Convert session messages to LangChain format
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        langchain_messages.append(AIMessage(content=msg["content"]))

                # Handle "what do you remember" query specially
                if "remember" in prompt.lower() and "me" in prompt.lower():
                    if st.session_state.user_summary:
                        response_content = f"Here's what I remember about you:\n\n{st.session_state.user_summary}\n\nI've had {len(st.session_state.messages) // 2} conversations with you in this session."
                    else:
                        response_content = "This is our first interaction, so I don't have any information about you yet. As we chat more, I'll learn about your interests and preferences!"

                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                else:
                    # Prepare initial state
                    initial_state = {
                        "messages": [langchain_messages[-1]],  # Only the latest message
                        "query_type": "",
                        "user_summary": st.session_state.user_summary,
                        "conversation_count": len(st.session_state.messages) // 2,
                        "last_query_type": "",
                        "current_session": session_id
                    }

                    # Show planning mode info
                    if planning_mode == "Planning":
                        st.info("🧠 **Planning Mode**: Pre-analyzing query and planning tool usage...")

                    # Invoke the agent
                    result = agent.invoke(initial_state, config=config)

                    # Extract response from result
                    if result and "messages" in result:
                        assistant_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
                        if assistant_messages:
                            response_content = assistant_messages[-1].content
                        else:
                            response_content = "I apologize, but I couldn't process your request properly."

                        # Update user summary if available
                        if "user_summary" in result:
                            st.session_state.user_summary = result["user_summary"]
                    else:
                        response_content = "I apologize, but I encountered an issue processing your request."

                    # Display query classification if in planning mode
                    if planning_mode == "Planning" and "query_type" in result:
                        query_type = result["query_type"]
                        st.success(f"📋 **Query Classification**: {query_type.replace('_', ' ').title()}")

                    # Display response
                    st.markdown(response_content)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}\n\nPlease check your API configuration and try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# Process example query if one was selected
if st.session_state.example_query:
    prompt = st.session_state.example_query
    st.session_state.example_query = None  # Clear the example query

    process_query(prompt)
    st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about the customer service dataset..."):
    process_query(prompt)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🎯 Features:**")
    st.markdown("• Query Classification")
    st.markdown("• Memory Persistence")
    st.markdown("• Follow-up Support")
    st.markdown("• Session Explorer")

with col2:
    st.markdown("**🛠️ Tools Available:**")
    st.markdown("• Data Filtering & Counting")
    st.markdown("• Example Retrieval")
    st.markdown("• Summarization & Analysis")
    st.markdown("• Session Management")

with col3:
    st.markdown("**💡 Tips:**")
    st.markdown("• Use Session ID for persistence")
    st.markdown("• Try follow-up questions")
    st.markdown("• Explore different sessions")
    st.markdown("• Test memory features")

# Debug info (only show in development)
if os.getenv("DEBUG", "false").lower() == "true":
    with st.expander("🔧 Debug Info"):
        st.write("**Session State:**")
        st.json({
            "session_id": st.session_state.get("session_id", "none"),
            "message_count": len(st.session_state.messages),
            "user_summary": st.session_state.get("user_summary", "none"),
            "planning_mode": planning_mode,
            "all_sessions": len(session_manager.get_all_sessions())
        })

        # Show all sessions in debug
        st.write("**All Sessions:**")
        for session_info in session_manager.get_all_sessions():
            st.write(f"- {session_info}")
