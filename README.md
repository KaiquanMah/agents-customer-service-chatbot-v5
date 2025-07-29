# Data Analyst Chatbot - Assignment 2

A conversational AI system built with LangGraph and Streamlit for analyzing the Bitext Customer Service dataset. The system implements a multi-agent architecture with persistent memory and intelligent query routing.

Students: Yair Reichman, Kaiquan Mah


## System Overview

This chatbot provides intelligent analysis of customer service data through specialized agents that handle different types of queries. The system maintains conversation history and user preferences across sessions while providing contextual recommendations.

### Core Architecture

The application uses LangGraph's StateGraph to orchestrate multiple specialized agents:

- **Query Classifier**: Categorizes incoming queries into structured, unstructured, recommendation, or out-of-scope types
- **Structured Agent**: Handles specific data queries like counts, examples, and distributions
- **Unstructured Agent**: Manages summarization and trend analysis requests  
- **Recommendation Agent**: Suggests relevant follow-up queries based on conversation history
- **Out-of-Scope Handler**: Manages queries unrelated to the dataset

### Key Features

**Multi-Agent Processing**
- Specialized agents with distinct tool sets and capabilities
- Intelligent routing based on query classification
- Tool execution with error handling and fallback responses

**Session Management** 
- Persistent conversations using SQLite checkpointer with MemorySaver fallback
- Thread-based session isolation
- Cross-session memory retention using configurable thread IDs

**Memory System**
- User profile tracking across conversations
- Automatic summary updates after each interaction
- Context-aware follow-up query support

**Advanced Interface**
- Session explorer for viewing conversation history
- Planning mode toggle for debugging agent decisions
- Quick-action buttons for testing different scenarios

## Technical Implementation

### State Management

The system uses a TypedDict called `ChatState` to maintain conversation state:

```python
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query_type: str
    user_summary: str
    conversation_count: int
    last_query_type: str
    current_session: str
```

### Agent Architecture

**Structured Agent Tools:**
- Data filtering by intent and category
- Count operations for specific data types
- Example retrieval with filtering options
- Frequency analysis and distribution calculations

**Unstructured Agent Tools:**
- Content summarization with category/intent focus
- Trend analysis and pattern identification
- Contextual example retrieval for analysis

### Query Processing Flow

1. **Input Reception**: User query received through Streamlit interface
2. **Classification**: LLM-powered classifier determines query type
3. **Routing**: Conditional router directs to appropriate agent
4. **Tool Execution**: Agent executes relevant tools based on query
5. **Memory Update**: User profile and conversation metadata updated
6. **Response Generation**: Final response returned to user

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Required packages listed in requirements.txt
- Nebius API key for LLM access
- Bitext Customer Service dataset (CSV format)

### Environment Configuration

Create a `.env` file with your API configuration:

```bash
NEBIUS_YAIR=your-api-key-here
DEBUG=false
```

### Dataset Requirements

Place the Bitext dataset file in the project root directory:
`Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`

The dataset should contain these columns:
- instruction (customer queries)
- category (REFUND, ACCOUNT, BILLING, etc.)
- intent (get_refund, billing_question, etc.)
- response (agent responses)

### Running the Application

```bash
pip install -r requirements.txt
streamlit run streamlit_app_v2.py
```

## Usage Guide

### Session Management

Use unique session IDs to maintain separate conversation threads. The same session ID preserves conversation history across browser sessions, while different IDs create isolated conversation contexts.

### Query Types and Examples

**Structured Data Queries:**
- "What are the most frequent categories?"
- "Show 5 examples from the REFUND category"
- "How many get_refund intents exist?"
- "Display intent distribution for ACCOUNT category"

**Unstructured Analysis Queries:**
- "Summarize the REFUND category patterns"
- "What trends appear in customer complaints?"
- "Analyze agent response strategies for billing issues"

**Follow-up Query Examples:**
- "Show me more examples" (after viewing initial examples)
- "What about cancel_order intents?" (following previous intent query)
- "Expand on that analysis" (requesting deeper insight)

**Memory and Recommendation Queries:**
- "What do you remember about me?"
- "What should I explore next?"
- "Recommend a query based on our conversation"

### Testing Memory Functionality

1. Start with a specific session ID (e.g., "billing_analysis")
2. Ask several queries about billing-related topics
3. Switch to a different session ID
4. Return to the original session ID
5. Ask "What do you remember about me?" to verify memory persistence

### Interface Features

**Planning Mode Toggle:**
- ReAct Mode: Standard dynamic tool execution
- Planning Mode: Shows query classification and tool planning information

**Session Explorer:**
- View all active sessions and their checkpoint counts
- Switch between different conversation threads
- Examine message history from previous sessions

**Quick Actions:**
- Test memory functionality with pre-built queries
- Access example questions for different query types
- Clear current session or start fresh conversations

## Development and Customization

### Adding New Tools

1. Define the tool function in `langgraph_agent.py`
2. Add to appropriate agent's tool list
3. Update tool execution logic in the relevant agent function
4. Test with appropriate query examples

### Extending Memory Capabilities

Modify the `update_memory()` function to track additional user preferences or conversation patterns. Adjust the `ChatState` definition to include new state variables.

### Custom Query Classification

Update the `classify_query()` function to recognize new query types. Add corresponding routing logic and create new agent nodes as needed.

## Performance Considerations

- SQLite checkpointer provides efficient conversation storage
- MemorySaver fallback ensures system reliability
- Tool execution includes error handling for robust operation
- Session isolation prevents cross-conversation interference

## Troubleshooting

**Common Issues:**

1. **Checkpointer Errors**: System automatically falls back to MemorySaver if SQLite fails
2. **API Connection Issues**: Verify NEBIUS_YAIR environment variable is set correctly
3. **Dataset Loading**: Ensure CSV file is in project root with correct filename
4. **Memory Not Persisting**: Check that session IDs remain consistent across sessions

**Debug Mode:**

Set `DEBUG=true` in environment variables to enable detailed logging and debug information in the Streamlit interface.

## Project Structure

```
├── streamlit_app_v2.py          # Main Streamlit interface
├── langgraph_agent.py           # LangGraph implementation and agents
├── checkpointer_test.py         # Checkpointer compatibility testing
├── debug_agent_invocation.py    # Agent debugging utilities
├── setup_script.py              # Installation verification script
├── requirements.txt             # Python dependencies
└── Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
```

## Testing Scripts

- `checkpointer_test.py`: Tests different checkpointer implementations for compatibility
- `debug_agent_invocation.py`: Provides step-by-step agent debugging capabilities
- `setup_script.py`: Verifies installation and configuration completeness
