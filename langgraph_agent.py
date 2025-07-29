import pandas as pd
import json
import os
from typing import TypedDict, Annotated, Literal, List, Dict, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()


# Load dataset
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')
df = pd.read_csv(file_path)

# Configure LLM
NEBIUS_API_KEY = os.getenv("NEBIUS_YAIR")
llm = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=NEBIUS_API_KEY,
    model="Qwen/Qwen2.5-32B-Instruct",
    temperature=0
)

class ChatState(TypedDict):
    """State for the chat agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    query_type: str
    user_summary: str
    conversation_count: int
    last_query_type: str
    current_session: str

# Tools for structured queries
def select_semantic_intent(intent_names: List[str]) -> str:
    """Filter the dataframe by a list of intents"""
    if isinstance(intent_names, str):
        intent_names = [intent_names]
    
    dataset = df[df["intent"].isin(intent_names)]
    return f"Filtered intents {intent_names} with {len(dataset)} rows"

def select_semantic_category(category_names: List[str]) -> str:
    """Filter the dataframe by a list of categories"""
    if isinstance(category_names, str):
        category_names = [category_names]
    
    dataset = df[df["category"].isin(category_names)]
    return f"Filtered categories {category_names} with {len(dataset)} rows"

def get_all_intents() -> List[str]:
    """Return all available intent names"""
    return df["intent"].unique().tolist()

def get_all_categories() -> List[str]:
    """Return all available category names"""
    return df["category"].unique().tolist()

def count_intent(intent_name: str) -> int:
    """Count rows with specific intent"""
    if df is None or len(df) == 0:
        return 0
    return int((df["intent"] == intent_name).sum())

def count_category(category_name: str) -> int:
    """Count rows with specific category"""
    if df is None or len(df) == 0:
        return 0
    return int((df["category"] == category_name).sum())

def show_examples(n: int, category: str = None, intent: str = None) -> Dict:
    """Show random sample of n examples, optionally filtered by category or intent"""
    if df is None or len(df) == 0:
        return {"error": "No data available"}
    
    filtered_df = df.copy()
    
    if category:
        filtered_df = filtered_df[filtered_df["category"] == category]
    if intent:
        filtered_df = filtered_df[filtered_df["intent"] == intent]
    
    if len(filtered_df) < n:
        n = len(filtered_df)
    
    if n == 0:
        return {"error": "No data found for the specified filters"}
    
    sample = filtered_df.sample(n).to_dict(orient="records")
    return {"examples": sample, "count": len(sample)}

def get_most_frequent_categories(limit: int = 10) -> Dict:
    """Get most frequent categories"""
    category_counts = df["category"].value_counts().head(limit)
    return {
        "categories": category_counts.to_dict(),
        "total_categories": len(df["category"].unique())
    }

def get_intent_distribution(category: str = None) -> Dict:
    """Get intent distribution, optionally for a specific category"""
    if category:
        filtered_df = df[df["category"] == category]
        intent_counts = filtered_df["intent"].value_counts()
        return {
            "distribution": intent_counts.to_dict(),
            "category": category,
            "total_intents": len(intent_counts)
        }
    else:
        intent_counts = df["intent"].value_counts()
        return {
            "distribution": intent_counts.to_dict(),
            "total_intents": len(intent_counts)
        }

def summarise(user_request: str, category: str = None, intent: str = None) -> str:
    """Summarize data based on user request"""
    if category:
        category_data = df[df["category"] == category]
        if len(category_data) == 0:
            return f"No data found for category: {category}"
        
        # Analyze the category
        intent_counts = category_data["intent"].value_counts()
        total_records = len(category_data)
        
        summary = f"""Summary for category '{category}':
- Total records: {total_records}
- Most common intents: {dict(intent_counts.head(3))}
- Represents {(total_records/len(df)*100):.1f}% of all data
"""
        return summary
    
    elif intent:
        intent_data = df[df["intent"] == intent]
        if len(intent_data) == 0:
            return f"No data found for intent: {intent}"
        
        # Analyze the intent
        category_counts = intent_data["category"].value_counts()
        total_records = len(intent_data)
        
        summary = f"""Summary for intent '{intent}':
- Total records: {total_records}
- Categories: {dict(category_counts)}
- Represents {(total_records/len(df)*100):.1f}% of all data
"""
        return summary
    
    return f"General summary requested: {user_request}"

# Structured agent tools
structured_tools = [
    select_semantic_intent,
    select_semantic_category,
    get_all_intents,
    get_all_categories,
    count_intent,
    count_category,
    show_examples,
    get_most_frequent_categories,
    get_intent_distribution
]

# Unstructured agent tools
unstructured_tools = [
    summarise,
    get_all_categories,
    get_all_intents,
    show_examples
]

# Create tool nodes
structured_tool_node = ToolNode(structured_tools)
unstructured_tool_node = ToolNode(unstructured_tools)


def classify_query(state: ChatState) -> ChatState:
    """Classify the user query - FIXED VERSION"""

    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    classification_prompt = f"""
    Classify the following user query into one of these categories:

    1. "structured" - Questions about specific data analysis, counts, examples, distributions
    2. "unstructured" - Questions asking for summaries or explanations  
    3. "recommend" - Questions asking for query suggestions
    4. "out_of_scope" - Questions unrelated to the customer service dataset

    User query: "{last_message}"

    Respond with only one word: structured, unstructured, recommend, or out_of_scope
    """

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    query_type = response.content.strip().lower()

    # Ensure valid classification
    if query_type not in ["structured", "unstructured", "recommend", "out_of_scope"]:
        query_type = "out_of_scope"

    # CRITICAL FIX: Return a NEW state dict, don't modify the original
    return {
        **state,  # Copy all existing state
        "query_type": query_type,
        "last_query_type": query_type
    }

def route_query(state: ChatState) -> Literal["structured_agent", "unstructured_agent", "recommend_agent", "out_of_scope_handler"]:
    """Route the query to appropriate agent based on classification"""
    query_type = state.get("query_type", "out_of_scope")
    
    if query_type == "structured":
        return "structured_agent"
    elif query_type == "unstructured":
        return "unstructured_agent"
    elif query_type == "recommend":
        return "recommend_agent"
    else:
        return "out_of_scope_handler"


def structured_agent(state: ChatState) -> ChatState:
    """Handle structured queries - FIXED VERSION"""

    messages = state["messages"]

    # Bind tools to LLM for structured queries
    llm_with_tools = llm.bind_tools(structured_tools)

    system_message = """You are a data analyst for customer service data. Use the available tools to answer structured questions about the Bitext Customer Service dataset. Always use tools to get accurate data. Be precise and helpful."""

    # Create conversation with system message
    conversation = [{"role": "system", "content": system_message}]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            conversation.append({"role": "assistant", "content": msg.content})

    response = llm_with_tools.invoke(conversation)

    # If tools are called, execute them
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            try:
                if tool_name == "get_all_categories":
                    result = get_all_categories()
                elif tool_name == "get_all_intents":
                    result = get_all_intents()
                elif tool_name == "count_category":
                    result = count_category(tool_args["category_name"])
                elif tool_name == "count_intent":
                    result = count_intent(tool_args["intent_name"])
                elif tool_name == "show_examples":
                    result = show_examples(
                        tool_args["n"],
                        tool_args.get("category"),
                        tool_args.get("intent")
                    )
                elif tool_name == "get_most_frequent_categories":
                    result = get_most_frequent_categories(tool_args.get("limit", 10))
                elif tool_name == "get_intent_distribution":
                    result = get_intent_distribution(tool_args.get("category"))
                else:
                    result = f"Unknown tool: {tool_name}"

                # Add tool result to conversation and get final response
                tool_message = f"Tool {tool_name} result: {result}"
                final_response = llm.invoke(conversation + [
                    {"role": "assistant", "content": f"I'll use {tool_name} to help answer your question."},
                    {"role": "user", "content": tool_message}
                ])


                # this fix might work for single-tool calls
                # also, if there is an error in the tool call, we still have the final_response.content
                # instead of having only the exception (without earlier tool call results) -> if we shifted this return block below 'exception Exception as e' just outside the for loop, but before the else-statement
                # CRITICAL FIX: Return NEW state, don't modify original
                return {
                    **state,
                    "messages": messages + [AIMessage(content=final_response.content)]
                }

            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                return {
                    **state,
                    "messages": messages + [AIMessage(content=error_msg)]
                }
    else:
        # CRITICAL FIX: Return NEW state for no tool calls case too
        return {
            **state,
            "messages": messages + [AIMessage(content=response.content)]
        }


def unstructured_agent(state: ChatState) -> ChatState:
    """Handle unstructured queries requiring summarization - FIXED VERSION"""

    messages = state["messages"]

    # Bind tools to LLM for unstructured queries
    llm_with_tools = llm.bind_tools(unstructured_tools)

    system_message = """You are a data analyst specializing in summarization and trend analysis of customer service data.

Available tools:
- summarise: Create summaries based on user requests
- get_all_categories: Get all available categories
- get_all_intents: Get all available intents
- show_examples: Show sample records for context

Focus on providing insights, trends, and comprehensive summaries. Use tools to gather data, then provide thoughtful analysis."""

    conversation = [{"role": "system", "content": system_message}]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            conversation.append({"role": "assistant", "content": msg.content})

    response = llm_with_tools.invoke(conversation)

    # Handle tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            try:
                if tool_name == "summarise":
                    result = summarise(
                        tool_args["user_request"],
                        tool_args.get("category"),
                        tool_args.get("intent")
                    )
                elif tool_name == "get_all_categories":
                    result = get_all_categories()
                elif tool_name == "get_all_intents":
                    result = get_all_intents()
                elif tool_name == "show_examples":
                    result = show_examples(
                        tool_args["n"],
                        tool_args.get("category"),
                        tool_args.get("intent")
                    )
                else:
                    result = f"Unknown tool: {tool_name}"

                tool_message = f"Tool {tool_name} result: {result}"
                final_response = llm.invoke(conversation + [
                    {"role": "assistant", "content": f"I'll use {tool_name} to help answer your question."},
                    {"role": "user", "content": tool_message}
                ])

                # FIXED: Return new state instead of modifying
                return {
                    **state,
                    "messages": messages + [AIMessage(content=final_response.content)]
                }

            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                # FIXED: Return new state instead of modifying
                return {
                    **state,
                    "messages": messages + [AIMessage(content=error_msg)]
                }
    else:
        # FIXED: Return new state for no tool calls case
        return {
            **state,
            "messages": messages + [AIMessage(content=response.content)]
        }


def recommend_agent(state: ChatState) -> ChatState:
    """Handle query recommendation requests based on conversation history - FIXED VERSION"""

    messages = state["messages"]
    user_summary = state.get("user_summary", "No previous interactions")
    last_query_type = state.get("last_query_type", "none")

    recommendation_prompt = f"""
    Based on the conversation history and user profile, suggest 2-3 relevant queries the user might want to explore next.

    User Profile: {user_summary}
    Last Query Type: {last_query_type}

    Recent conversation:
    {messages[-5:] if len(messages) > 5 else messages}

    Available query types:
    1. Structured queries: "What are the most frequent categories?", "Show examples of X category", "How many X intents?"
    2. Unstructured queries: "Summarize category X", "What trends do you see in Y?"

    Provide 2-3 specific, actionable query suggestions that would logically follow from their previous interactions.
    Be conversational and explain why each suggestion might be interesting.
    """

    response = llm.invoke([HumanMessage(content=recommendation_prompt)])

    # FIXED: Return new state instead of modifying
    return {
        **state,
        "messages": messages + [AIMessage(content=response.content)]
    }


def out_of_scope_handler(state: ChatState) -> ChatState:
    """Handle out-of-scope queries - FIXED VERSION"""

    messages = state["messages"]

    out_of_scope_response = """I'm a customer service data analyst chatbot focused on the Bitext Customer Service dataset. 

I can help you with:
📊 **Structured queries**: Category counts, intent distributions, examples
📝 **Unstructured queries**: Summaries and trend analysis  
💡 **Recommendations**: Suggest what to explore next

I cannot help with general questions outside of customer service data analysis. 

Would you like me to suggest some queries you can try instead?"""

    # FIXED: Return new state instead of modifying
    return {
        **state,
        "messages": messages + [AIMessage(content=out_of_scope_response)]
    }


def update_memory(state: ChatState) -> ChatState:
    """Update user summary and conversation metadata - FIXED VERSION"""

    messages = state["messages"]
    current_summary = state.get("user_summary", "")
    conversation_count = state.get("conversation_count", 0) + 1

    # Extract recent interactions for summary update
    recent_messages = messages[-4:] if len(messages) > 4 else messages

    if conversation_count > 1:  # Only update summary after first interaction
        summary_prompt = f"""
        Update the user profile based on this conversation. Keep it concise (2-3 sentences max).

        Current profile: {current_summary if current_summary else "New user"}

        Recent interaction:
        {recent_messages}

        Focus on: interests shown, types of queries preferred, specific categories/intents explored.
        Only include meaningful patterns, not one-off questions.
        """

        try:
            response = llm.invoke([HumanMessage(content=summary_prompt)])
            updated_summary = response.content.strip()
        except:
            updated_summary = current_summary
    else:
        # First interaction - create initial summary
        if recent_messages:
            last_query = recent_messages[-2].content if len(recent_messages) > 1 else ""
            query_type = state.get("query_type", "unknown")
            updated_summary = f"User interested in {query_type} analysis. First query: {last_query[:50]}..."
        else:
            updated_summary = current_summary

    # CRITICAL FIX: Return NEW state dict
    return {
        **state,
        "user_summary": updated_summary,
        "conversation_count": conversation_count
    }

# Create the graph
def create_graph():
    """Create and return the LangGraph workflow"""
    
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("structured_agent", structured_agent)
    workflow.add_node("unstructured_agent", unstructured_agent)
    workflow.add_node("recommend_agent", recommend_agent)
    workflow.add_node("out_of_scope_handler", out_of_scope_handler)
    workflow.add_node("update_memory", update_memory)
    
    # Set entry point
    workflow.set_entry_point("classify_query")
    
    # Add conditional edges from classifier
    workflow.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "structured_agent": "structured_agent",
            "unstructured_agent": "unstructured_agent", 
            "recommend_agent": "recommend_agent",
            "out_of_scope_handler": "out_of_scope_handler"
        }
    )
    
    # All agents go to memory update
    workflow.add_edge("structured_agent", "update_memory")
    workflow.add_edge("unstructured_agent", "update_memory")
    workflow.add_edge("recommend_agent", "update_memory")
    workflow.add_edge("out_of_scope_handler", "update_memory")
    
    # Memory update goes to end
    workflow.add_edge("update_memory", END)
    
    return workflow

# Create checkpointer for memory persistence
memory_saver = SqliteSaver.from_conn_string("./session_memory.db")

def get_agent():
    """Get compiled agent with memory"""
    """Get compiled agent with MemorySaver"""
    try:
        from langgraph.checkpoint.memory import MemorySaver

        memory_saver = MemorySaver()
        workflow = create_graph()
        return workflow.compile(checkpointer=memory_saver)
    except Exception as e:
        print(f"Solution 3 failed: {e}")
        return None
