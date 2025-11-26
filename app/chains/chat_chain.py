# app/chains/chat_chain.py
"""
Data Analyst Chat Chain using LangGraph.

This module provides a conversational AI agent specialized as a data analyst
with access to various tools and datasets. Refactored from LangChain to LangGraph
for better control flow and state management.
"""

import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# Import the model factory from agents
from app.agents.langgraph_agent import get_llm
from app.mcp_client import get_mcp_tools
import operator

# Load environment variables
load_dotenv()

# System prompt for the data analyst persona
SYSTEM_PROMPT = """You are a very knowledgeable AI Data Analyst. You have access to many curated datasets, including data from the U.S. Census Bureau, the FDA, the World Health Organization, and more. Use this information to answer questions accurately and seamlessly.

When analyzing data:
- Be precise and cite your sources when possible
- Explain your methodology clearly
- Provide context for statistical findings
- Use visualizations concepts when describing data patterns
- Ask clarifying questions if the request is ambiguous"""


# Define the agent state
class DataAnalystState(TypedDict):
    """State for the data analyst conversation."""
    messages: Annotated[List[BaseMessage], operator.add]


# Define example tools for data analysis
@tool
def calculate_statistics(data: str):
    """
    Calculate basic statistics (mean, median, mode, std dev) for a dataset.

    Args:
        data: Comma-separated numbers or description of data to analyze
    """
    # This is a placeholder - in production, this would perform actual calculations
    return f"Statistics calculated for data: {data}. Mean: 45.2, Median: 43.0, Std Dev: 12.5"


@tool
def search_census_data(query: str):
    """
    Search U.S. Census Bureau data for demographic and economic information.

    Args:
        query: What demographic or economic data to search for
    """
    # Placeholder - in production, this would query actual Census API
    return f"Census data search results for '{query}': Population trends, income statistics, and demographic breakdowns available."


@tool
def search_health_data(query: str):
    """
    Search WHO and FDA databases for health-related data and statistics.

    Args:
        query: Health topic or condition to search for
    """
    # Placeholder - in production, this would query actual health databases
    return f"Health data for '{query}': Latest statistics, trends, and research findings from WHO and FDA databases."


async def create_data_analyst_chain(
    model_name: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000
):
    """
    Creates and compiles a LangGraph-based data analyst chat chain.

    Args:
        model_name: Name of the model to use (e.g., "gemini-2.5-flash", "gpt-4o", "claude-3-5-sonnet-20241022")
                   If None, uses DEFAULT_MODEL from environment or defaults to "gemini-2.5-flash"
        temperature: Temperature setting for the model (0-1). Default 0.7 for more creative responses
        max_tokens: Maximum tokens for responses (default 1000)

    Returns:
        A compiled LangGraph agent with memory and data analyst tools
    """
    # Initialize the model with higher temperature for conversational responses
    llm = get_llm(model_name, temperature)

    # Load local data analyst tools
    local_tools = [
        calculate_statistics,
        search_census_data,
        search_health_data,
    ]

    # Load MCP tools if configured
    try:
        mcp_tools = await get_mcp_tools()
        all_tools = local_tools + mcp_tools
        print(f"Data Analyst Chain loaded with {len(local_tools)} local tools and {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
        all_tools = local_tools

    # Bind tools to the model
    llm_with_tools = llm.bind_tools(all_tools)

    # Define the chatbot node with system prompt
    def chatbot(state: DataAnalystState):
        """Main chatbot node that processes messages with data analyst context."""
        messages = state["messages"]

        # Add system prompt if this is the first message or if there's no system message
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    graph_builder = StateGraph(DataAnalystState)

    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(all_tools))

    # Set entry point
    graph_builder.set_entry_point("chatbot")

    # Add conditional edges for tool calling
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    # Add edge from tools back to chatbot
    graph_builder.add_edge("tools", "chatbot")

    # Compile with memory saver for conversation persistence
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph


async def invoke_data_analyst(
    message: str,
    thread_id: str,
    model_name: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000
):
    """
    Convenience function to invoke the data analyst chain.

    Args:
        message: User's message/question
        thread_id: Thread ID for conversation continuity
        model_name: Optional model override
        temperature: Temperature for response generation
        max_tokens: Maximum tokens for response

    Returns:
        The assistant's response text
    """
    # Create the chain
    chain = await create_data_analyst_chain(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Prepare inputs
    inputs = {"messages": [HumanMessage(content=message)]}
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke the chain
    result = await chain.ainvoke(inputs, config=config)

    # Extract the last message
    last_message = result["messages"][-1]
    return last_message.content


# For backward compatibility and testing
if __name__ == "__main__":
    import asyncio

    async def test_chain():
        """Test the data analyst chain."""
        print("Testing Data Analyst Chain...")

        # Test with a data analysis question
        response = await invoke_data_analyst(
            message="What are the latest population trends in the United States?",
            thread_id="test-thread-1",
            model_name="gemini-2.5-flash"
        )

        print(f"\nResponse: {response}")

    # Run test
    asyncio.run(test_chain())
