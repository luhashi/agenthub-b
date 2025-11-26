from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from app.mcp_client import get_mcp_tools
import operator
import os

# Define the state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Define a simple tool for testing
@tool
def get_weather(city: str):
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 25°C."

def get_llm(model_name: str = None, temperature: float = 0):
    """
    Factory function to get the appropriate LLM based on model name.

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.5-flash")
                   If None, uses DEFAULT_MODEL from environment or defaults to "gemini-2.5-flash"
        temperature: Temperature setting for the model (0-1)

    Returns:
        A LangChain chat model instance

    Supported providers:
        - OpenAI: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, o1-preview, o1-mini
        - Anthropic: claude-3-5-sonnet-20241022, claude-3-opus-20240229, claude-3-sonnet-20240229
        - Google: gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash
    """
    model_name = model_name or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")

    # Determine provider based on model name
    if "gpt" in model_name.lower() or "o1" in model_name.lower():
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif "claude" in model_name.lower():
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    elif "gemini" in model_name.lower():
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        # Default to OpenAI if model not recognized
        print(f"Warning: Model '{model_name}' not recognized, defaulting to gpt-4o")
        return ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

async def create_agent(model_name: str = None, temperature: float = 0):
    """
    Creates and compiles the LangGraph agent with all tools.

    Args:
        model_name: Name of the model to use (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.5-flash")
                   If None, uses DEFAULT_MODEL from environment or defaults to "gemini-2.5-flash"
        temperature: Temperature setting for the model (0-1)

    Returns:
        A compiled LangGraph agent
    """
    # Initialize the model using factory function
    llm = get_llm(model_name, temperature)

    # Load tools
    local_tools = [get_weather]
    mcp_tools = await get_mcp_tools()
    all_tools = local_tools + mcp_tools
    
    # Bind tools to the model
    llm_with_tools = llm.bind_tools(all_tools)

    # Define the chatbot node
    def chatbot(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Build the graph
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(all_tools))

    graph_builder.set_entry_point("chatbot")

    # Add conditional edges
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    # Add edge from tools back to chatbot
    graph_builder.add_edge("tools", "chatbot")

    # Compile the graph
    return graph_builder.compile()
