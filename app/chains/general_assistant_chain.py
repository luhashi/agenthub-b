# app/chains/general_assistant_chain.py
"""
General Purpose Assistant Chain using LangGraph.

This module provides a helpful, versatile AI assistant with access to web search,
calculations, and other utility tools for general-purpose conversations.
"""

import os
from typing import TypedDict, Annotated, List
from datetime import datetime
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

# System prompt for the general assistant persona
SYSTEM_PROMPT = """You are a helpful, knowledgeable, and friendly AI assistant. You can help with a wide variety of tasks including:

- Answering general knowledge questions
- Performing calculations and conversions
- Searching for current information on the web
- Providing weather updates
- Helping with problem-solving and brainstorming
- Explaining complex topics in simple terms
- Offering creative suggestions and ideas

You have access to tools for web search, calculations, and more. When you don't know something current or specific, use your search tool. Always be accurate, honest, and helpful. If you're unsure about something, say so rather than making up information."""


# Define the agent state
class AssistantState(TypedDict):
    """State for the general assistant conversation."""
    messages: Annotated[List[BaseMessage], operator.add]


# Define utility tools
@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)", "sin(3.14)")

    Returns:
        The result of the calculation

    Examples:
        - "25 * 4" -> "100"
        - "sqrt(144)" -> "12.0"
        - "(100 + 50) / 3" -> "50.0"
    """
    try:
        # Safe evaluation with math functions
        import math
        # Create a safe namespace with math functions
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
            'tan': math.tan, 'log': math.log, 'log10': math.log10,
            'exp': math.exp, 'pi': math.pi, 'e': math.e,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current date and time.

    Args:
        timezone: The timezone (currently only supports UTC)

    Returns:
        Current date and time as a formatted string
    """
    now = datetime.utcnow()
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\nDay of week: {now.strftime('%A')}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for current information.

    Args:
        query: The search query

    Returns:
        Search results and information

    Note: This is a placeholder implementation. In production, integrate with:
        - Google Custom Search API
        - Bing Search API
        - DuckDuckGo API
        - Tavily AI Search
        - SerpAPI
    """
    # Placeholder - in production, integrate with actual search API
    return f"""Web Search Results for: "{query}"

[This is a placeholder. To enable real web search, integrate one of these APIs:

1. Google Custom Search API:
   - Get API key from: https://console.cloud.google.com/
   - Add to .env: GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID

2. Tavily AI Search (Recommended):
   - Get API key from: https://tavily.com/
   - Add to .env: TAVILY_API_KEY
   - Install: pip install tavily-python

3. SerpAPI:
   - Get API key from: https://serpapi.com/
   - Add to .env: SERPAPI_API_KEY
   - Install: pip install google-search-results

For now, I'll provide information based on my training data.]

Search query: {query}
Note: Connect a real search API for current web results."""


@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for a city.

    Args:
        city: Name of the city

    Returns:
        Weather information

    Note: This is a placeholder. In production, integrate with:
        - OpenWeatherMap API
        - WeatherAPI.com
        - Visual Crossing Weather API
    """
    # Placeholder - in production, integrate with actual weather API
    return f"""Weather for {city}:

[This is a placeholder. To enable real weather data, integrate a weather API:

1. OpenWeatherMap (Free tier available):
   - Get API key from: https://openweathermap.org/api
   - Add to .env: OPENWEATHER_API_KEY

2. WeatherAPI.com (Free tier available):
   - Get API key from: https://www.weatherapi.com/
   - Add to .env: WEATHER_API_KEY

For now, I can provide general weather information based on typical patterns.]

Location: {city}
Note: Connect a weather API for real-time data."""


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between different units of measurement.

    Args:
        value: The numeric value to convert
        from_unit: Unit to convert from (e.g., "celsius", "km", "lbs")
        to_unit: Unit to convert to (e.g., "fahrenheit", "miles", "kg")

    Returns:
        The converted value

    Supported conversions:
        - Temperature: celsius, fahrenheit, kelvin
        - Distance: km, miles, meters, feet
        - Weight: kg, lbs, grams, ounces
    """
    conversions = {
        ('celsius', 'fahrenheit'): lambda x: (x * 9/5) + 32,
        ('fahrenheit', 'celsius'): lambda x: (x - 32) * 5/9,
        ('celsius', 'kelvin'): lambda x: x + 273.15,
        ('kelvin', 'celsius'): lambda x: x - 273.15,
        ('km', 'miles'): lambda x: x * 0.621371,
        ('miles', 'km'): lambda x: x * 1.60934,
        ('meters', 'feet'): lambda x: x * 3.28084,
        ('feet', 'meters'): lambda x: x * 0.3048,
        ('kg', 'lbs'): lambda x: x * 2.20462,
        ('lbs', 'kg'): lambda x: x * 0.453592,
        ('grams', 'ounces'): lambda x: x * 0.035274,
        ('ounces', 'grams'): lambda x: x * 28.3495,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    else:
        return f"Conversion from {from_unit} to {to_unit} not supported. Supported units: celsius, fahrenheit, kelvin, km, miles, meters, feet, kg, lbs, grams, ounces"


@tool
def get_fun_fact(topic: str = "general") -> str:
    """
    Get an interesting fun fact about a topic.

    Args:
        topic: The topic for the fun fact (e.g., "science", "history", "nature")

    Returns:
        An interesting fun fact
    """
    facts = {
        "science": "Honey never spoils. Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still perfectly edible!",
        "history": "Oxford University is older than the Aztec Empire. Teaching started in Oxford in 1096, while the Aztec Empire began in 1428.",
        "nature": "Octopuses have three hearts and blue blood. Two hearts pump blood to the gills, while the third pumps it to the rest of the body.",
        "space": "A day on Venus is longer than its year. Venus takes 243 Earth days to rotate once, but only 225 Earth days to orbit the Sun.",
        "animals": "Sea otters hold hands while sleeping to prevent drifting apart from their group.",
        "technology": "The first computer mouse was made of wood and was invented by Doug Engelbart in 1964.",
        "general": "Bananas are berries, but strawberries aren't! In botanical terms, a berry is a fruit from one flower with one ovary."
    }

    fact = facts.get(topic.lower(), facts["general"])
    return f"🎓 Fun Fact about {topic}:\n\n{fact}"


async def create_general_assistant_chain(
    model_name: str = None,
    temperature: float = 0.7,
):
    """
    Creates and compiles a LangGraph-based general assistant chat chain.

    Args:
        model_name: Name of the model to use (e.g., "gemini-2.5-flash", "gpt-4o", "claude-3-5-sonnet-20241022")
                   If None, uses DEFAULT_MODEL from environment or defaults to "gemini-2.5-flash"
        temperature: Temperature setting for the model (0-1). Default 0.7 for balanced responses

    Returns:
        A compiled LangGraph agent with memory and utility tools
    """
    # Initialize the model
    llm = get_llm(model_name, temperature)

    # Load local utility tools
    local_tools = [
        calculate,
        get_current_time,
        search_web,
        get_weather,
        convert_units,
        get_fun_fact,
    ]

    # Load MCP tools if configured
    try:
        mcp_tools = await get_mcp_tools()
        all_tools = local_tools + mcp_tools
        print(f"General Assistant loaded with {len(local_tools)} local tools and {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
        all_tools = local_tools

    # Bind tools to the model
    llm_with_tools = llm.bind_tools(all_tools)

    # Define the chatbot node with system prompt
    def chatbot(state: AssistantState):
        """Main chatbot node that processes messages with assistant context."""
        messages = state["messages"]

        # Add system prompt if this is the first message or if there's no system message
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    graph_builder = StateGraph(AssistantState)

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


async def invoke_general_assistant(
    message: str,
    thread_id: str,
    model_name: str = None,
    temperature: float = 0.7,
):
    """
    Convenience function to invoke the general assistant chain.

    Args:
        message: User's message/question
        thread_id: Thread ID for conversation continuity
        model_name: Optional model override
        temperature: Temperature for response generation

    Returns:
        The assistant's response text
    """
    # Create the chain
    chain = await create_general_assistant_chain(
        model_name=model_name,
        temperature=temperature
    )

    # Prepare inputs
    inputs = {"messages": [HumanMessage(content=message)]}
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke the chain
    result = await chain.ainvoke(inputs, config=config)

    # Extract the last message
    last_message = result["messages"][-1]
    return last_message.content


# For testing
if __name__ == "__main__":
    import asyncio

    async def test_chain():
        """Test the general assistant chain."""
        print("Testing General Assistant Chain...\n")

        # Test 1: Calculator
        print("Test 1: Calculator")
        response = await invoke_general_assistant(
            message="What is 123 * 456?",
            thread_id="test-thread-1",
            model_name="gemini-2.5-flash"
        )
        print(f"Response: {response}\n")

        # Test 2: Unit conversion
        print("Test 2: Unit Conversion")
        response = await invoke_general_assistant(
            message="Convert 100 degrees fahrenheit to celsius",
            thread_id="test-thread-2",
            model_name="gemini-2.5-flash"
        )
        print(f"Response: {response}\n")

        # Test 3: Current time
        print("Test 3: Current Time")
        response = await invoke_general_assistant(
            message="What time is it?",
            thread_id="test-thread-3",
            model_name="gemini-2.5-flash"
        )
        print(f"Response: {response}\n")

    # Run test
    asyncio.run(test_chain())
