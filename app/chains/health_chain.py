# app/chains/health_chain.py
"""
Health Monitor Chain using LangGraph.

This module provides a specialized AI health companion for wellness tracking,
health metrics, and lifestyle guidance.
"""

import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.agents.langgraph_agent import get_llm
from app.mcp_client import get_mcp_tools
import operator

load_dotenv()

SYSTEM_PROMPT = """You are an expert Health Monitor AI. Your goal is to help users track their wellness, maintain healthy habits, and monitor important health metrics.

Your capabilities include:
- Tracking daily health metrics (sleep, water intake, steps, etc.)
- Providing wellness advice and healthy habit recommendations
- Monitoring mood and energy levels
- Setting and tracking health goals
- **Logging Health Data**: You can save daily health logs to the user's profile
- **Tracking Health Trends**: You can retrieve past logs to monitor patterns

When interacting with users:
- Be supportive, non-judgmental, and encouraging
- Focus on sustainable, healthy lifestyle changes
- Help users understand their health patterns
- Celebrate progress and small wins

Disclaimer: This is for informational purposes only. Always consult healthcare professionals for medical advice, diagnosis, or treatment.
- **Managing Profile**: You can view and update the user's health profile (goals, conditions, medications, etc.)"""


class HealthState(TypedDict):
    """State for the health monitor conversation."""
    messages: Annotated[List[BaseMessage], operator.add]


@tool
async def log_health_entry(
    config: RunnableConfig,
    mood: str = None,
    energy_level: int = None,
    sleep_hours: float = None,
    water_glasses: int = None,
    steps: int = None,
    notes: str = None
) -> str:
    """
    Log daily health metrics.

    Args:
        mood: Mood description ("great", "good", "okay", "poor")
        energy_level: Energy level (1-10 scale)
        sleep_hours: Hours of sleep
        water_glasses: Glasses of water consumed
        steps: Step count for the day
        notes: Additional notes about health/wellness
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot log health entry."

    # Build metrics array
    metrics = []
    if sleep_hours is not None:
        metrics.append({"metric_type": "SLEEP", "value": sleep_hours, "unit": "hours"})
    if water_glasses is not None:
        metrics.append({"metric_type": "WATER", "value": water_glasses, "unit": "glasses"})
    if steps is not None:
        metrics.append({"metric_type": "STEPS", "value": steps, "unit": "steps"})

    if steps is not None:
        metrics.append({"metric_type": "STEPS", "value": steps, "unit": "steps"})

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/health/log"
    payload = {
        "userId": user_id,
        "mood": mood,
        "energy_level": energy_level,
        "notes": notes,
        "metrics": metrics if metrics else None
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return f"Successfully logged health entry."
                else:
                    error_text = await response.text()
                    return f"Failed to log entry. Status: {response.status}, Error: {error_text}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def get_health_history(config: RunnableConfig) -> str:
    """
    Retrieve the user's health log history.
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve history."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/health/log?userId={user_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    logs = await response.json()
                    if not logs:
                        return "No health history found."

                    history_str = "Recent Health Logs:\n"
                    for log in logs[:7]:  # Last 7 days
                        date = log['date'].split('T')[0]
                        mood = log.get('mood', 'N/A')
                        energy = log.get('energy_level', 'N/A')
                        history_str += f"- {date}: Mood: {mood}, Energy: {energy}/10\n"
                    return history_str
                else:
                    return f"Failed to fetch history. Status: {response.status}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def get_health_profile(config: RunnableConfig) -> str:
    """
    Retrieve the user's health profile (goals, conditions, medications).
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve profile."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/health?userId={user_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    profile = await response.json()
                    return f"""Health Profile:
Age: {profile.get('age', 'Not set')}
Health Goals: {profile.get('health_goals', 'Not set')}
Conditions: {profile.get('conditions', 'None')}
Medications: {profile.get('medications', 'None')}
Allergies: {profile.get('allergies', 'None')}
"""
                else:
                    return f"Failed to fetch profile. Status: {response.status}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def update_health_profile(
    config: RunnableConfig,
    age: int = None,
    health_goals: str = None,
    conditions: str = None,
    medications: str = None,
    allergies: str = None
) -> str:
    """
    Create or update the user's health profile.
    
    Args:
        age: User's age
        health_goals: Description of health goals
        conditions: Medical conditions
        medications: Current medications
        allergies: Known allergies
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot update profile."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/health"
    
    payload = {"userId": user_id}
    if age is not None: payload["age"] = age
    if health_goals is not None: payload["health_goals"] = health_goals
    if conditions is not None: payload["conditions"] = conditions
    if medications is not None: payload["medications"] = medications
    if allergies is not None: payload["allergies"] = allergies

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return "Successfully updated health profile."
                else:
                    error_text = await response.text()
                    return f"Failed to update profile. Status: {response.status}, Error: {error_text}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


async def create_health_chain(
    model_name: str = None,
    temperature: float = 0.7,
):
    """
    Creates and compiles a LangGraph-based health monitor chain.
    """
    llm = get_llm(model_name, temperature)

    local_tools = [
        log_health_entry,
        get_health_history,
        get_health_profile,
        update_health_profile,
    ]

    try:
        mcp_tools = await get_mcp_tools()
        all_tools = local_tools + mcp_tools
        print(f"Health Monitor loaded with {len(local_tools)} local tools and {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
        all_tools = local_tools

    llm_with_tools = llm.bind_tools(all_tools)

    def chatbot(state: HealthState, config):
        messages = state["messages"]
        user_id = config.get("configurable", {}).get("user_id")

        prompt_content = SYSTEM_PROMPT
        if user_id:
            prompt_content += f"\n\nCurrent User ID: {user_id}"

        system_message = SystemMessage(content=prompt_content)

        if messages and isinstance(messages[0], SystemMessage):
            messages = [system_message] + messages[1:]
        else:
            messages = [system_message] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph_builder = StateGraph(HealthState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(all_tools))
    graph_builder.set_entry_point("chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import aiosqlite

    conn = await aiosqlite.connect("health_chat.db", check_same_thread=False)
    memory = AsyncSqliteSaver(conn)
    graph = graph_builder.compile(checkpointer=memory)

    return graph
