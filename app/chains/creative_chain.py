# app/chains/creative_chain.py
"""
Creative Assistant Chain using LangGraph.

This module provides a specialized AI creative partner for brainstorming,
project planning, and creative development.
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

SYSTEM_PROMPT = """You are an expert Creative Assistant AI. Your goal is to help users unleash their creativity, develop projects, and generate innovative ideas.

Your capabilities include:
- Brainstorming and idea generation
- Creative project planning and development
- Providing inspiration and creative prompts
- Helping overcome creative blocks
- **Logging Creative Projects**: You can save creative projects and ideas
- **Tracking Creative Progress**: You can retrieve past projects to monitor creative journey

When interacting with users:
- Be imaginative, enthusiastic, and encouraging
- Think outside the box and suggest unique perspectives
- Help users develop and refine their creative ideas
- Celebrate creativity in all its forms

Remember: Creativity is a skill that grows with practice. Every idea has potential!
- **Managing Profile**: You can view and update the user's creative profile (fields, skills, goals, etc.)"""


class CreativeState(TypedDict):
    """State for the creative assistant conversation."""
    messages: Annotated[List[BaseMessage], operator.add]


@tool
async def log_creative_project(
    project_name: str,
    project_type: str,
    config: RunnableConfig,
    description: str = None,
    status: str = "planning",
    duration_minutes: int = None,
    ideas: List[str] = None
) -> str:
    """
    Log a creative project.

    Args:
        project_name: Name of the project
        project_type: Type of project ("WRITING", "ART", "MUSIC", "DESIGN", "VIDEO", "PHOTOGRAPHY", "OTHER")
        description: Project description
        status: Current status ("planning", "in_progress", "completed")
        duration_minutes: Time spent on the project
        ideas: List of ideas or concepts for the project
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot log project."

    # Build ideas array if provided
    ideas_array = []
    if ideas:
        for idea in ideas:
            ideas_array.append({"idea_title": idea, "idea_description": idea})

        for idea in ideas:
            ideas_array.append({"idea_title": idea, "idea_description": idea})

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/creative/project"
    payload = {
        "userId": user_id,
        "project_name": project_name,
        "project_type": project_type,
        "description": description,
        "status": status,
        "duration_minutes": duration_minutes,
        "ideas": ideas_array if ideas_array else None
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return f"Successfully logged creative project '{project_name}'."
                else:
                    error_text = await response.text()
                    return f"Failed to log project. Status: {response.status}, Error: {error_text}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def get_creative_history(config: RunnableConfig) -> str:
    """
    Retrieve the user's creative project history.
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve history."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/creative/project?userId={user_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    projects = await response.json()
                    if not projects:
                        return "No creative project history found."

                    history_str = "Recent Creative Projects:\n"
                    for p in projects[:5]:
                        date = p['date'].split('T')[0]
                        status = p.get('status', 'unknown')
                        history_str += f"- {date}: {p['project_name']} ({p['project_type']}) - Status: {status}\n"
                    return history_str
                else:
                    return f"Failed to fetch history. Status: {response.status}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def get_creative_profile(config: RunnableConfig) -> str:
    """
    Retrieve the user's creative profile (fields, skills, goals).
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve profile."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/creative?userId={user_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    profile = await response.json()
                    return f"""Creative Profile:
Fields: {profile.get('creative_fields', 'Not set')}
Skill Level: {profile.get('skill_level', 'Not set')}
Goals: {profile.get('goals', 'Not set')}
Preferences: {profile.get('preferences', 'Not set')}
"""
                else:
                    return f"Failed to fetch profile. Status: {response.status}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def update_creative_profile(
    config: RunnableConfig,
    creative_fields: str = None,
    skill_level: str = None,
    goals: str = None,
    preferences: str = None
) -> str:
    """
    Create or update the user's creative profile.
    
    Args:
        creative_fields: Fields of interest (writing, art, etc.)
        skill_level: e.g., "beginner", "intermediate"
        goals: Creative goals
        preferences: Preferences
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot update profile."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/creative"
    
    payload = {"userId": user_id}
    if creative_fields is not None: payload["creative_fields"] = creative_fields
    if skill_level is not None: payload["skill_level"] = skill_level
    if goals is not None: payload["goals"] = goals
    if preferences is not None: payload["preferences"] = preferences

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return "Successfully updated creative profile."
                else:
                    error_text = await response.text()
                    return f"Failed to update profile. Status: {response.status}, Error: {error_text}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


async def create_creative_chain(
    model_name: str = None,
    temperature: float = 0.9,  # Higher temperature for more creativity
):
    """
    Creates and compiles a LangGraph-based creative assistant chain.
    """
    llm = get_llm(model_name, temperature)

    local_tools = [
        log_creative_project,
        get_creative_history,
        get_creative_profile,
        update_creative_profile,
    ]

    try:
        mcp_tools = await get_mcp_tools()
        all_tools = local_tools + mcp_tools
        print(f"Creative Assistant loaded with {len(local_tools)} local tools and {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
        all_tools = local_tools

    llm_with_tools = llm.bind_tools(all_tools)

    def chatbot(state: CreativeState, config):
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

    graph_builder = StateGraph(CreativeState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(all_tools))
    graph_builder.set_entry_point("chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import aiosqlite

    conn = await aiosqlite.connect("creative_chat.db", check_same_thread=False)
    memory = AsyncSqliteSaver(conn)
    graph = graph_builder.compile(checkpointer=memory)

    return graph
