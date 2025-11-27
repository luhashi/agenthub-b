# app/chains/career_chain.py
"""
Career Coach Chain using LangGraph.

This module provides a specialized AI career coach for job search,
skill development, and professional growth.
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

SYSTEM_PROMPT = """You are an expert Career Coach AI. Your goal is to help users navigate their career journey, find opportunities, and develop professionally.

Your capabilities include:
- Resume and cover letter guidance
- Interview preparation and tips
- Career path planning and exploration
- Skill development recommendations
- **Logging Career Activities**: You can save job applications, interviews, and skill development
- **Tracking Career Progress**: You can retrieve past activities to monitor job search progress

When interacting with users:
- Be supportive, motivating, and professional
- Provide actionable advice and specific recommendations
- Help users identify their strengths and areas for growth
- Tailor guidance to their industry and career goals

Remember: Every career journey is unique. Help users discover and pursue their professional aspirations.
- **Managing Profile**: You can view and update the user's career profile (role, industry, skills, goals, etc.)"""


class CareerState(TypedDict):
    """State for the career coach conversation."""
    messages: Annotated[List[BaseMessage], operator.add]


@tool
async def log_career_activity(
    activity_type: str,
    config: RunnableConfig,
    company: str = None,
    position: str = None,
    description: str = None,
    outcome: str = None,
    notes: str = None
) -> str:
    """
    Log a career activity.

    Args:
        activity_type: Type of activity ("APPLICATION", "INTERVIEW", "NETWORKING", "SKILL_DEVELOPMENT", "CERTIFICATION", "PROJECT")
        company: Company name (for applications/interviews)
        position: Position/role name
        description: Activity description
        outcome: Outcome status ("pending", "accepted", "rejected")
        notes: Additional notes
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot log activity."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/career/activity"
    payload = {
        "userId": user_id,
        "activity_type": activity_type,
        "company": company,
        "position": position,
        "description": description,
        "outcome": outcome,
        "notes": notes
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return f"Successfully logged {activity_type} activity."
                else:
                    error_text = await response.text()
                    return f"Failed to log activity. Status: {response.status}, Error: {error_text}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def get_career_history(config: RunnableConfig) -> str:
    """
    Retrieve the user's career activity history.
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve history."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/career/activity?userId={user_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    activities = await response.json()
                    if not activities:
                        return "No career activity history found."

                    history_str = "Recent Career Activities:\n"
                    for a in activities[:5]:
                        date = a['date'].split('T')[0]
                        company_info = f" at {a['company']}" if a.get('company') else ""
                        history_str += f"- {date}: {a['activity_type']}{company_info}\n"
                    return history_str
                else:
                    return f"Failed to fetch history. Status: {response.status}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def get_career_profile(config: RunnableConfig) -> str:
    """
    Retrieve the user's career profile (role, industry, skills, goals).
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve profile."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/career?userId={user_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    profile = await response.json()
                    return f"""Career Profile:
Current Role: {profile.get('current_role', 'Not set')}
Target Role: {profile.get('target_role', 'Not set')}
Industry: {profile.get('industry', 'Not set')}
Experience: {profile.get('experience_years', 0)} years
Skills: {profile.get('skills', 'Not set')}
Career Goals: {profile.get('career_goals', 'Not set')}
"""
                else:
                    return f"Failed to fetch profile. Status: {response.status}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def update_career_profile(
    config: RunnableConfig,
    current_role: str = None,
    target_role: str = None,
    industry: str = None,
    experience_years: float = None,
    skills: str = None,
    career_goals: str = None
) -> str:
    """
    Create or update the user's career profile.
    
    Args:
        current_role: Current job title
        target_role: Desired job title
        industry: Industry sector
        experience_years: Years of experience
        skills: Skills list
        career_goals: Description of goals
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot update profile."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/career"
    
    payload = {"userId": user_id}
    if current_role is not None: payload["current_role"] = current_role
    if target_role is not None: payload["target_role"] = target_role
    if industry is not None: payload["industry"] = industry
    if experience_years is not None: payload["experience_years"] = experience_years
    if skills is not None: payload["skills"] = skills
    if career_goals is not None: payload["career_goals"] = career_goals

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return "Successfully updated career profile."
                else:
                    error_text = await response.text()
                    return f"Failed to update profile. Status: {response.status}, Error: {error_text}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


async def create_career_chain(
    model_name: str = None,
    temperature: float = 0.7,
):
    """
    Creates and compiles a LangGraph-based career coach chain.
    """
    llm = get_llm(model_name, temperature)

    local_tools = [
        log_career_activity,
        get_career_history,
        get_career_profile,
        update_career_profile,
    ]

    try:
        mcp_tools = await get_mcp_tools()
        all_tools = local_tools + mcp_tools
        print(f"Career Coach loaded with {len(local_tools)} local tools and {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
        all_tools = local_tools

    llm_with_tools = llm.bind_tools(all_tools)

    def chatbot(state: CareerState, config):
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

    graph_builder = StateGraph(CareerState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(all_tools))
    graph_builder.set_entry_point("chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import aiosqlite

    conn = await aiosqlite.connect("career_chat.db", check_same_thread=False)
    memory = AsyncSqliteSaver(conn)
    graph = graph_builder.compile(checkpointer=memory)

    return graph
