# app/chains/education_chain.py
"""
Study Assistant Chain using LangGraph.

This module provides a specialized AI study assistant for learning,
exam preparation, and educational guidance.
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

SYSTEM_PROMPT = """You are an expert Study Assistant AI. Your goal is to help users learn effectively, prepare for exams, and develop strong study habits.

Your capabilities include:
- Creating personalized study plans and schedules
- Explaining complex concepts in simple terms
- Generating practice questions and quizzes
- Providing study techniques and learning strategies
- **Logging Study Sessions**: You can save the user's study sessions to their profile
- **Tracking Learning Progress**: You can retrieve past sessions to monitor progress

When interacting with users:
- Be encouraging, patient, and supportive
- Break down complex topics into manageable chunks
- Use examples and analogies to aid understanding
- Ask questions to assess comprehension
- Adapt your teaching style to the user's learning preferences

Remember: Learning is a journey. Celebrate progress and help users develop effective study habits.
- **Managing Profile**: You can view and update the user's education profile (level, subjects, goals, etc.)"""


class EducationState(TypedDict):
    """State for the study assistant conversation."""
    messages: Annotated[List[BaseMessage], operator.add]


@tool
async def log_study_session(
    subject: str,
    topic: str,
    duration_minutes: int,
    config: RunnableConfig,
    session_type: str = "reading",
    notes: str = None,
    comprehension_level: int = None
) -> str:
    """
    Log a study session.

    Args:
        subject: Subject studied (e.g., "Mathematics", "History")
        topic: Specific topic (e.g., "Calculus", "World War II")
        duration_minutes: Duration in minutes
        session_type: Type of session ("reading", "practice", "review")
        notes: Optional notes about the session
        comprehension_level: Self-assessment (1-10 scale)
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot log session."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/education/session"
    payload = {
        "userId": user_id,
        "subject": subject,
        "topic": topic,
        "duration_minutes": duration_minutes,
        "session_type": session_type,
        "notes": notes,
        "comprehension_level": comprehension_level
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return f"Successfully logged {duration_minutes}-minute study session on {subject}: {topic}."
                else:
                    error_text = await response.text()
                    return f"Failed to log session. Status: {response.status}, Error: {error_text}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def get_study_history(config: RunnableConfig) -> str:
    """
    Retrieve the user's study session history.
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve history."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/education/session?userId={user_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    sessions = await response.json()
                    if not sessions:
                        return "No study history found."

                    history_str = "Recent Study Sessions:\n"
                    for s in sessions[:5]:
                        date = s['date'].split('T')[0]
                        history_str += f"- {date}: {s['subject']} - {s['topic']} ({s['duration_minutes']} min)\n"
                    return history_str
                else:
                    return f"Failed to fetch history. Status: {response.status}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def get_education_profile(config: RunnableConfig) -> str:
    """
    Retrieve the user's education profile (level, subjects, goals).
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve profile."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/education?userId={user_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    profile = await response.json()
                    return f"""Education Profile:
Level: {profile.get('current_level', 'Not set')}
Subjects: {profile.get('subjects', 'Not set')}
Learning Style: {profile.get('learning_style', 'Not set')}
Study Goals: {profile.get('study_goals', 'Not set')}
"""
                else:
                    return f"Failed to fetch profile. Status: {response.status}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def update_education_profile(
    config: RunnableConfig,
    current_level: str = None,
    subjects: str = None,
    learning_style: str = None,
    study_goals: str = None
) -> str:
    """
    Create or update the user's education profile.
    
    Args:
        current_level: e.g., "high school", "undergraduate"
        subjects: Subjects being studied
        learning_style: e.g., "visual", "auditory"
        study_goals: Description of goals
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot update profile."

    frontend_url = os.getenv("FRONTEND_URL", "https://agenthub-omega.vercel.app")
    api_secret = os.getenv("GRAPHASH_API_SECRET", "default_secret")
    headers = {"x-api-secret": api_secret}

    url = f"{frontend_url}/api/user/education"
    
    payload = {"userId": user_id}
    if current_level is not None: payload["current_level"] = current_level
    if subjects is not None: payload["subjects"] = subjects
    if learning_style is not None: payload["learning_style"] = learning_style
    if study_goals is not None: payload["study_goals"] = study_goals

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return "Successfully updated education profile."
                else:
                    error_text = await response.text()
                    return f"Failed to update profile. Status: {response.status}, Error: {error_text}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


async def create_education_chain(
    model_name: str = None,
    temperature: float = 0.7,
):
    """
    Creates and compiles a LangGraph-based study assistant chain.
    """
    llm = get_llm(model_name, temperature)

    local_tools = [
        log_study_session,
        get_study_history,
        get_education_profile,
        update_education_profile,
    ]

    try:
        mcp_tools = await get_mcp_tools()
        all_tools = local_tools + mcp_tools
        print(f"Study Assistant loaded with {len(local_tools)} local tools and {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
        all_tools = local_tools

    llm_with_tools = llm.bind_tools(all_tools)

    def chatbot(state: EducationState, config):
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

    graph_builder = StateGraph(EducationState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(all_tools))
    graph_builder.set_entry_point("chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import aiosqlite

    conn = await aiosqlite.connect("education_chat.db", check_same_thread=False)
    memory = AsyncSqliteSaver(conn)
    graph = graph_builder.compile(checkpointer=memory)

    return graph
