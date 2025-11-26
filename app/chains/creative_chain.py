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

Remember: Creativity is a skill that grows with practice. Every idea has potential!"""


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

    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
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
            async with session.post(url, json=payload) as response:
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

    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    url = f"{frontend_url}/api/user/creative/project?userId={user_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
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
