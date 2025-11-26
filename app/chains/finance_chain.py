# app/chains/finance_chain.py
"""
Finance Agent Chain using LangGraph.

This module provides a specialized AI financial advisor for budgeting,
expense tracking, and investment guidance.
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

# Import the model factory from agents
from app.agents.langgraph_agent import get_llm
from app.mcp_client import get_mcp_tools
import operator

# Load environment variables
load_dotenv()

# System prompt for the finance advisor persona
SYSTEM_PROMPT = """You are an expert Financial Advisor AI. Your goal is to help users manage their finances, create budgets, track expenses, and make informed financial decisions.

Your capabilities include:
- Creating personalized budgets based on income and expenses
- Tracking income and expenses by category
- Providing investment advice and financial planning guidance
- Calculating savings rates, budgets, and financial metrics
- **Logging Transactions**: You can save the user's income and expenses to their profile
- **Tracking Financial Progress**: You can retrieve past transactions to monitor spending patterns

When interacting with users:
- Be professional, clear, and financially responsible
- Ask clarifying questions about their financial goals and situation
- Tailor your advice to their specific income level and goals
- Use the available tools to provide accurate calculations and **save their financial data**
- Focus on long-term financial health and sustainable habits

Disclaimer: This is for informational purposes only. Users should consult with a licensed financial advisor for personalized investment advice."""


# Define the agent state
class FinanceState(TypedDict):
    """State for the finance advisor conversation."""
    messages: Annotated[List[BaseMessage], operator.add]


# Define finance utility tools
@tool
def calculate_savings_rate(monthly_income: float, monthly_expenses: float) -> str:
    """
    Calculate savings rate as a percentage.

    Args:
        monthly_income: Total monthly income
        monthly_expenses: Total monthly expenses

    Returns:
        Savings rate percentage and amount
    """
    try:
        savings = monthly_income - monthly_expenses
        if monthly_income > 0:
            rate = (savings / monthly_income) * 100
            return f"Savings: ${savings:.2f}/month ({rate:.1f}%)"
        else:
            return "Error: Income must be greater than 0"
    except Exception as e:
        return f"Error calculating savings rate: {str(e)}"


@tool
def budget_recommendation(monthly_income: float) -> str:
    """
    Provide 50/30/20 budget recommendation.

    Args:
        monthly_income: Total monthly income

    Returns:
        Recommended budget breakdown
    """
    try:
        needs = monthly_income * 0.50
        wants = monthly_income * 0.30
        savings = monthly_income * 0.20

        return f"""50/30/20 Budget for ${monthly_income:.2f}/month:
Needs (50%): ${needs:.2f} - Housing, utilities, groceries, transportation
Wants (30%): ${wants:.2f} - Entertainment, dining out, hobbies
Savings (20%): ${savings:.2f} - Emergency fund, retirement, investments"""
    except Exception as e:
        return f"Error calculating budget: {str(e)}"


@tool
async def log_transaction(
    transaction_type: str,
    amount: float,
    category: str,
    config: RunnableConfig,
    description: str = None,
    date: str = None
) -> str:
    """
    Log a financial transaction (income or expense).

    Args:
        transaction_type: Either "INCOME" or "EXPENSE"
        amount: Transaction amount
        category: Category (e.g., "Salary", "Groceries", "Rent")
        description: Optional description
        date: Optional date (YYYY-MM-DD format)
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot log transaction."

    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    url = f"{frontend_url}/api/user/finance/transaction"
    payload = {
        "userId": user_id,
        "type": transaction_type,
        "amount": amount,
        "category": category,
        "description": description,
        "date": date
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return f"Successfully logged {transaction_type.lower()} of ${amount} in category '{category}'."
                else:
                    error_text = await response.text()
                    return f"Failed to log transaction. Status: {response.status}, Error: {error_text}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


@tool
async def get_transaction_history(config: RunnableConfig) -> str:
    """
    Retrieve the user's transaction history.
    """
    import aiohttp

    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve history."

    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    url = f"{frontend_url}/api/user/finance/transaction?userId={user_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    transactions = await response.json()
                    if not transactions:
                        return "No transaction history found."

                    history_str = "Recent Transactions:\n"
                    for t in transactions[:10]:  # Show last 10
                        date = t['date'].split('T')[0]
                        history_str += f"- {date}: {t['type']} ${t['amount']:.2f} ({t['category']})\n"
                    return history_str
                else:
                    return f"Failed to fetch history. Status: {response.status}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


async def create_finance_chain(
    model_name: str = None,
    temperature: float = 0.7,
):
    """
    Creates and compiles a LangGraph-based finance advisor chain.

    Args:
        model_name: Name of the model to use
        temperature: Temperature setting for the model (0-1)

    Returns:
        A compiled LangGraph agent with memory and finance tools
    """
    # Initialize the model
    llm = get_llm(model_name, temperature)

    # Load local tools
    local_tools = [
        calculate_savings_rate,
        budget_recommendation,
        log_transaction,
        get_transaction_history,
    ]

    # Load MCP tools if configured
    try:
        mcp_tools = await get_mcp_tools()
        all_tools = local_tools + mcp_tools
        print(f"Finance Advisor loaded with {len(local_tools)} local tools and {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
        all_tools = local_tools

    # Bind tools to the model
    llm_with_tools = llm.bind_tools(all_tools)

    # Define the chatbot node with system prompt
    def chatbot(state: FinanceState, config):
        """Main chatbot node that processes messages with finance context."""
        messages = state["messages"]
        user_id = config.get("configurable", {}).get("user_id")

        # Construct the system prompt with current user_id
        prompt_content = SYSTEM_PROMPT
        if user_id:
            prompt_content += f"\n\nCurrent User ID: {user_id}\nUse this ID for any tools that require a user_id."

        system_message = SystemMessage(content=prompt_content)

        # Ensure the system message is always the first message and up-to-date
        if messages and isinstance(messages[0], SystemMessage):
            messages = [system_message] + messages[1:]
        else:
            messages = [system_message] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    graph_builder = StateGraph(FinanceState)

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

    # Compile with AsyncSqliteSaver for conversation persistence
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import aiosqlite

    # Use a file-based SQLite database for persistence
    conn = await aiosqlite.connect("finance_chat.db", check_same_thread=False)
    memory = AsyncSqliteSaver(conn)
    graph = graph_builder.compile(checkpointer=memory)

    return graph
