# app/chains/fitness_chain.py
"""
Personal Trainer Chain using LangGraph.

This module provides a specialized AI fitness coach for workout plans,
nutrition advice, and health guidance.
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

# System prompt for the fitness trainer persona
SYSTEM_PROMPT = """You are an expert Personal Trainer and Nutritionist AI. Your goal is to help users achieve their fitness and health goals through personalized advice, workout plans, and nutritional guidance.

Your capabilities include:
- Creating personalized workout routines (strength, cardio, flexibility, etc.)
- Providing nutritional advice and meal planning suggestions
- Explaining proper exercise form and technique
- Helping with motivation and habit formation
- Calculating fitness metrics (BMI, BMR, TDEE, macros)
- **Logging Workouts**: You can save the user's workout details (exercises, sets, reps, weight) to their profile.
- **Tracking Progress**: You can retrieve past workouts to monitor progression and suggest weight increases.

When interacting with users:
- Be encouraging, motivating, and positive
- Prioritize safety and proper form
- Ask clarifying questions about their fitness level, goals, and limitations
- Tailor your advice to their specific needs and equipment availability
- Use the available tools to provide accurate calculations and **save their progress**.

Disclaimer: Always remind users to consult with a healthcare professional before starting any new diet or exercise program, especially if they have pre-existing conditions."""


# Define the agent state
class FitnessState(TypedDict):
    """State for the fitness trainer conversation."""
    messages: Annotated[List[BaseMessage], operator.add]


# Define fitness utility tools
@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """
    Calculate Body Mass Index (BMI).

    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters

    Returns:
        BMI value and category
    """
    try:
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        
        category = ""
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
        elif 25 <= bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
            
        return f"BMI: {bmi:.1f} ({category})"
    except Exception as e:
        return f"Error calculating BMI: {str(e)}"

@tool
def calculate_macros(tdee: float, goal: str = "maintenance") -> str:
    """
    Calculate recommended macronutrient split based on TDEE and goal.

    Args:
        tdee: Total Daily Energy Expenditure in calories
        goal: Fitness goal ("maintenance", "loss", "gain")

    Returns:
        Recommended grams of Protein, Carbs, and Fats
    """
    try:
        calories = tdee
        if goal == "loss":
            calories = tdee - 500
        elif goal == "gain":
            calories = tdee + 500
            
        # Standard split: 30% Protein, 35% Carbs, 35% Fats
        protein_cals = calories * 0.30
        carb_cals = calories * 0.35
        fat_cals = calories * 0.35
        
        protein_g = protein_cals / 4
        carb_g = carb_cals / 4
        fat_g = fat_cals / 9
        
        return f"""Daily Targets for {goal}:
Calories: {int(calories)} kcal
Protein: {int(protein_g)}g
Carbs: {int(carb_g)}g
Fats: {int(fat_g)}g"""
    except Exception as e:
        return f"Error calculating macros: {str(e)}"

@tool
def get_exercise_info(exercise_name: str) -> str:
    """
    Get information about a specific exercise.

    Args:
        exercise_name: Name of the exercise (e.g., "squat", "bench press")

    Returns:
        Description of muscles worked and basic form tips
    """
    # Placeholder - in production this could query an exercise database
    return f"""Information for '{exercise_name}':
    
[This is a placeholder. In a real app, this would query an exercise database API.]

General tips for {exercise_name}:
- Focus on controlled movement
- Maintain proper posture
- Breathe rhythmically
- Start with lighter weights to master form"""

@tool
async def log_workout(name: str, exercises: List[dict], config: RunnableConfig, duration_minutes: int = None, notes: str = None) -> str:
    """
    Log a completed workout session to the user's profile.

    Args:
        name: Name of the workout (e.g., "Leg Day")
        exercises: List of exercises. Each exercise is a dict with:
                   - name: str
                   - notes: str (optional)
                   - sets: List[dict] where each set has:
                           - set_number: int
                           - reps: int
                           - weight: float (kg)
                           - rpe: float (optional)
        duration_minutes: Total duration in minutes (optional)
        notes: General notes about the session (optional)
    """
    import aiohttp
    
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot log workout."

    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    url = f"{frontend_url}/api/user/fitness/workout"
    payload = {
        "userId": user_id,
        "name": name,
        "exercises": exercises,
        "duration_minutes": duration_minutes,
        "notes": notes
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return f"Successfully logged workout '{name}'."
                else:
                    error_text = await response.text()
                    return f"Failed to log workout. Status: {response.status}, Error: {error_text}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"

@tool
async def get_workout_history(config: RunnableConfig) -> str:
    """
    Retrieve the user's past workout history.
    """
    import aiohttp
    
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve history."

    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    url = f"{frontend_url}/api/user/fitness/workout?userId={user_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    workouts = await response.json()
                    if not workouts:
                        return "No workout history found."
                    
                    history_str = "Recent Workouts:\n"
                    for w in workouts[:5]: # Show last 5
                        history_str += f"- {w['date'].split('T')[0]}: {w['name']} ({w['duration_minutes']} min)\n"
                        for ex in w['exercises']:
                            history_str += f"  * {ex['exercise_name']}: {len(ex['sets'])} sets\n"
                    return history_str
                else:
                    return f"Failed to fetch history. Status: {response.status}"
    except Exception as e:
        return f"Error connecting to database API: {str(e)}"


async def create_fitness_chain(
    model_name: str = None,
    temperature: float = 0.7,
):
    """
    Creates and compiles a LangGraph-based fitness trainer chain.

    Args:
        model_name: Name of the model to use
        temperature: Temperature setting for the model (0-1)

    Returns:
        A compiled LangGraph agent with memory and fitness tools
    """
    # Initialize the model
    llm = get_llm(model_name, temperature)

    # Load local tools
    local_tools = [
        calculate_bmi,
        calculate_macros,
        get_exercise_info,
        log_workout,
        get_workout_history,
    ]

    # Load MCP tools if configured
    try:
        mcp_tools = await get_mcp_tools()
        all_tools = local_tools + mcp_tools
        print(f"Fitness Trainer loaded with {len(local_tools)} local tools and {len(mcp_tools)} MCP tools")
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
        all_tools = local_tools

    # Bind tools to the model
    llm_with_tools = llm.bind_tools(all_tools)

    # Define the chatbot node with system prompt
    def chatbot(state: FitnessState, config):
        """Main chatbot node that processes messages with fitness context."""
        messages = state["messages"]
        user_id = config.get("configurable", {}).get("user_id")

        # Construct the system prompt with current user_id
        prompt_content = SYSTEM_PROMPT
        if user_id:
            prompt_content += f"\n\nCurrent User ID: {user_id}\nUse this ID for any tools that require a user_id."
        
        system_message = SystemMessage(content=prompt_content)

        # Ensure the system message is always the first message and up-to-date
        if messages and isinstance(messages[0], SystemMessage):
            # Replace the existing system message with the new one
            messages = [system_message] + messages[1:]
        else:
            # Prepend the system message
            messages = [system_message] + messages
        
        # If we have a user_id in config, we might want to ensure tools get it
        # But for now, relying on the LLM to pick it up from the system prompt is the standard way
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    graph_builder = StateGraph(FitnessState)

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
    # We manually create the connection to keep it open
    db_path = os.getenv("DB_PATH", "fitness_chat.db")
    conn = await aiosqlite.connect(db_path, check_same_thread=False)
    memory = AsyncSqliteSaver(conn)
    graph = graph_builder.compile(checkpointer=memory)

    return graph
