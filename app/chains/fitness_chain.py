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
    Log a completed workout session to the user's profile directly to the database.
    """
    import asyncpg
    import uuid
    from datetime import datetime
    
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot log workout."

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return "Error: Database connection not configured."
    
    try:
        # Force SSL mode for Neon DB
        conn = await asyncpg.connect(database_url, ssl='require')
        try:
            # 1. Get or Create FitnessProfile
            profile_id = await conn.fetchval(
                'SELECT id FROM "FitnessProfile" WHERE user_clerk_id = $1',
                user_id
            )
            
            if not profile_id:
                profile_id = str(uuid.uuid4())
                await conn.execute(
                    '''
                    INSERT INTO "FitnessProfile" (id, user_clerk_id, "createdAt", "updatedAt")
                    VALUES ($1, $2, NOW(), NOW())
                    ''',
                    profile_id, user_id
                )
            
            # 2. Create WorkoutSession
            workout_id = str(uuid.uuid4())
            await conn.execute(
                '''
                INSERT INTO "WorkoutSession" (id, fitness_profile_id, name, date, duration_minutes, notes, "createdAt", "updatedAt")
                VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
                ''',
                workout_id, profile_id, name, datetime.now(), duration_minutes, notes
            )
            
            # 3. Create Exercises and Sets
            for ex in exercises:
                ex_name = ex.get('name')
                ex_notes = ex.get('notes')
                
                ex_id = str(uuid.uuid4())
                await conn.execute(
                    '''
                    INSERT INTO "WorkoutExercise" (id, workout_session_id, exercise_name, notes, "createdAt", "updatedAt")
                    VALUES ($1, $2, $3, $4, NOW(), NOW())
                    ''',
                    ex_id, workout_id, ex_name, ex_notes
                )
                
                sets = ex.get('sets', [])
                for s in sets:
                    set_num = s.get('set_number')
                    reps = s.get('reps')
                    weight = s.get('weight')
                    rpe = s.get('rpe')
                    
                    set_id = str(uuid.uuid4())
                    await conn.execute(
                        '''
                        INSERT INTO "WorkoutSet" (id, workout_exercise_id, set_number, reps, weight, rpe, "createdAt", "updatedAt")
                        VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
                        ''',
                        set_id, ex_id, set_num, reps, float(weight) if weight else None, float(rpe) if rpe else None
                    )
            
            # 4. Update Profile Stats (Last Workout)
            await conn.execute(
                'UPDATE "FitnessProfile" SET last_workout = $1 WHERE id = $2',
                datetime.now(), profile_id
            )
            
            return f"Successfully logged workout '{name}' to database."
            
        finally:
            await conn.close()
            
    except Exception as e:
        return f"Error logging workout to database: {str(e)}"

@tool
async def get_workout_history(config: RunnableConfig) -> str:
    """
    Retrieve the user's past workout history from the database.
    """
    import asyncpg
    
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Error: User ID not found in context. Cannot retrieve history."

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return "Error: Database connection not configured."
    
    try:
        # Force SSL mode for Neon DB
        conn = await asyncpg.connect(database_url, ssl='require')
        try:
            # Get fitness profile
            profile = await conn.fetchrow(
                'SELECT id FROM "FitnessProfile" WHERE user_clerk_id = $1',
                user_id
            )
            
            if not profile:
                return "No workout history found. Start logging your workouts!"
            
            # Get recent workouts with exercises and sets
            workouts = await conn.fetch('''
                SELECT ws.id, ws.name, ws.date, ws.duration_minutes,
                       json_agg(
                           json_build_object(
                               'exercise_name', we.exercise_name,
                               'sets', (
                                   SELECT json_agg(
                                       json_build_object(
                                           'set_number', s.set_number,
                                           'reps', s.reps,
                                           'weight', s.weight
                                       ) ORDER BY s.set_number
                                   )
                                   FROM "WorkoutSet" s
                                   WHERE s.workout_exercise_id = we.id
                               )
                           ) ORDER BY we.id
                       ) as exercises
                FROM "WorkoutSession" ws
                LEFT JOIN "WorkoutExercise" we ON we.workout_session_id = ws.id
                WHERE ws.fitness_profile_id = $1
                GROUP BY ws.id, ws.name, ws.date, ws.duration_minutes
                ORDER BY ws.date DESC
                LIMIT 5
            ''', profile['id'])
            
            if not workouts:
                return "No workout history found."
            
            history_str = "Recent Workouts:\n"
            for w in workouts:
                date_str = w['date'].strftime('%Y-%m-%d')
                duration = f" ({w['duration_minutes']} min)" if w['duration_minutes'] else ""
                history_str += f"- {date_str}: {w['name']}{duration}\n"
                
                if w['exercises']:
                    for ex in w['exercises']:
                        if ex and ex.get('exercise_name'):
                            sets_count = len(ex.get('sets', [])) if ex.get('sets') else 0
                            history_str += f"  * {ex['exercise_name']}: {sets_count} sets\n"
            
            return history_str
            
        finally:
            await conn.close()
            
    except Exception as e:
        return f"Error retrieving workout history: {str(e)}"


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
