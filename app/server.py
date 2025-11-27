from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schema import ChatRequest, ChatResponse, AgentListResponse, AgentData, Agent
from app.agents.langgraph_agent import create_agent
from app.chains.chat_chain import create_data_analyst_chain
from app.chains.general_assistant_chain import create_general_assistant_chain
from app.chains.fitness_chain import create_fitness_chain
from app.chains.finance_chain import create_finance_chain
from app.chains.education_chain import create_education_chain
from app.chains.career_chain import create_career_chain
from app.chains.health_chain import create_health_chain
from app.chains.creative_chain import create_creative_chain
from app.fourier.routes import router as fourier_router
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
import uuid
import os
from contextlib import asynccontextmanager

def extract_content(message) -> str:
    """
    Extract text content from a message, handling different formats.

    Gemini and other models may return content as:
    - A simple string
    - A list of content blocks with 'text' field
    """
    content = message.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract text from content blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict) and 'text' in block:
                text_parts.append(block['text'])
            elif isinstance(block, str):
                text_parts.append(block)
        return ''.join(text_parts)
    else:
        return str(content)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize agent cache for different models
    app.state.agent_cache = {}
    # Optionally pre-load default agent on startup
    default_model = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
    try:
        # Pre-load default agent
        app.state.agent_cache[default_model] = await create_agent(model_name=default_model)
        print(f"Pre-loaded default agent with model: {default_model}")
        
        # Pre-load fitness agent
        fitness_key = f"fitness:{default_model}:0.7"
        app.state.agent_cache[fitness_key] = await create_fitness_chain(model_name=default_model, temperature=0.7)
        print(f"Pre-loaded fitness agent: {fitness_key}")
    except Exception as e:
        print(f"Warning: Could not pre-load agents: {e}")
    yield
    # Clean up if needed

app = FastAPI(title="Serverless Agent Host", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include fourier router
app.include_router(fourier_router)

@app.get("/")
async def root():
    return {"status": "ok", "service": "serverless-agent-host"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        model_name = request.model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        temperature = request.temperature if request.temperature is not None else 0

        # Check agent cache
        if not hasattr(app.state, "agent_cache"):
            raise HTTPException(status_code=503, detail="Agent cache not initialized")

        # Create cache key (model + temperature)
        cache_key = f"{model_name}:{temperature}"

        # Get or create agent for this model
        if cache_key not in app.state.agent_cache:
            print(f"Creating new agent for model: {model_name} with temperature: {temperature}")
            try:
                app.state.agent_cache[cache_key] = await create_agent(
                    model_name=model_name,
                    temperature=temperature
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to create agent with model '{model_name}': {str(e)}"
                )

        agent = app.state.agent_cache[cache_key]

        # Prepare input for the agent
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the agent
        result = await agent.ainvoke(inputs, config=config)

        # Extract the last message
        last_message = result["messages"][-1]
        response_text = extract_content(last_message)

        return ChatResponse(
            response=response_text,
            thread_id=thread_id,
            model_used=model_name
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/analyst", response_model=ChatResponse)
async def chat_with_data_analyst(request: ChatRequest):
    """
    Chat endpoint specifically for the Data Analyst agent.

    This endpoint uses a specialized data analyst persona with access to
    tools for analyzing census data, health statistics, and more.
    """
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        model_name = request.model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        temperature = request.temperature if request.temperature is not None else 0.7  # Higher default for analyst

        # Check agent cache
        if not hasattr(app.state, "agent_cache"):
            raise HTTPException(status_code=503, detail="Agent cache not initialized")

        # Create cache key for data analyst (prefix with 'analyst:')
        cache_key = f"analyst:{model_name}:{temperature}"

        # Get or create data analyst agent
        if cache_key not in app.state.agent_cache:
            print(f"Creating new data analyst agent for model: {model_name} with temperature: {temperature}")
            try:
                app.state.agent_cache[cache_key] = await create_data_analyst_chain(
                    model_name=model_name,
                    temperature=temperature
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to create data analyst agent with model '{model_name}': {str(e)}"
                )

        agent = app.state.agent_cache[cache_key]

        # Prepare input for the agent
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the agent
        result = await agent.ainvoke(inputs, config=config)

        # Extract the last message
        last_message = result["messages"][-1]
        response_text = extract_content(last_message)

        return ChatResponse(
            response=response_text,
            thread_id=thread_id,
            model_used=f"{model_name} (Data Analyst)"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/assistant", response_model=ChatResponse)
async def chat_with_general_assistant(request: ChatRequest):
    """
    Chat endpoint for the General Purpose Assistant.

    This endpoint provides a helpful assistant with practical tools including:
    - Web search capabilities
    - Calculator and unit conversions
    - Current time/date
    - Weather information
    - Fun facts and general knowledge

    Perfect for general-purpose conversations and everyday tasks.
    """
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        model_name = request.model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        temperature = request.temperature if request.temperature is not None else 0.7  # Balanced default

        # Check agent cache
        if not hasattr(app.state, "agent_cache"):
            raise HTTPException(status_code=503, detail="Agent cache not initialized")

        # Create cache key for general assistant (prefix with 'assistant:')
        cache_key = f"assistant:{model_name}:{temperature}"

        # Get or create general assistant agent
        if cache_key not in app.state.agent_cache:
            print(f"Creating new general assistant for model: {model_name} with temperature: {temperature}")
            try:
                app.state.agent_cache[cache_key] = await create_general_assistant_chain(
                    model_name=model_name,
                    temperature=temperature
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to create general assistant with model '{model_name}': {str(e)}"
                )

        agent = app.state.agent_cache[cache_key]

        # Prepare input for the agent
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the agent
        result = await agent.ainvoke(inputs, config=config)

        # Extract the last message
        last_message = result["messages"][-1]
        response_text = extract_content(last_message)

        return ChatResponse(
            response=response_text,
            thread_id=thread_id,
            model_used=f"{model_name} (Assistant)"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/fitness", response_model=ChatResponse)
async def chat_with_fitness_trainer(request: ChatRequest):
    """
    Chat endpoint for the Personal Trainer.

    This endpoint provides a specialized fitness coach with tools for:
    - BMI calculation
    - Macro planning
    - Exercise information
    - Workout routines
    """
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        model_name = request.model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        temperature = request.temperature if request.temperature is not None else 0.7

        # Check agent cache
        if not hasattr(app.state, "agent_cache"):
            raise HTTPException(status_code=503, detail="Agent cache not initialized")

        # Create cache key for fitness trainer (prefix with 'fitness:')
        cache_key = f"fitness:{model_name}:{temperature}"

        # Get or create fitness trainer agent
        if cache_key not in app.state.agent_cache:
            print(f"Creating new fitness trainer for model: {model_name} with temperature: {temperature}")
            try:
                app.state.agent_cache[cache_key] = await create_fitness_chain(
                    model_name=model_name,
                    temperature=temperature
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to create fitness trainer with model '{model_name}': {str(e)}"
                )

        agent = app.state.agent_cache[cache_key]

        # Prepare input for the agent
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the agent
        result = await agent.ainvoke(inputs, config=config)

        # Extract the last message
        last_message = result["messages"][-1]
        response_text = extract_content(last_message)

        return ChatResponse(
            response=response_text,
            thread_id=thread_id,
            model_used=f"{model_name} (Fitness Trainer)"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/finance", response_model=ChatResponse)
async def chat_with_finance_advisor(request: ChatRequest):
    """Chat endpoint for the Finance Advisor."""
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        model_name = request.model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        temperature = request.temperature if request.temperature is not None else 0.7

        cache_key = f"finance:{model_name}:{temperature}"

        if cache_key not in app.state.agent_cache:
            print(f"Creating new finance advisor for model: {model_name}")
            app.state.agent_cache[cache_key] = await create_finance_chain(model_name, temperature)

        agent = app.state.agent_cache[cache_key]
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": thread_id, "user_id": request.user_id}}

        result = await agent.ainvoke(inputs, config=config)
        last_message = result["messages"][-1]
        response_text = extract_content(last_message)

        return ChatResponse(response=response_text, thread_id=thread_id, model_used=f"{model_name} (Finance)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/education", response_model=ChatResponse)
async def chat_with_study_assistant(request: ChatRequest):
    """Chat endpoint for the Study Assistant."""
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        model_name = request.model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        temperature = request.temperature if request.temperature is not None else 0.7

        cache_key = f"education:{model_name}:{temperature}"

        if cache_key not in app.state.agent_cache:
            print(f"Creating new study assistant for model: {model_name}")
            app.state.agent_cache[cache_key] = await create_education_chain(model_name, temperature)

        agent = app.state.agent_cache[cache_key]
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": thread_id, "user_id": request.user_id}}

        result = await agent.ainvoke(inputs, config=config)
        last_message = result["messages"][-1]
        response_text = extract_content(last_message)

        return ChatResponse(response=response_text, thread_id=thread_id, model_used=f"{model_name} (Education)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/career", response_model=ChatResponse)
async def chat_with_career_coach(request: ChatRequest):
    """Chat endpoint for the Career Coach."""
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        model_name = request.model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        temperature = request.temperature if request.temperature is not None else 0.7

        cache_key = f"career:{model_name}:{temperature}"

        if cache_key not in app.state.agent_cache:
            print(f"Creating new career coach for model: {model_name}")
            app.state.agent_cache[cache_key] = await create_career_chain(model_name, temperature)

        agent = app.state.agent_cache[cache_key]
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": thread_id, "user_id": request.user_id}}

        result = await agent.ainvoke(inputs, config=config)
        last_message = result["messages"][-1]
        response_text = extract_content(last_message)

        return ChatResponse(response=response_text, thread_id=thread_id, model_used=f"{model_name} (Career)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/health", response_model=ChatResponse)
async def chat_with_health_monitor(request: ChatRequest):
    """Chat endpoint for the Health Monitor."""
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        model_name = request.model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        temperature = request.temperature if request.temperature is not None else 0.7

        cache_key = f"health:{model_name}:{temperature}"

        if cache_key not in app.state.agent_cache:
            print(f"Creating new health monitor for model: {model_name}")
            app.state.agent_cache[cache_key] = await create_health_chain(model_name, temperature)

        agent = app.state.agent_cache[cache_key]
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": thread_id, "user_id": request.user_id}}

        result = await agent.ainvoke(inputs, config=config)
        last_message = result["messages"][-1]
        response_text = extract_content(last_message)

        return ChatResponse(response=response_text, thread_id=thread_id, model_used=f"{model_name} (Health)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/creative", response_model=ChatResponse)
async def chat_with_creative_assistant(request: ChatRequest):
    """Chat endpoint for the Creative Assistant."""
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        model_name = request.model or os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        temperature = request.temperature if request.temperature is not None else 0.9

        cache_key = f"creative:{model_name}:{temperature}"

        if cache_key not in app.state.agent_cache:
            print(f"Creating new creative assistant for model: {model_name}")
            app.state.agent_cache[cache_key] = await create_creative_chain(model_name, temperature)

        agent = app.state.agent_cache[cache_key]
        inputs = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": thread_id, "user_id": request.user_id}}

        result = await agent.ainvoke(inputs, config=config)
        last_message = result["messages"][-1]
        response_text = extract_content(last_message)

        return ChatResponse(response=response_text, thread_id=thread_id, model_used=f"{model_name} (Creative)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history", response_model=List[dict])
async def get_chat_history(thread_id: str, agent_type: str = "fitness"):
    """
    Retrieve chat history for a given thread.
    """
    try:
        # Determine model and temperature based on agent type (using defaults for now)
        model_name = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash")
        temperature = 0.7
        
        if agent_type == "fitness":
            cache_key = f"fitness:{model_name}:{temperature}"
            if cache_key not in app.state.agent_cache:
                app.state.agent_cache[cache_key] = await create_fitness_chain(model_name, temperature)
            agent = app.state.agent_cache[cache_key]
        elif agent_type == "finance":
            cache_key = f"finance:{model_name}:{temperature}"
            if cache_key not in app.state.agent_cache:
                app.state.agent_cache[cache_key] = await create_finance_chain(model_name, temperature)
            agent = app.state.agent_cache[cache_key]
        elif agent_type == "education":
            cache_key = f"education:{model_name}:{temperature}"
            if cache_key not in app.state.agent_cache:
                app.state.agent_cache[cache_key] = await create_education_chain(model_name, temperature)
            agent = app.state.agent_cache[cache_key]
        elif agent_type == "career":
            cache_key = f"career:{model_name}:{temperature}"
            if cache_key not in app.state.agent_cache:
                app.state.agent_cache[cache_key] = await create_career_chain(model_name, temperature)
            agent = app.state.agent_cache[cache_key]
        elif agent_type == "health":
            cache_key = f"health:{model_name}:{temperature}"
            if cache_key not in app.state.agent_cache:
                app.state.agent_cache[cache_key] = await create_health_chain(model_name, temperature)
            agent = app.state.agent_cache[cache_key]
        elif agent_type == "creative":
            cache_key = f"creative:{model_name}:0.9"  # Creative uses higher temperature
            if cache_key not in app.state.agent_cache:
                app.state.agent_cache[cache_key] = await create_creative_chain(model_name, 0.9)
            agent = app.state.agent_cache[cache_key]
        elif agent_type == "analyst":
             cache_key = f"analyst:{model_name}:{temperature}"
             if cache_key not in app.state.agent_cache:
                app.state.agent_cache[cache_key] = await create_data_analyst_chain(model_name, temperature)
             agent = app.state.agent_cache[cache_key]
        elif agent_type == "assistant":
             cache_key = f"assistant:{model_name}:{temperature}"
             if cache_key not in app.state.agent_cache:
                app.state.agent_cache[cache_key] = await create_general_assistant_chain(model_name, temperature)
             agent = app.state.agent_cache[cache_key]
        else:
             # Default/General agent
             cache_key = f"{model_name}:{temperature}"
             if cache_key not in app.state.agent_cache:
                app.state.agent_cache[cache_key] = await create_agent(model_name, temperature)
             agent = app.state.agent_cache[cache_key]

        config = {"configurable": {"thread_id": thread_id}}
        state = await agent.aget_state(config)
        
        if not state.values:
            return []
            
        messages = state.values.get("messages", [])
        history = []
        
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "model"
            content = extract_content(msg)
            # Skip system messages
            if not isinstance(msg, (HumanMessage, AIMessage)):
                continue
                
            history.append({
                "role": role,
                "content": content,
            })
            
        return history

    except Exception as e:
        print(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents", response_model=AgentListResponse)
async def get_agents():
    """
    Get a list of available agents.
    """
    agents = [
        AgentData(
            userAgentId="ua-general",
            agent=Agent(
                id="agent-general",
                name="General Assistant",
                description="Your helpful everyday AI assistant with web search and tools",
                agent_type="assistant",
            ),
            summary={
                "Status": "Active",
                "Capabilities": "Web Search, Calculator, Weather",
                "Model": "Gemini 2.5 Flash",
            },
        ),
        AgentData(
            userAgentId="ua-1",
            agent=Agent(
                id="agent-1",
                name="Personal Trainer",
                description="Your AI fitness coach for personalized workout plans and nutrition guidance",
                agent_type="fitness",
            ),
            summary={
                "Total Workouts": 24,
                "Current Streak": "7 days",
                "Calories Burned": "12,450 kcal",
                "Next Session": "Tomorrow, 8:00 AM",
            },
        ),
        AgentData(
            userAgentId="ua-2",
            agent=Agent(
                id="agent-2",
                name="Finance Agent",
                description="Smart financial advisor for budgeting, investments, and savings strategies",
                agent_type="finance",
            ),
            summary={
                "Monthly Budget": "$4,250",
                "Savings Rate": "28%",
                "Investment Return": "+12.4%",
                "Next Goal": "Emergency Fund",
            },
        ),
        AgentData(
            userAgentId="ua-3",
            agent=Agent(
                id="agent-3",
                name="Study Assistant",
                description="Academic support for learning, research, and exam preparation",
                agent_type="education",
            ),
            summary={
                "Study Hours": "156 hrs",
                "Topics Mastered": 18,
                "Upcoming Exam": "Math - Dec 15",
                "Progress": "82%",
            },
        ),
        AgentData(
            userAgentId="ua-4",
            agent=Agent(
                id="agent-4",
                name="Career Coach",
                description="Professional development guide for career planning and skill building",
                agent_type="career",
            ),
            summary={
                "Skills Learning": 3,
                "Applications Sent": 12,
                "Interviews": "2 scheduled",
                "Network Size": "145 contacts",
            },
        ),
        AgentData(
            userAgentId="ua-5",
            agent=Agent(
                id="agent-5",
                name="Health Monitor",
                description="Track and optimize your overall health and wellness journey",
                agent_type="health",
            ),
            summary={
                "Sleep Average": "7.2 hrs",
                "Water Intake": "2.1 L/day",
                "Steps Today": "8,456",
                "Health Score": "87/100",
            },
        ),
        AgentData(
            userAgentId="ua-6",
            agent=Agent(
                id="agent-6",
                name="Creative Assistant",
                description="Unleash creativity with brainstorming, content ideas, and artistic inspiration",
                agent_type="creative",
            ),
            summary={
                "Projects Active": 4,
                "Ideas Generated": 67,
                "Drafts Completed": 9,
                "Last Session": "2 hours ago",
            },
        ),
    ]
    return AgentListResponse(agents=agents)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
