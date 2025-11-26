# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a serverless FastAPI application that hosts a LangGraph-based AI agent with integrated MCP (Model Context Protocol) tool support. The agent supports multiple LLM providers (OpenAI, Anthropic, Google) with Google's Gemini 2.5 Flash as the default model. It can dynamically load tools from external MCP servers, enabling extensible capabilities beyond local functions.

**Tech Stack:**
- Backend: FastAPI, Uvicorn
- AI Framework: LangGraph, LangChain
- LLM: Google Gemini 2.5 Flash (default), with support for OpenAI GPT-4o and Anthropic Claude
- MCP Integration: langchain-mcp-adapters
- Audio Processing: Spotify Basic Pitch, ONNX Runtime
- Development: Python 3.10-3.12

## Features

- **Multi-Agent Architecture**: Host and serve multiple, distinct AI agents and chains.
- **FastAPI Backend**: A robust and easy-to-use API server.
- **Serverless Deployment**: Pre-configured for easy deployment on Modal.
- **Extensible**: Designed to be easily extended with new agents and capabilities.

## Development Commands

### Setup

**Install dependencies:**
```bash
uv sync
```
Or alternatively with pip:
```bash
pip install -e .
```
Note: This project uses `pyproject.toml` for dependency management with hatchling as the build backend.

**Configure environment:**
```bash
cp .env.example .env
# Edit .env and add:
# - OPENAI_API_KEY (required)
# - GOOGLE_API_KEY (optional, for future integrations)
# - MCP_SSE_URL (optional, for SSE MCP servers)
# - MCP_STDIO_COMMAND (optional, for stdio MCP servers)
```

**Run development server:**
```bash
uvicorn app.server:app --reload --port 8000
```

**Access the API:**
- API: http://localhost:8000
- Health check: http://localhost:8000/health
- Interactive docs: http://localhost:8000/docs
- Fourier audio health: http://localhost:8000/fourier/health

### Testing

**Run tests:**
```bash
pytest
```

**Run specific test:**
```bash
pytest tests/test_app.py -v
```

Note: The test suite mocks the agent creation to avoid requiring API keys during testing.

### Docker

**Build and run:**
```bash
docker build -t serverless-agent-host .
docker run -p 8000:8000 --env-file .env serverless-agent-host
```

The Dockerfile uses a multi-stage build optimized for minimal image size with system dependencies for audio processing (`libgomp1`, `libsndfile1`).

## Architecture

The project is centered around a FastAPI server (`app/server.py`) that exposes several API endpoints. The core agent logic is housed in the `app/agents/` directory, with specialized chains in `app/chains/` and audio processing in `app/fourier/`. The application is designed to be deployed as a serverless application using `modal_deploy.py`.

### Core Components

**`app/server.py`** - Main FastAPI application with lifespan management:
- Initializes the LangGraph agent on startup via `lifespan` context manager
- Stores agents in `app.state.agent_cache` for request handling
- Exposes endpoints:
  - `GET /` - Service status
  - `GET /health` - Health check
  - `POST /chat` - Basic chat endpoint with minimal agent
  - `POST /chat/assistant` - General-purpose helpful assistant with utility tools
  - `POST /chat/analyst` - Data analyst specialized agent with data analysis tools
  - `POST /chat/fitness` - Personal trainer agent with fitness and nutrition tools
  - `POST /chat/finance` - Finance advisor agent with budgeting and transaction tools
  - `POST /chat/education` - Study assistant agent with learning and session tracking tools
  - `POST /chat/career` - Career coach agent with job search and skill development tools
  - `POST /chat/health` - Health monitor agent with wellness and metrics tracking tools
  - `POST /chat/creative` - Creative assistant agent with brainstorming and project tools
  - `GET /chat/history` - Retrieve chat history for a given thread
  - `GET /agents` - Get list of available agents
  - `POST /fourier/audio-to-midi` - Convert WAV audio to MIDI (via Fourier router)
  - `GET /fourier/download/{filename}` - Download generated MIDI files (via Fourier router)
  - `GET /fourier/health` - Audio module health check (via Fourier router)
- Includes CORS middleware (all origins allowed)
- Integrates Fourier audio processing router

**`app/agents/langgraph_agent.py`** - LangGraph agent implementation:
- `create_agent()` - Async factory that builds and compiles the agent graph
- Uses `StateGraph` with `AgentState` TypedDict for message state management
- Agent flow: chatbot node → conditional tool execution → back to chatbot
- Local tools: `get_weather` (example tool for testing)
- MCP tools: Loaded dynamically from configured MCP servers
- All tools bound to GPT-4o model with temperature=0

The agent uses LangGraph's prebuilt `ToolNode` and `tools_condition` for automatic tool routing based on the model's tool calls.

**`app/mcp_client.py`** - MCP server connection logic:
- `get_mcp_tools()` - Async function that connects to MCP servers and returns tools as LangChain tools
- Supports two connection types:
  - **SSE (Server-Sent Events)**: Via `MCP_SSE_URL` environment variable
  - **Stdio**: Via `MCP_STDIO_COMMAND` environment variable (less common for serverless)
- Uses `langchain_mcp_adapters.tools.load_mcp_tools` to convert MCP tools to LangChain format
- Gracefully handles connection failures and missing packages
- Returns empty list if no servers configured or if connection fails

**`app/chains/general_assistant_chain.py`** - General-purpose helpful assistant:
- `create_general_assistant_chain()` - Async factory for versatile assistant with utility tools
- Uses multi-model support via `get_llm()` factory
- Utility tools:
  - `calculate` - Mathematical expression evaluation
  - `get_current_time` - Current date and time (UTC)
  - `search_web` - Web search (placeholder for integration with Google/Bing/Tavily APIs)
  - `get_weather` - Weather information (placeholder for OpenWeatherMap/WeatherAPI)
  - `convert_units` - Unit conversions (temperature, distance, weight)
  - `get_fun_fact` - Interesting facts by topic
- Default temperature (0.7) for balanced, helpful responses
- System prompt emphasizes helpfulness and accuracy
- Perfect for general-purpose conversations and everyday tasks

**`app/chains/chat_chain.py`** - Data Analyst specialized chain:
- `create_data_analyst_chain()` - Async factory for data analyst agent with specialized tools
- Uses same multi-model support via `get_llm()` factory
- Specialized tools:
  - `calculate_statistics` - Basic statistical analysis
  - `search_census_data` - U.S. Census Bureau data queries
  - `search_health_data` - WHO and FDA database queries
- Default temperature (0.7) for conversational responses
- System prompt focused on data analysis expertise
- Uses LangGraph's `MemorySaver` for conversation persistence

**`app/chains/fitness_chain.py`** - Personal Trainer specialized chain:
- `create_fitness_chain()` - Async factory for fitness trainer agent with health and workout tools
- Uses same multi-model support via `get_llm()` factory
- Specialized tools:
  - `calculate_bmi` - Body Mass Index calculation
  - `calculate_macros` - Macro nutrient planning (protein, carbs, fats)
  - `get_exercise_info` - Exercise descriptions and benefits
  - `log_workout` - Log completed workout sessions to user profile
  - `get_workout_history` - Retrieve past workout history
- Default temperature (0.7) for conversational, motivational responses
- System prompt focused on fitness coaching and motivation
- Uses AsyncSqliteSaver for conversation persistence (fitness_chat.db)

**`app/chains/finance_chain.py`** - Finance Advisor specialized chain:
- `create_finance_chain()` - Async factory for finance advisor agent with budgeting tools
- Specialized tools:
  - `calculate_savings_rate` - Calculate savings as percentage of income
  - `budget_recommendation` - Provide 50/30/20 budget breakdown
  - `log_transaction` - Log income/expense transactions to user profile
  - `get_transaction_history` - Retrieve past financial transactions
- Default temperature (0.7) for professional financial advice
- System prompt focused on financial planning and budgeting
- Uses AsyncSqliteSaver for conversation persistence (finance_chat.db)

**`app/chains/education_chain.py`** - Study Assistant specialized chain:
- `create_education_chain()` - Async factory for study assistant agent with learning tools
- Specialized tools:
  - `log_study_session` - Log study sessions with subject, topic, and duration
  - `get_study_history` - Retrieve past study sessions
- Default temperature (0.7) for educational, supportive responses
- System prompt focused on learning strategies and exam preparation
- Uses AsyncSqliteSaver for conversation persistence (education_chat.db)

**`app/chains/career_chain.py`** - Career Coach specialized chain:
- `create_career_chain()` - Async factory for career coach agent with job search tools
- Specialized tools:
  - `log_career_activity` - Log job applications, interviews, networking events
  - `get_career_history` - Retrieve past career activities
- Default temperature (0.7) for professional career guidance
- System prompt focused on career development and job search
- Uses AsyncSqliteSaver for conversation persistence (career_chat.db)

**`app/chains/health_chain.py`** - Health Monitor specialized chain:
- `create_health_chain()` - Async factory for health monitor agent with wellness tracking tools
- Specialized tools:
  - `log_health_entry` - Log daily health metrics (sleep, water, steps, mood, energy)
  - `get_health_history` - Retrieve past health logs
- Default temperature (0.7) for supportive wellness guidance
- System prompt focused on healthy habits and wellness tracking
- Uses AsyncSqliteSaver for conversation persistence (health_chat.db)

**`app/chains/creative_chain.py`** - Creative Assistant specialized chain:
- `create_creative_chain()` - Async factory for creative assistant agent with brainstorming tools
- Specialized tools:
  - `log_creative_project` - Log creative projects with ideas and status
  - `get_creative_history` - Retrieve past creative projects
- Default temperature (0.9) for enhanced creativity and ideation
- System prompt focused on brainstorming and creative inspiration
- Uses AsyncSqliteSaver for conversation persistence (creative_chat.db)

**`app/schema.py`** - Pydantic models:
- `ChatRequest`: `message` (str), `thread_id` (optional str), `model` (optional str), `temperature` (optional float)
- `ChatResponse`: `response` (str), `thread_id` (str), `model_used` (optional str)

The `thread_id` enables conversation continuity across requests using LangGraph's checkpointing system.

### Audio Processing Module (Fourier)

**`app/fourier/`** - Audio-to-MIDI conversion module using Spotify's Basic Pitch algorithm:

**Components:**
- `basic_pitch_processor.py` - Core processing logic with polyphonic audio transcription
- `routes.py` - FastAPI router with audio processing endpoints

**Endpoints:**

1. **`POST /fourier/audio-to-midi`** - Convert WAV audio to MIDI
   - Accepts multipart/form-data with:
     - `audio_file` (required): WAV file to convert
     - `min_note_duration` (optional, default 0.05): Minimum note duration in seconds
     - `min_frequency` (optional, default 27.5): Minimum frequency in Hz
     - `max_frequency` (optional, default 4186.0): Maximum frequency in Hz
   - Returns JSON with:
     - `filename`: Name of generated MIDI file
     - `midi_url`: Download URL path
     - `note_count`: Number of notes detected
     - `duration_seconds`: Duration of the audio/MIDI
   - Example:
     ```bash
     curl -X POST http://localhost:8000/fourier/audio-to-midi \
       -F "audio_file=@path/to/audio.wav" \
       -F "min_note_duration=0.05" \
       -F "min_frequency=27.5" \
       -F "max_frequency=4186.0"
     ```
   - Example response:
     ```json
     {
       "filename": "upload_audio.mid",
       "midi_url": "/fourier/download/upload_audio.mid",
       "note_count": 8,
       "duration_seconds": 9.41
     }
     ```

2. **`GET /fourier/download/{filename}`** - Download generated MIDI files
   - Returns MIDI file as `audio/midi` attachment
   - Files are automatically cleaned up after 5 minutes
   - Example:
     ```bash
     curl http://localhost:8000/fourier/download/upload_audio.mid \
       --output output.mid
     ```

3. **`GET /fourier/health`** - Module health check
   - Returns status and basic information about the audio processing module

**Technical Details:**
- Uses ONNX Runtime as ML backend (lighter alternative to TensorFlow)
- Temporary file storage in `/tmp/langhashi_fourier`
- Background tasks for automatic file cleanup (5-minute retention)
- Supports polyphonic audio (multiple simultaneous notes)
- Compatible with most WAV formats

### Request Flow

**Basic Chat Request:**
```
POST /chat → server.py chat() → Basic LangGraph agent → [Minimal tool execution] → Response
```

**General Assistant Chat Request:**
```
POST /chat/assistant → server.py chat_with_general_assistant() → General Assistant Chain → [Utility tools] → Response
```

**Data Analyst Chat Request:**
```
POST /chat/analyst → server.py chat_with_data_analyst() → Data Analyst Chain → [Data analysis tools] → Response
```

**Fitness Trainer Chat Request:**
```
POST /chat/fitness → server.py chat_with_fitness_trainer() → Fitness Chain → [BMI, macros, workout tools] → Response
```

All agents maintain thread state via `thread_id` parameter in config, enabling multi-turn conversations within the same thread. Agents are cached by `{type}:{model}:{temperature}` combination for optimal performance.

**Audio Processing Request:**
```
POST /fourier/audio-to-midi → routes.py → BasicPitchProcessor → MIDI generation → Response with download URL
```

## Environment Variables

**Required:**
- `GOOGLE_API_KEY` - For Gemini models (default model: gemini-2.5-flash)

**Optional:**
- `DEFAULT_MODEL` - Default model to use (default: "gemini-2.5-flash")
- `OPENAI_API_KEY` - For OpenAI models (gpt-4o, gpt-4-turbo, etc.)
- `ANTHROPIC_API_KEY` - For Anthropic models (claude-3-5-sonnet, etc.)
- `MCP_SSE_URL` - URL of an SSE-based MCP server
- `MCP_STDIO_COMMAND` - Command to launch a stdio-based MCP server (e.g., "node mcp-server.js")

## Adding Features

### Adding New Local Tools

1. Define tool in `app/agents/langgraph_agent.py`:
```python
@tool
def my_tool(param: str):
    """Tool description for the LLM."""
    return "result"
```

2. Add to `local_tools` list in `create_agent()`:
```python
local_tools = [get_weather, my_tool]
```

The tool will automatically be available to the agent via LangGraph's tool routing.

### Adding MCP Servers

MCP servers provide external tools without modifying code:

1. **SSE Server**: Set environment variable:
```bash
MCP_SSE_URL=http://your-mcp-server.com/sse
```

2. **Stdio Server**: Set environment variable:
```bash
MCP_STDIO_COMMAND="node /path/to/mcp-server.js"
```

Tools from MCP servers are loaded automatically on startup via `get_mcp_tools()`.

### Adding New API Endpoints

1. Create a new router file (e.g., `app/mymodule/routes.py`)
2. Define endpoints using FastAPI router
3. Import and include router in `app/server.py`:
```python
from app.mymodule.routes import router as mymodule_router
app.include_router(mymodule_router)
```

## Key Implementation Notes

1. **Stateful Conversations**: Agent supports thread-based conversations via `thread_id`, but threads are not persisted across server restarts
2. **Async Agent Creation**: The agent is created asynchronously during application startup using lifespan context manager
3. **MCP Fallback**: If `langchain-mcp-adapters` is not available, `get_mcp_tools()` returns an empty list with a warning
4. **Tool Execution**: LangGraph handles tool execution automatically using `tools_condition` and `ToolNode`
5. **CORS**: All origins are allowed for cross-origin requests
6. **Error Handling**: FastAPI automatically converts exceptions to HTTP 500 responses with error details
7. **Test Mocking**: Tests mock `create_agent` to avoid requiring API keys and external dependencies
8. **Audio Backend**: Uses ONNX Runtime for Basic Pitch inference (lighter than TensorFlow)

## Dependency Management

This project uses `pyproject.toml` with hatchling as the build backend. Key dependencies:

**Core:**
- `fastapi>=0.104.1`
- `uvicorn>=0.24.0`
- `langchain>=0.3.0`, `langchain-core>=0.3.0`
- `langchain-openai` - OpenAI integration
- `langgraph` - Agent graph framework
- `langchain-mcp-adapters` - MCP tool integration

**Audio Processing:**
- `basic-pitch==0.4.0` - Audio-to-MIDI conversion
- `onnxruntime>=1.23.2` - ML inference backend
- `pretty-midi>=0.2.5` - MIDI file manipulation
- `librosa>=0.10.0`, `soundfile>=0.12.0` - Audio I/O

**Utilities:**
- `python-dotenv>=0.21.0` - Environment variable management
- `python-multipart>=0.0.20` - File upload support
- `sse-starlette>=2.2.1` - SSE support
- `httpx` - HTTP client

**Adding dependencies:**
```bash
# Edit pyproject.toml dependencies array
# Then reinstall:
pip install -e .
```

## Important Notes

- **Multi-Model Support**: The application now supports OpenAI, Anthropic, and Google models with intelligent routing based on model name
- **Default Model**: Gemini 2.5 Flash is the default model for all requests unless specified otherwise
- **Agent Caching**: Agents are cached by model+temperature combination for performance
- **Per-Request Model Selection**: Clients can specify different models per request via the `model` field
- **NumPy Compatibility**: The application requires NumPy < 2.0 due to tflite-runtime compatibility. If you encounter `AttributeError: _ARRAY_API not found`, downgrade NumPy: `pip install 'numpy<2.0'`
- **Gemini Response Format**: Gemini returns content as a list of content blocks rather than plain strings. The `extract_content()` helper function in `app/server.py` handles this cross-model compatibility
- Thread IDs enable conversation continuity but are not persisted to a database
- MCP integration is designed for extensibility but requires external MCP servers to be configured
- The `tests/test_app.py` file imports from `app.main` which should be `app.server` (this is a bug in the test file)
