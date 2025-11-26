# GEMINI.md: Project Overview & Context

This document provides a comprehensive overview of the Serverless Agent Host project for Gemini. It details the project's purpose, architecture, and instructions for building, running, and development.

## 1. Project Overview

This is a Python-based, flexible backend service designed to host and serve a variety of AI agents and models. It is built with **FastAPI** for the web server, and is configured for serverless deployment using **Modal**.

## Features

*   **Multi-Agent Architecture**: Host and serve multiple, distinct AI agents and chains.
*   **FastAPI Backend**: A robust and easy-to-use API server.
*   **Serverless Deployment**: Pre-configured for easy deployment on Modal.
*   **Extensible**: Designed to be easily extended with new agents and capabilities.
*   **Frontend Integration**: Fully integrated with the **MicroSaaSFast** frontend via REST API.

## Frontend Integration

This backend serves as the AI engine for the **GitFit/MicroSaaSFast** frontend.
- **Chat**: The frontend communicates with `/chat`, `/chat/assistant`, `/chat/analyst`, `/chat/fitness`, `/chat/finance`, `/chat/education`, `/chat/career`, `/chat/health`, and `/chat/creative` endpoints.
- **Persistence**: Conversation threads are managed by the frontend (storing `thread_id` in PostgreSQL) and passed to the backend for context.
- **CORS**: Configured to allow requests from the frontend domain.

## Architecture

The project is centered around a FastAPI server (`app/server.py`) that exposes several API endpoints. The core agent logic is housed in the `app/agents/` directory, with specialized chains in `app/chains/` and audio processing in `app/fourier/`. The application is designed to be deployed as a serverless application using `modal_deploy.py`.

The repository currently contains multiple distinct agent and chain implementations:

1.  **LangGraph Agent (`agents/langgraph_agent.py`)**: A flexible agent powered by **LangGraph** with support for tool calling and MCP (Model Context Protocol) integration. Uses **Google's Gemini 2.5 Flash** by default but can be configured for other LLMs (OpenAI GPT-4o, Anthropic Claude).

2.  **MCP Tools Integration (`mcp_client.py`)**: Dynamic tool loading from MCP servers (SSE or Stdio), allowing the agent to connect to external systems and data sources in a standardized way.

3. **Audio Processing (`fourier/`)**: A module for audio processing, specifically converting WAV audio files to MIDI using Spotify's Basic Pitch algorithm. This feature is exposed via its own set of API endpoints.

The project is structured to be extensible, allowing new agents to be added to the `app/agents` directory and integrated into the `app/server.py` FastAPI router.

### Key Technologies
- **Backend Framework**: FastAPI
- **AI Frameworks**: LangChain, LangGraph
- **LLMs**: Google Gemini 2.5 Flash (default), OpenAI GPT-4o, Anthropic Claude (configurable)
- **Audio Processing**: Basic Pitch, Librosa, SciPy
- **Deployment**: Modal, Docker
- **Package Manager**: uv
- **Dependencies**: Managed via `pyproject.toml`

## 2. Building and Running

### Local Development

1.  **Install Dependencies** (using uv):
    ```bash
    uv sync
    ```

2.  **Configure Environment**:
    Create a `.env` file from the example and add your API keys and other configuration.
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file with your details. You'll need `GOOGLE_API_KEY` for the default Gemini model. Optionally add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for alternative models. Set `MCP_SSE_URL` or `MCP_STDIO_COMMAND` for MCP tools.

3.  **Run the Server**:
    ```bash
    uvicorn app.server:app --reload --port 8000
    ```
    The API will be available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`.

### Docker Deployment

1.  **Build**:
    ```bash
    docker build -t serverless-agent-host .
    ```

2.  **Run**:
    ```bash
    docker run -p 8000:8000 --env-file .env serverless-agent-host
    ```

### Serverless Deployment (Modal)

The application is pre-configured for deployment on the Modal serverless platform.

1.  **Set Modal Secrets**:
    Provide your API keys and credentials to Modal's secret management.
    ```bash
    modal secret create openai-api-key OPENAI_API_KEY=<your-openai-key>
    ```

2.  **Deploy**:
    ```bash
    modal deploy modal_deploy.py
    ```
    Modal will deploy the app and provide a public URL.

## 3. Development Conventions

### Project Structure
- `app/`: Contains the main application source code.
  - `server.py`: The main FastAPI application, defining endpoints and routing.
  - `agents/`: Houses the different AI agent implementations.
    - `langgraph_agent.py`: The LangGraph agent with tool support.
  - `chains/`: Contains conversational chains (placeholder for future chains).
  - `fourier/`: Contains the audio processing module.
    - `routes.py`: FastAPI routes for audio endpoints.
    - `basic_pitch_processor.py`: Core audio-to-MIDI conversion logic.
  - `mcp_client.py`: MCP tool loader.
  - `schema.py`: Pydantic models for API requests/responses.
- `modal_deploy.py`: The script for deploying the application to Modal.
- `pyproject.toml`: Project configuration and dependencies.
- `Dockerfile`: Multi-stage Docker build configuration using uv.
- `.env.example`: An example file for environment variable configuration.

### API Endpoints
The `app/server.py` file defines the following key endpoints:
- `GET /`: Root status endpoint.
- `GET /health`: Health check endpoint (for Docker healthcheck).
- `POST /chat`: Chat with the basic LangGraph agent. Accepts a message and optional thread_id.
- `POST /chat/assistant`: Chat with the General Purpose Assistant (web search, calculator, weather, etc.).
- `POST /chat/analyst`: Chat with the Data Analyst agent (census data, health statistics).
- `POST /chat/fitness`: Chat with the Personal Trainer agent (BMI, macros, workout routines).
- `POST /chat/finance`: Chat with the Finance Advisor agent (budgeting, transactions, savings).
- `POST /chat/education`: Chat with the Study Assistant agent (learning, study sessions, exam prep).
- `POST /chat/career`: Chat with the Career Coach agent (job search, skill development, networking).
- `POST /chat/health`: Chat with the Health Monitor agent (wellness, sleep, mood, energy tracking).
- `POST /chat/creative`: Chat with the Creative Assistant agent (brainstorming, project planning, ideation).
- `GET /chat/history`: Retrieve chat history for a given thread.
- `GET /agents`: Get list of available agents.
- `POST /fourier/audio-to-midi`: Converts a WAV audio file to MIDI.
- `GET /fourier/download/{filename}`: Downloads a generated MIDI file.
- `GET /fourier/health`: Audio module health check.

### Adding New Agents
To extend the service with a new agent:
1.  Create your agent's logic in a new file within the `app/agents/` directory.
2.  Import your agent's functions into `app/server.py`.
3.  Create a new FastAPI endpoint to expose your agent's functionality.

### Adding New Chains
To add a conversational chain:
1.  Create your chain logic in a new file within the `app/chains/` directory.
2.  Import and integrate it into `app/server.py` or the relevant agent.
