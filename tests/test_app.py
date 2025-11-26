from fastapi.testclient import TestClient
from app.server import app
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from langchain_core.messages import AIMessage

# Mock the agent creation to avoid needing API keys or real MCP servers
async def mock_create_agent(*args, **kwargs):
    mock_agent = AsyncMock()
    # Mock the ainvoke method to return a valid state
    mock_agent.ainvoke.return_value = {
        "messages": [AIMessage(content="Hello! I am a mocked agent.")]
    }
    return mock_agent

@pytest.fixture
def client():
    # Patch the create_agent function in app.main
    with patch("app.server.create_agent", side_effect=mock_create_agent):
        with TestClient(app) as c:
            yield c

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "serverless-agent-host"}

def test_chat(client):
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Hello! I am a mocked agent."
    assert "thread_id" in data

def test_get_agents(client):
    response = client.get("/agents")
    assert response.status_code == 200
    data = response.json()
    assert "agents" in data
    assert len(data["agents"]) > 0
    # Check for General Assistant
    assert any(a["agent"]["name"] == "General Assistant" for a in data["agents"])
    assert any(a["agent"]["agent_type"] == "assistant" for a in data["agents"])
