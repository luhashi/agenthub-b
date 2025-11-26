from typing import List
from langchain_core.tools import BaseTool
# Note: This import assumes langchain-mcp-adapters is installed and follows this structure.
# If the package structure is different, this will need adjustment.
# We are using a defensive import or a placeholder if the package is not yet fully available in the environment.

try:
    from langchain_mcp_adapters.tools import load_mcp_tools
except ImportError:
    # Fallback or mock if the package isn't installed in this environment yet
    print("Warning: langchain-mcp-adapters not found. Using mock.")
    def load_mcp_tools(session) -> List[BaseTool]:
        return []

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import os

async def get_mcp_tools() -> List[BaseTool]:
    """
    Connects to configured MCP servers and returns their tools as LangChain tools.
    """
    tools = []
    
    # Example: Load from an SSE MCP server if configured
    mcp_sse_url = os.getenv("MCP_SSE_URL")
    if mcp_sse_url:
        try:
            async with sse_client(mcp_sse_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    mcp_tools = await load_mcp_tools(session)
                    tools.extend(mcp_tools)
        except Exception as e:
            print(f"Error connecting to MCP SSE server at {mcp_sse_url}: {e}")

    # Example: Load from a Stdio MCP server if configured
    # This is less common for serverless but useful for local dev
    mcp_stdio_command = os.getenv("MCP_STDIO_COMMAND")
    if mcp_stdio_command:
        try:
            command_parts = mcp_stdio_command.split(" ")
            server_params = StdioServerParameters(
                command=command_parts[0],
                args=command_parts[1:],
                env=None
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    mcp_tools = await load_mcp_tools(session)
                    tools.extend(mcp_tools)
        except Exception as e:
            print(f"Error connecting to MCP Stdio server: {e}")

    return tools
