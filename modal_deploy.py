import modal
import os

# Define the image with necessary system and Python dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "basic-pitch==0.4.0",
        "fastapi>=0.104.1",
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-openai",
        "langchain-anthropic",
        "langchain-google-genai",
        "langchain-mcp-adapters",
        "langgraph",
        "librosa>=0.10.0",
        "numpy>=1.20.0,<2.0",
        "onnxruntime>=1.23.2",
        "pretty-midi>=0.2.5",
        "python-dotenv>=0.21.0",
        "python-multipart>=0.0.20",
        "scipy>=1.4.0",
        "setuptools>=80.9.0",
        "soundfile>=0.12.0",
        "sse-starlette>=2.2.1",
        "uvicorn>=0.24.0",
        "httpx",
        "aiosqlite" # Added for DB persistence
    )
)

app = modal.App("serverless-agent-host")

# Create a volume for persistent storage (database)
volume = modal.Volume.from_name("fitness-chat-volume", create_if_missing=True)

@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()], # Load secrets from local .env file
    volumes={"/data": volume}, # Mount volume to /data
    timeout=600, # 10 minute timeout for long-running agent tasks
    mounts=[modal.Mount.from_local_dir("app", remote_path="/root/app")]
)
@modal.asgi_app()
def fastapi_app():
    import os
    
    # Configure the application to use the persistent volume for the database
    os.environ["DB_PATH"] = "/data/fitness_chat.db"
    
    from app.server import app as server_app
    return server_app
