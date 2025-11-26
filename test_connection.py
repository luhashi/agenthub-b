import aiohttp
import asyncio

async def test_connection():
    url = "http://localhost:3000/api/user/fitness/workout?userId=user_test_123"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                print(f"Status: {response.status}")
                text = await response.text()
                print(f"Response: {text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
