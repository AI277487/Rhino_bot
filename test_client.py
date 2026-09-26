import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PY = "/home/botuser/rhino-bot/venv/bin/python"
SERVER = "/home/botuser/rhino-bot/mcp_server.py"

async def main():
    params = StdioServerParameters(command=PY, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("TOOLS FOUND:", [t.name for t in tools.tools])
            print("\n--- calling search_ent ---")
            res = await session.call_tool(
                "search_ent",
                {"question": "complications of FESS", "k": 3},
            )
            print(res.content[0].text[:1500])

asyncio.run(main())
