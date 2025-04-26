import asyncio
import os
import mcp_use

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def main():
    # Load environment variables
    load_dotenv()
    #mcp_use.set_debug(2)
    
    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):        
        raise ValueError("GROQ_API_KEY not set in environment variables")

    # Create MCPClient from configuration dictionary
    client = MCPClient.from_config_file(
        os.path.join(os.path.dirname(__file__), "server_mcp.json")
    )
    
    print(client.get_server_names())

    # Create LLM (using Groq)

    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",  # ou troca por outro modelo do Groq se quiser
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.getenv("GROQ_API_KEY")
    )
    #llm = ChatOpenAI(model="gpt-4o")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # Run the query
    try:
        # Run a query to find 3 restaurants in Moita using Mojeek
        result = await agent.run(
            "Use search_google to find 3 restaurants in Moita, Portugal. Please provide only 3 options."
        )
        print(result)
    finally:
        # Ensure we clean up resources properly
        if client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")
