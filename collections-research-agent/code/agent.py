import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- LLM Imports and Initialization ---
from langchain_openai import AzureChatOpenAI

os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_API_BASE')
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_API_KEY')
azure_deployment = os.getenv('AZURE_DEPLOYMENT_NAME')
api_version = os.getenv('AZURE_API_VERSION')
temperature = os.getenv('TEMPERATURE')

llm = AzureChatOpenAI(
    azure_deployment=azure_deployment,
    api_version=api_version,
    temperature=temperature
)

# --- MCP Tool Fetching Logic ---
from langchain_mcp_adapters.client import MultiServerMCPClient

def _fetch_mcp_tools_sync() -> list:
    """
    Helper function: runs the async MultiServerMCPClient code in a synchronous manner.
    Fetches the remote tools from your MCP server(s).
    """
    servers_config = {
        "mcp_server": {
            "transport": "sse",
            "url": os.getenv("MCP_SERVER_URL"),
        }
    }

    async def _fetch_tools():
        client = MultiServerMCPClient(servers_config)
        return await client.get_tools()

    return asyncio.run(_fetch_tools())

# --- Agent Definition ---
class CollectionsResearchAgent:
    """
    Analyzes customer collections data by reading emails and receipts from AWS S3,
    identifies reasons for pending receipts or payments by correlating with the customer payments table,
    updates the research findings, and uploads the updated table to S3.
    """

    SYSTEM_INSTRUCTION = (
        "You are a collections research agent specializing in analyzing customer collections data. "
        "You will be provided AWS account details and S3 paths. Your tasks: "
        "1) List file names in folders containing emails (eml) and receipts (pdf); "
        "2) Select relevant files based on customer payment and pending receipt information; "
        "3) Match pending payments from CSV data with information in related emails and receipts; "
        "4) Identify reasons for pending receipts/payments (e.g., missing receipts, delayed confirmations, etc.); "
        "5) Update your findings in the data table; "
        "6) Upload the updated table back to S3. "
        "Always attempt to use existing tools for interacting with S3, reading/analyzing email and PDF files, "
        "and handling tables/documents. Use tool calls whenever file operations or data extraction/manipulation outside LLM reasoning are required."
    )

    # These supported content types are referenced by __main__.py
    SUPPORTED_CONTENT_TYPES = ["text", "table", "csv", "application/json"]

    def __init__(self):
        # Always fetch MCP tools, even if config has an empty tool list
        self.tools = _fetch_mcp_tools_sync()
        self.llm = llm

    def get_tools(self):
        """Return the initialized tool list."""
        return self.tools

    def get_system_instruction(self):
        """Return the agent's system prompt/instruction."""
        return self.SYSTEM_INSTRUCTION

    def get_llm(self):
        """Return the LLM client."""
        return self.llm

if __name__ == "__main__":
    agent = CollectionsResearchAgent()
    print("Loaded tools from MCP:", agent.get_tools())
    print("System prompt:", agent.get_system_instruction())
