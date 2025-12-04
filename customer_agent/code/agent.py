import os
from dotenv import load_dotenv
load_dotenv()

import json
from typing import Any, AsyncIterable

from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams

# LLM setup using environment variables
model = os.getenv('AZURE_MODEL_NAME_LITELLM')
api_base = os.getenv('AZURE_API_BASE')
api_version = os.getenv('API_VERSION')
api_key = os.getenv('AZURE_API_KEY')

llm = LiteLlm(
    model=model,
    api_base=api_base,
    api_version=api_version,
    api_key=api_key
)

class CustomerSupportAgent:
    '''An AI customer support agent specialized in replying to customer emails.'''

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = 'customer_support_user'
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def get_processing_message(self) -> str:
        return 'Composing customer support reply...'

    def _build_agent(self) -> LlmAgent:
        '''Builds the LLM agent for customer support replies.'''
        # MCP Tool Initialization (ALWAYS include, even if tool list is empty)
        mcp_toolset = MCPToolset(
            connection_params=SseConnectionParams(
                url=os.getenv("MCP_SERVER_URL"),
                headers={
                    # Populate headers if needed, or leave empty for defaults
                }
            )
        )
        return LlmAgent(
            model=llm.model,
            name='customer_support_reply_agent',
            description=(
                'An AI customer support agent specialized in replying to customer emails with accurate, clear, and helpful responses.'
            ),
            instruction=(
                "You are an AI Customer Support Reply Agent. Your role is to review customer email queries and compose appropriate replies. "
                "Create responses that are clear, polite, and directly address the customer's issues or questions. Use existing tools such as reading emails from S3 if provided, or data extraction tools as needed. "
                "Always attempt to utilize available tools to fetch necessary information for the reply or context before generating your response. "
                "If no tool is needed for the task, proceed with language-model-based reasoning and reply generation."
            ),
            tools=[
                mcp_toolset
            ],
        )

    async def stream(self, query: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        content = types.Content(
            role='user', parts=[types.Part.from_text(text=query)]
        )
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ''
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = '\n'.join(
                        [p.text for p in event.content.parts if p.text]
                    )
                elif (
                    event.content
                    and event.content.parts
                    and any(
                        [
                            True
                            for p in event.content.parts
                            if p.function_response
                        ]
                    )
                ):
                    response = next(
                        p.function_response.model_dump()
                        for p in event.content.parts
                    )
                yield {
                    'is_task_complete': True,
                    'content': response,
                }
            else:
                yield {
                    'is_task_complete': False,
                    'updates': self.get_processing_message(),
                }

# For direct script use - sample agent initialization
if __name__ == '__main__':
    import asyncio

    agent = CustomerSupportAgent()
    sample_query = "Hello, I have a question about my recent billing statement. Can you explain the additional charges?"
    sample_session_id = "sample-session-001"

    async def run_sample():
        async for msg in agent.stream(sample_query, sample_session_id):
            print(msg)

    asyncio.run(run_sample())
