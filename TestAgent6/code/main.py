import logging
import os

import click

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
)
from agent import EmailAgent
from agent_executor import EmailAgentExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MissingAPIKeyError(Exception):
    '''Exception for missing API key.'''
    pass

@click.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8005)
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=True)
        agent_card = AgentCard(
            name='Gmail Reply Agent',
            description='An agent that replies to received emails via Gmail in a friendly manner. It interacts with users to request Gmail authentication credentials when required.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=EmailAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=EmailAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[],
        )
        request_handler = DefaultRequestHandler(
            agent_executor=EmailAgentExecutor(),
            task_store=InMemoryTaskStore(),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )
        import uvicorn

        uvicorn.run(server.build(), host=host, port=port)
    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)

if __name__ == '__main__':
    main()
