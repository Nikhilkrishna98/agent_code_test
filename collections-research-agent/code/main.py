import logging
import os

import click

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent import CollectionsResearchAgent
from agent_executor import CollectionsResearchAgentExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    '''Exception for missing API key.'''
    pass

@click.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=8008)
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=True)
        # No tools listed, but describe agent's main expertise
        skill = AgentSkill(
            id='collections_data_analysis',
            name='Collections Data Analysis',
            description='Analyzes customer collections data by reading emails and receipts from AWS S3, identifies reasons for pending receipts/payments by correlating with the customer payments table, updates findings, and uploads the updated table to S3.',
            tags=['collections', 's3', 'email', 'receipts', 'csv', 'data-analysis'],
            examples=[
                "Identify why payments for Acme Corp are still pending based on the latest email and receipt files in our S3 bucket.",
                "Update the customer payments table with findings from the receipts and emails associated with XYZ Ltd."
            ],
        )
        agent_card = AgentCard(
            name='Collections Research Agent',
            description='Analyzes customer collections data from AWS S3, correlates with payment tables, identifies reasons for pending payments/receipts, and uploads findings.',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=CollectionsResearchAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=CollectionsResearchAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        request_handler = DefaultRequestHandler(
            agent_executor=CollectionsResearchAgentExecutor(),
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
