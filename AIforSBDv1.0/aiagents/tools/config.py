
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configuration values
ADO_ORGANIZATION = os.getenv("ADO_ORGANIZATION")
ADO_PROJECT = os.getenv("ADO_PROJECT")
ADO_WIKI_NAME = os.getenv("ADO_WIKI_NAME")
ADO_PAT_TOKEN = os.getenv("ADO_PAT_TOKEN")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME")
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")

VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")
VISION_API_KEY = os.getenv("VISION_API_KEY")

FOUNDRY_PROJECT_ID = os.getenv("FOUNDRY_PROJECT_ID")
FOUNDRY_AGENT_NAME = os.getenv("FOUNDRY_AGENT_NAME", "architecture_agent")
FOUNDRY_AGENT_ID = os.getenv("FOUNDRY_AGENT_ID")
FOUNDRY_PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
