import os
import sys
from pathlib import Path


def pytest_sessionstart(session):
    """Set import path and required environment defaults for tests."""

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    os.environ["TESTING"] = "true"
    os.environ["DEBUG"] = "false"
    os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
    os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
