"""
Shared fixtures for OWASP LLM Top 10 scanner.
Target: LangChain RAG pipeline + agent over Ollama (son-of-anton).
"""

import pytest
import httpx
import os

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def pytest_configure(config):
    for i, name in enumerate([
        "Prompt Injection", "Sensitive Info Disclosure", "Supply Chain",
        "Data & Model Poisoning", "Improper Output Handling", "Excessive Agency",
        "System Prompt Leakage", "Vector & Embedding Weaknesses",
        "Misinformation", "Unbounded Consumption"
    ], 1):
        config.addinivalue_line("markers", f"llm{i:02d}: {name}")


@pytest.fixture(scope="session")
def ollama_ok():
    """Verify Ollama is reachable before any test runs."""
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        pytest.skip(f"Ollama not reachable at {OLLAMA_BASE}: {e}")


@pytest.fixture(scope="session")
def rag_chain(ollama_ok):
    """Build the LangChain RAG chain once per session — expensive (embeds corpus)."""
    from target.rag_chain import build_rag_chain
    return build_rag_chain(include_poisoned=True)


@pytest.fixture(scope="session")
def rag_chain_clean(ollama_ok):
    """RAG chain with clean corpus only — baseline for comparison."""
    from target.rag_chain import build_rag_chain
    return build_rag_chain(include_poisoned=False)


@pytest.fixture(scope="session")
def agent(ollama_ok):
    """LangChain agent with tools — target for LLM06 excessive agency probes."""
    from target.agent import build_agent
    return build_agent()


@pytest.fixture(scope="session")
def raw_ollama(ollama_ok):
    """Direct Ollama HTTP client — for checks that bypass LangChain intentionally."""
    client = httpx.Client(base_url=OLLAMA_BASE, timeout=60)
    yield client
    client.close()
