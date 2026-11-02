import pytest


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_OFFLINE", "1")
    monkeypatch.setenv("AGENT_VENDOR_LATENCY_MS", "0")
    yield


@pytest.fixture
def settings():
    from onboarding_agent.config import get_settings

    return get_settings()


@pytest.fixture
def agent(settings):
    from onboarding_agent.graph import Agent

    return Agent(settings)
