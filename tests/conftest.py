"""Pytest configuration and shared fixtures."""

import pytest

from router.backends import resilience
from router.backends.health import reset_health_manager

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(autouse=True)
def reset_health_state_before_tests():
    """Reset global health manager state before each test.

    This ensures that health tracking state from one test doesn't
    affect other tests, preventing issues where backends marked as
    unhealthy in one test cause failures in subsequent tests.
    """
    reset_health_manager()
    resilience._health_manager = None
    yield


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


@pytest.fixture
def mock_ollama_response():
    """Fixture for mock Ollama API response."""
    return {
        "model": "llama3",
        "created_at": "2024-01-01T00:00:00Z",
        "response": "Test response",
        "done": True,
        "context": [1, 2, 3],
        "total_duration": 1000000,
        "load_duration": 100000,
        "prompt_eval_count": 10,
        "eval_count": 20,
    }


@pytest.fixture
def mock_model_info():
    """Fixture for mock model info."""
    return {
        "name": "llama3",
        "size": 1000000000,
        "modified_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_benchmark_data():
    """Fixture for sample benchmark data."""
    return {
        "ollama_name": "llama3",
        "full_name": "Meta-Llama-3-8B",
        "mmlu": 70.0,
        "humaneval": 50.0,
        "math": 40.0,
        "gpqa": 45.0,
        "hellaswag": 80.0,
        "winogrande": 75.0,
        "truthfulqa": 60.0,
        "mmlu_pro": 65.0,
        "reasoning_score": 0.75,
        "coding_score": 0.60,
        "general_score": 0.70,
        "elo_rating": 1200.0,
        "throughput": 50.0,
        "context_window": 8192,
        "parameters": "8B",
    }
