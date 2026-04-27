"""Routing decision history persistence tests.

Tests that verify routing decision history (recent_selections, _model_frequency)
is properly persisted to the RoutingDecision table and can be recovered on restart.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from router.backends.base import ModelInfo
from router.database import get_session
from router.models import Base, RoutingDecision
from router.router import RouterEngine, RoutingResult


class _DummyBackend:
    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(name="llama3:8b"),
            ModelInfo(name="mistral:7b"),
            ModelInfo(name="codellama:7b"),
        ]

    async def chat(self, *args, **kwargs) -> dict:
        return {"message": {"content": "ok"}}

    async def chat_streaming(self, *args, **kwargs):
        raise NotImplementedError

    async def unload_model(self, model_name: str) -> bool:
        return False

    async def load_model(
        self, model_name: str, keep_alive: float = -1, timeout: float | None = None
    ) -> bool:
        return False

    async def embed(self, model: str, input_text, **kwargs) -> dict:
        return {"embedding": [0.1, 0.2, 0.3]}

    async def get_model_vram_usage(self, model_name: str) -> float | None:
        return None

    async def close(self) -> None:
        return None

    def is_external_model(self, model_name: str) -> bool:
        return False


@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine("sqlite:///:memory:")
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    with patch("router.database.engine", engine):
        with patch("router.database.SessionLocal", testing_session_local):
            yield engine


@pytest.fixture
def engine() -> RouterEngine:
    return RouterEngine(client=_DummyBackend(), cache_enabled=True)


@pytest.mark.asyncio
async def test_log_decision_persists_to_db(test_db) -> None:
    """Test that log_decision() persists routing decisions to RoutingDecision table."""
    router = RouterEngine(client=_DummyBackend(), cache_enabled=True)

    prompt = "Write a Python function"
    selected_model = "llama3:8b"
    confidence = 0.95
    reasoning = "Selected for general programming task"

    await router.log_decision(
        prompt=prompt,
        selected=selected_model,
        confidence=confidence,
        reasoning=reasoning,
    )

    with get_session() as session:
        decisions = session.query(RoutingDecision).all()
        assert len(decisions) == 1
        assert decisions[0].selected_model == selected_model
        assert decisions[0].confidence == confidence
        assert decisions[0].reasoning == reasoning
        assert decisions[0].prompt_hash is not None
        assert decisions[0].timestamp is not None


@pytest.mark.asyncio
async def test_recent_selections_recovered_from_db(test_db) -> None:
    """Test that recent_selections is properly recovered from RoutingDecision table."""
    with get_session() as session:
        for i, model in enumerate(["llama3:8b", "mistral:7b", "codellama:7b"]):
            decision = RoutingDecision(
                prompt_hash=f"hash_{i}",
                selected_model=model,
                confidence=0.8 + i * 0.05,
                reasoning=f"Test decision {i}",
            )
            session.add(decision)
        session.commit()

    router = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    assert router.semantic_cache is not None
    assert len(router.semantic_cache.recent_selections) == 0

    await router.load_persistent_cache()

    assert len(router.semantic_cache.recent_selections) == 3
    models_in_order = [m for m, _ in router.semantic_cache.recent_selections]
    assert "llama3:8b" in models_in_order
    assert "mistral:7b" in models_in_order
    assert "codellama:7b" in models_in_order


@pytest.mark.asyncio
async def test_model_frequency_recovered_from_db(test_db) -> None:
    """Test that _model_frequency is properly calculated from RoutingDecision table."""
    with get_session() as session:
        for _ in range(5):
            session.add(
                RoutingDecision(
                    prompt_hash="hash",
                    selected_model="llama3:8b",
                    confidence=0.9,
                    reasoning="test",
                )
            )
        for _ in range(3):
            session.add(
                RoutingDecision(
                    prompt_hash="hash",
                    selected_model="mistral:7b",
                    confidence=0.8,
                    reasoning="test",
                )
            )
        session.commit()

    router = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    assert router.semantic_cache is not None

    await router.load_persistent_cache()

    assert router.semantic_cache._model_frequency["llama3:8b"] == 5
    assert router.semantic_cache._model_frequency["mistral:7b"] == 3


@pytest.mark.asyncio
async def test_max_recent_limit_enforced(test_db) -> None:
    """Test that max_recent limit is enforced when loading history."""
    with get_session() as session:
        for i in range(30):
            session.add(
                RoutingDecision(
                    prompt_hash=f"hash_{i}",
                    selected_model=f"model_{i % 3}",
                    confidence=0.9,
                    reasoning="test",
                )
            )
        session.commit()

    router = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    assert router.semantic_cache is not None
    max_recent = router.semantic_cache.max_recent
    assert max_recent == 20

    await router.load_persistent_cache()

    assert len(router.semantic_cache.recent_selections) == max_recent


@pytest.mark.asyncio
async def test_get_model_frequency_uses_recovered_data(test_db) -> None:
    """Test that get_model_frequency() uses data recovered from database."""
    with get_session() as session:
        for _ in range(4):
            session.add(
                RoutingDecision(
                    prompt_hash="hash",
                    selected_model="llama3:8b",
                    confidence=0.9,
                    reasoning="test",
                )
            )
        for _ in range(1):
            session.add(
                RoutingDecision(
                    prompt_hash="hash",
                    selected_model="mistral:7b",
                    confidence=0.8,
                    reasoning="test",
                )
            )
        session.commit()

    router = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    assert router.semantic_cache is not None

    await router.load_persistent_cache()

    llama_freq = await router.semantic_cache.get_model_frequency("llama3:8b")
    mistral_freq = await router.semantic_cache.get_model_frequency("mistral:7b")

    assert llama_freq == 0.8
    assert mistral_freq == 0.2


@pytest.mark.asyncio
async def test_empty_database_returns_empty_history(test_db) -> None:
    """Test that loading from empty database doesn't cause errors."""
    router = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    assert router.semantic_cache is not None

    await router.load_persistent_cache()

    assert len(router.semantic_cache.recent_selections) == 0
    assert len(router.semantic_cache._model_frequency) == 0


@pytest.mark.asyncio
async def test_simulation_of_restart_scenario(test_db) -> None:
    """Full simulation: make decisions, "restart", verify history is recovered."""
    router1 = RouterEngine(client=_DummyBackend(), cache_enabled=True)

    decisions = [
        ("Write Python code", "codellama:7b", 0.9, "Coding task"),
        ("Explain quantum physics", "llama3:8b", 0.85, "General reasoning"),
        ("Write Python code", "codellama:7b", 0.92, "Coding task"),
    ]

    for prompt, model, conf, reason in decisions:
        await router1.log_decision(
            prompt=prompt,
            selected=model,
            confidence=conf,
            reasoning=reason,
        )

    assert router1.semantic_cache is not None
    assert len(router1.semantic_cache.recent_selections) == 3
    assert router1.semantic_cache._model_frequency["codellama:7b"] == 2
    assert router1.semantic_cache._model_frequency["llama3:8b"] == 1

    router2 = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    assert router2.semantic_cache is not None
    assert len(router2.semantic_cache.recent_selections) == 0

    await router2.load_persistent_cache()

    assert len(router2.semantic_cache.recent_selections) == 3
    assert router2.semantic_cache._model_frequency["codellama:7b"] == 2
    assert router2.semantic_cache._model_frequency["llama3:8b"] == 1

    freq_codellama = await router2.semantic_cache.get_model_frequency("codellama:7b")
    freq_llama = await router2.semantic_cache.get_model_frequency("llama3:8b")

    assert freq_codellama == 2 / 3
    assert freq_llama == 1 / 3


@pytest.mark.asyncio
async def test_new_decisions_add_to_recovered_history(test_db) -> None:
    """Test that new decisions are properly added to recovered history."""
    with get_session() as session:
        session.add(
            RoutingDecision(
                prompt_hash="old_hash",
                selected_model="llama3:8b",
                confidence=0.9,
                reasoning="old decision",
            )
        )
        session.commit()

    router = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    assert router.semantic_cache is not None

    await router.load_persistent_cache()

    assert len(router.semantic_cache.recent_selections) == 1
    assert router.semantic_cache._model_frequency["llama3:8b"] == 1

    await router.log_decision(
        prompt="new prompt",
        selected="mistral:7b",
        confidence=0.85,
        reasoning="new decision",
    )

    assert len(router.semantic_cache.recent_selections) == 2
    assert router.semantic_cache._model_frequency["llama3:8b"] == 1
    assert router.semantic_cache._model_frequency["mistral:7b"] == 1
