"""Cache persistence recovery tests (Item #58).

Uses the real RouterEngine/SemanticCache APIs and verifies persistent cache
behavior through reload scenarios.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from router.backends.base import ModelInfo
from router.database import get_session
from router.models import Base, RoutingCache, ResponseCache
from router.router import RouterEngine, RoutingResult


class _DummyBackend:
    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="model-a")]

    async def chat(self, *args, **kwargs) -> dict:
        return {"message": {"content": "ok"}}

    async def chat_streaming(self, *args, **kwargs):
        raise NotImplementedError

    async def unload_model(self, model_name: str) -> bool:
        return False

    async def load_model(self, model_name: str, keep_alive: float = -1, timeout: float | None = None) -> bool:
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
    """Create test database using StaticPool to ensure single connection for in-memory SQLite.

    SQLite in-memory databases are isolated per connection by default. Using StaticPool
    ensures all sessions share the same connection, allowing data to be visible across
    different session instances.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    with patch("router.database.engine", engine):
        with patch("router.database.SessionLocal", testing_session_local):
            yield engine


@pytest.mark.asyncio
async def test_routing_entry_round_trip_via_persistence(test_db) -> None:
    """Test that routing cache entries persist and can be loaded across RouterEngine instances."""
    router1 = RouterEngine(client=_DummyBackend(), cache_enabled=True)

    if not router1.semantic_cache or not router1.semantic_cache.persistent_cache:
        pytest.skip("Persistent cache is not enabled")

    prompt = "cache persistence prompt"
    result = RoutingResult(
        selected_model="model-a",
        confidence=0.9,
        reasoning="test",
    )

    await router1.semantic_cache.set(prompt, result, embedding=[0.1, 0.2, 0.3])

    with get_session() as session:
        entries = session.execute(select(RoutingCache)).scalars().all()
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
        assert entries[0].cache_key is not None
        print(f"\n=== DEBUG: Stored entry cache_key = {entries[0].cache_key} ===")
        print(f"=== DEBUG: Stored entry created_at = {entries[0].created_at} ===")
        print(f"=== DEBUG: Stored entry expires_at = {entries[0].expires_at} ===")

    router2 = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    if not router2.semantic_cache:
        pytest.fail("Semantic cache is unexpectedly disabled")

    with get_session() as session:
        entries = session.execute(select(RoutingCache)).scalars().all()
        assert len(entries) == 1, f"Expected 1 entry before load, got {len(entries)}"

    print(f"\n=== DEBUG: router2.semantic_cache.persistent_cache.enabled = {router2.semantic_cache.persistent_cache.enabled} ===")
    print(f"=== DEBUG: router2.semantic_cache.persistent_cache.max_age_days = {router2.semantic_cache.persistent_cache.max_age_days} ===")
    
    now = datetime.utcnow()
    cutoff_time = now - timedelta(days=router2.semantic_cache.persistent_cache.max_age_days)
    print(f"=== DEBUG: Current time (utcnow) = {now} ===")
    print(f"=== DEBUG: Cutoff time = {cutoff_time} ===")
    
    with get_session() as session:
        entries = session.execute(select(RoutingCache)).scalars().all()
        for entry in entries:
            print(f"=== DEBUG: Entry created_at = {entry.created_at}, type = {type(entry.created_at)} ===")
            print(f"=== DEBUG: created_at > cutoff_time = {entry.created_at > cutoff_time} ===")
            if entry.expires_at:
                print(f"=== DEBUG: Entry expires_at = {entry.expires_at}, type = {type(entry.expires_at)} ===")
                print(f"=== DEBUG: expires_at > now = {entry.expires_at > now} ===")
    
    routing_data = await router2.semantic_cache.persistent_cache.load_routing_cache()
    print(f"=== DEBUG: load_routing_cache returned {len(routing_data)} entries ===")
    print(f"=== DEBUG: routing_data keys = {list(routing_data.keys())} ===")

    await router2.semantic_cache.load_from_persistence()
    print(f"=== DEBUG: After load_from_persistence, cache has {len(router2.semantic_cache.cache)} entries ===")

    with get_session() as session:
        entries = session.execute(select(RoutingCache)).scalars().all()
        assert len(entries) == 1, f"Expected 1 entry after load, got {len(entries)}"

    loaded = await router2.semantic_cache.get(prompt, embedding=[0.1, 0.2, 0.3])

    assert loaded is not None, f"Expected cached result, got None. Cache has {len(router2.semantic_cache.cache)} entries"
    assert loaded.selected_model == "model-a"


@pytest.mark.asyncio
async def test_response_entry_round_trip_via_persistence(test_db) -> None:
    """Test that response cache entries persist and can be loaded across RouterEngine instances."""
    router1 = RouterEngine(client=_DummyBackend(), cache_enabled=True)

    if not router1.semantic_cache or not router1.semantic_cache.persistent_cache:
        pytest.skip("Persistent cache is not enabled")

    model = "model-a"
    prompt = "response cache prompt"
    text = "cached response"

    await router1.semantic_cache.set_response(model=model, prompt=prompt, response=text)

    with get_session() as session:
        entries = session.execute(select(ResponseCache)).scalars().all()
        assert len(entries) == 1, f"Expected 1 response entry, got {len(entries)}"

    router2 = RouterEngine(client=_DummyBackend(), cache_enabled=True)
    if not router2.semantic_cache:
        pytest.fail("Semantic cache is unexpectedly disabled")

    await router2.semantic_cache.load_from_persistence()
    loaded = await router2.semantic_cache.get_response(model=model, prompt=prompt)

    assert loaded == text, f"Expected '{text}', got {loaded}"


@pytest.mark.asyncio
async def test_clear_removes_entries_from_memory(test_db) -> None:
    """Test that clear() removes entries from in-memory cache."""
    router = RouterEngine(client=_DummyBackend(), cache_enabled=True)

    if not router.semantic_cache:
        pytest.fail("Semantic cache is unexpectedly disabled")

    prompt = "memory clear prompt"
    result = RoutingResult(selected_model="model-a", confidence=0.5, reasoning="test")
    await router.semantic_cache.set(prompt, result)

    assert await router.semantic_cache.get(prompt) is not None
    await router.semantic_cache.clear()
    assert await router.semantic_cache.get(prompt) is None
