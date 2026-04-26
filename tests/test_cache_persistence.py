"""Cache persistence recovery tests (Item #58).

Uses the real RouterEngine/SemanticCache APIs and verifies persistent cache
behavior through reload scenarios and TTL-based expiration.
"""

import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from router.backends.base import ModelInfo
from router.database import get_session
from router.models import Base, EmbeddingCache, ResponseCache, RoutingCache
from router.persistent_cache import PersistentCacheManager
from router.router import RouterEngine, RoutingResult, SemanticCache


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


@pytest.fixture(scope="function")
def temp_db_path():
    """Create a temporary SQLite database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)

    engine = create_engine(f"sqlite:///{temp_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    engine.dispose()

    yield temp_path

    try:
        temp_path.unlink()
    except Exception:
        pass


@pytest.fixture(scope="function")
def db_engine(temp_db_path):
    """Create a SQLAlchemy engine for the temporary database."""
    engine = create_engine(f"sqlite:///{temp_db_path.as_posix()}")
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def persistent_cache_manager(temp_db_path, db_engine):
    """Create a PersistentCacheManager that uses the test database.

    Patches router.database to use the temporary database.
    """
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    with patch("router.database.engine", db_engine):
        with patch("router.database.SessionLocal", TestingSessionLocal):
            pc = PersistentCacheManager(enabled=True, max_age_days=7)
            yield pc


@pytest.fixture
def semantic_cache(persistent_cache_manager):
    """Create a SemanticCache with persistent cache enabled."""
    cache = SemanticCache(
        max_size=100,
        ttl_seconds=3600,
        similarity_threshold=0.85,
        persistent_cache_manager=persistent_cache_manager,
        cache_stats_enabled=True,
    )
    return cache


def set_expired_in_db(model_cls, entry_id):
    """Set an entry's expires_at to a time in the past to simulate expiration.

    Uses get_session() to ensure the same connection context.
    """
    with get_session() as session:
        entry = session.query(model_cls).filter_by(id=entry_id).first()
        if entry:
            entry.expires_at = datetime.utcnow() - timedelta(hours=1)
            session.commit()


class TestCachePersistenceRoundTrip:
    """Test that cache entries can be saved and reloaded from persistent storage."""

    @pytest.mark.asyncio
    async def test_save_and_load_routing_entry(self, semantic_cache):
        """Test that routing cache entries can be saved and loaded."""
        prompt = "test routing prompt"
        result = RoutingResult(
            selected_model="model-a",
            confidence=0.9,
            reasoning="test reasoning",
        )

        await semantic_cache.persistent_cache.save_routing_entry(
            cache_key=semantic_cache._hash_prompt(prompt),
            result=result,
            embedding=[0.1, 0.2, 0.3],
            embedding_magnitude=1.0,
            ttl_seconds=3600,
        )

        with get_session() as session:
            entry = session.query(RoutingCache).first()
            assert entry is not None
            assert entry.selected_model == "model-a"
            assert entry.expires_at is not None
            assert entry.expires_at > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_save_and_load_response_entry(self, semantic_cache):
        """Test that response cache entries can be saved and loaded."""
        model = "model-a"
        prompt = "test response prompt"
        key = (model, semantic_cache._hash_prompt(prompt))
        response_text = "this is a cached response"

        await semantic_cache.persistent_cache.save_response_entry(
            cache_key=key,
            response_text=response_text,
            ttl_seconds=3600,
        )

        with get_session() as session:
            entry = session.query(ResponseCache).first()
            assert entry is not None
            assert entry.response_text == response_text
            assert entry.expires_at is not None

    @pytest.mark.asyncio
    async def test_save_and_load_embedding_entry(self, semantic_cache):
        """Test that embedding cache entries can be saved and loaded."""
        prompt = "test embedding prompt"
        key = semantic_cache._hash_prompt(prompt)
        embedding = [0.1, 0.2, 0.3, 0.4]
        magnitude = 1.0

        await semantic_cache.persistent_cache.save_embedding_entry(
            prompt_hash=key,
            embedding=embedding,
            magnitude=magnitude,
            ttl_seconds=86400,
        )

        with get_session() as session:
            entry = session.query(EmbeddingCache).first()
            assert entry is not None
            assert entry.embedding == embedding
            assert entry.magnitude == magnitude


class TestPersistentCacheTTLExpiration:
    """Test TTL-based expiration in persistent cache."""

    @pytest.mark.asyncio
    async def test_routing_entry_expires_in_future(self, semantic_cache):
        """Test that routing cache entries have expiration time in the future."""
        prompt = "ttl test prompt"
        result = RoutingResult(selected_model="model-a", confidence=0.9, reasoning="test")

        await semantic_cache.persistent_cache.save_routing_entry(
            cache_key=semantic_cache._hash_prompt(prompt),
            result=result,
            embedding=[0.1, 0.2, 0.3],
            embedding_magnitude=1.0,
            ttl_seconds=3600,
        )

        with get_session() as session:
            entry = session.query(RoutingCache).first()
            assert entry is not None
            assert entry.expires_at is not None
            assert entry.expires_at > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_response_entry_expires_in_future(self, semantic_cache):
        """Test that response cache entries have expiration time in the future."""
        key = ("model-a", semantic_cache._hash_prompt("test"))

        await semantic_cache.persistent_cache.save_response_entry(
            cache_key=key,
            response_text="test response",
            ttl_seconds=3600,
        )

        with get_session() as session:
            entry = session.query(ResponseCache).first()
            assert entry is not None
            assert entry.expires_at is not None
            assert entry.expires_at > datetime.utcnow()

    @pytest.mark.asyncio
    async def test_embedding_entry_expires_in_future(self, semantic_cache):
        """Test that embedding cache entries have expiration time in the future."""
        key = semantic_cache._hash_prompt("test")

        await semantic_cache.persistent_cache.save_embedding_entry(
            prompt_hash=key,
            embedding=[0.1, 0.2, 0.3],
            magnitude=1.0,
            ttl_seconds=86400,
        )

        with get_session() as session:
            entry = session.query(EmbeddingCache).first()
            assert entry is not None
            assert entry.expires_at is not None
            assert entry.expires_at > datetime.utcnow()


class TestPersistentCacheCleanup:
    """Test that expired entries are properly cleaned up from persistent cache."""

    @pytest.mark.asyncio
    async def test_delete_expired_routing_entries(self, semantic_cache):
        """Test that expired routing entries are deleted."""
        prompt = "expired routing prompt"
        result = RoutingResult(selected_model="model-a", confidence=0.9, reasoning="test")

        await semantic_cache.persistent_cache.save_routing_entry(
            cache_key=semantic_cache._hash_prompt(prompt),
            result=result,
            embedding=[0.1, 0.2, 0.3],
            embedding_magnitude=1.0,
            ttl_seconds=3600,
        )

        with get_session() as session:
            entry = session.query(RoutingCache).first()
            assert entry is not None
            entry_id = entry.id

        set_expired_in_db(RoutingCache, entry_id)

        with get_session() as session:
            entry = session.query(RoutingCache).first()
            assert entry.expires_at < datetime.utcnow()

        counts = await semantic_cache.persistent_cache.delete_expired_entries()

        assert counts["routing"] == 1
        assert counts["response"] == 0
        assert counts["embedding"] == 0

        with get_session() as session:
            entry = session.query(RoutingCache).first()
            assert entry is None

    @pytest.mark.asyncio
    async def test_delete_expired_response_entries(self, semantic_cache):
        """Test that expired response entries are deleted."""
        key = ("model-a", semantic_cache._hash_prompt("expired response"))

        await semantic_cache.persistent_cache.save_response_entry(
            cache_key=key,
            response_text="expired response",
            ttl_seconds=3600,
        )

        with get_session() as session:
            entry = session.query(ResponseCache).first()
            assert entry is not None
            entry_id = entry.id

        set_expired_in_db(ResponseCache, entry_id)

        with get_session() as session:
            entry = session.query(ResponseCache).first()
            assert entry.expires_at < datetime.utcnow()

        counts = await semantic_cache.persistent_cache.delete_expired_entries()

        assert counts["response"] == 1

        with get_session() as session:
            entry = session.query(ResponseCache).first()
            assert entry is None

    @pytest.mark.asyncio
    async def test_delete_expired_embedding_entries(self, semantic_cache):
        """Test that expired embedding entries are deleted."""
        key = semantic_cache._hash_prompt("expired embedding")

        await semantic_cache.persistent_cache.save_embedding_entry(
            prompt_hash=key,
            embedding=[0.1, 0.2, 0.3],
            magnitude=1.0,
            ttl_seconds=86400,
        )

        with get_session() as session:
            entry = session.query(EmbeddingCache).first()
            assert entry is not None
            entry_id = entry.id

        set_expired_in_db(EmbeddingCache, entry_id)

        with get_session() as session:
            entry = session.query(EmbeddingCache).first()
            assert entry.expires_at < datetime.utcnow()

        counts = await semantic_cache.persistent_cache.delete_expired_entries()

        assert counts["embedding"] == 1

        with get_session() as session:
            entry = session.query(EmbeddingCache).first()
            assert entry is None

    @pytest.mark.asyncio
    async def test_delete_only_expired_entries(self, semantic_cache):
        """Test that only expired entries are deleted, fresh ones remain."""
        prompt_expired = "expired prompt"
        result_expired = RoutingResult(selected_model="expired-model", confidence=0.9, reasoning="expired")
        await semantic_cache.persistent_cache.save_routing_entry(
            cache_key=semantic_cache._hash_prompt(prompt_expired),
            result=result_expired,
            embedding=[0.1, 0.2, 0.3],
            embedding_magnitude=1.0,
            ttl_seconds=3600,
        )

        prompt_fresh = "fresh prompt"
        result_fresh = RoutingResult(selected_model="fresh-model", confidence=0.9, reasoning="fresh")
        await semantic_cache.persistent_cache.save_routing_entry(
            cache_key=semantic_cache._hash_prompt(prompt_fresh),
            result=result_fresh,
            embedding=[0.4, 0.5, 0.6],
            embedding_magnitude=1.0,
            ttl_seconds=3600,
        )

        with get_session() as session:
            assert session.query(RoutingCache).count() == 2
            expired_entry = session.query(RoutingCache).filter_by(selected_model="expired-model").first()
            expired_id = expired_entry.id

        set_expired_in_db(RoutingCache, expired_id)

        counts = await semantic_cache.persistent_cache.delete_expired_entries()

        assert counts["routing"] == 1

        with get_session() as session:
            entries = session.query(RoutingCache).all()
            assert len(entries) == 1
            assert entries[0].selected_model == "fresh-model"

    @pytest.mark.asyncio
    async def test_get_stats_counts_active_entries(self, semantic_cache):
        """Test that get_stats() correctly counts only active (non-expired) entries."""
        stats = await semantic_cache.persistent_cache.get_stats()
        assert stats["routing"] == 0
        assert stats["response"] == 0
        assert stats["embedding"] == 0

        prompt_expired = "expired"
        result_expired = RoutingResult(selected_model="model-a", confidence=0.9, reasoning="test")
        await semantic_cache.persistent_cache.save_routing_entry(
            cache_key=semantic_cache._hash_prompt(prompt_expired),
            result=result_expired,
            ttl_seconds=3600,
        )

        prompt_fresh = "fresh"
        result_fresh = RoutingResult(selected_model="model-b", confidence=0.9, reasoning="test")
        await semantic_cache.persistent_cache.save_routing_entry(
            cache_key=semantic_cache._hash_prompt(prompt_fresh),
            result=result_fresh,
            ttl_seconds=3600,
        )

        with get_session() as session:
            expired_entry = session.query(RoutingCache).filter_by(selected_model="model-a").first()
            expired_id = expired_entry.id

        set_expired_in_db(RoutingCache, expired_id)

        stats = await semantic_cache.persistent_cache.get_stats()
        assert stats["routing"] == 1


class TestSemanticCacheWithPersistence:
    """Test SemanticCache integration with PersistentCacheManager."""

    @pytest.mark.asyncio
    async def test_set_saves_to_persistence(self, semantic_cache):
        """Test that SemanticCache.set() saves to persistent cache."""
        prompt = "test set persistence"
        result = RoutingResult(selected_model="model-a", confidence=0.9, reasoning="test")

        await semantic_cache.set(prompt, result, embedding=[0.1, 0.2, 0.3])

        with get_session() as session:
            entry = session.query(RoutingCache).first()
            assert entry is not None
            assert entry.selected_model == "model-a"

    @pytest.mark.asyncio
    async def test_set_response_saves_to_persistence(self, semantic_cache):
        """Test that SemanticCache.set_response() saves to persistent cache."""
        model = "model-a"
        prompt = "test set response"
        response = "cached response text"

        await semantic_cache.set_response(model=model, prompt=prompt, response=response)

        with get_session() as session:
            entry = session.query(ResponseCache).first()
            assert entry is not None
            assert entry.response_text == response

    @pytest.mark.asyncio
    async def test_load_from_persistence_loads_active_entries(self, semantic_cache):
        """Test that load_from_persistence() loads only active (non-expired) entries."""
        prompt_fresh = "fresh load test"
        result_fresh = RoutingResult(selected_model="fresh-model", confidence=0.9, reasoning="fresh")
        await semantic_cache.persistent_cache.save_routing_entry(
            cache_key=semantic_cache._hash_prompt(prompt_fresh),
            result=result_fresh,
            embedding=[0.1, 0.2, 0.3],
            embedding_magnitude=1.0,
            ttl_seconds=3600,
        )

        prompt_expired = "expired load test"
        result_expired = RoutingResult(selected_model="expired-model", confidence=0.9, reasoning="expired")
        await semantic_cache.persistent_cache.save_routing_entry(
            cache_key=semantic_cache._hash_prompt(prompt_expired),
            result=result_expired,
            embedding=[0.4, 0.5, 0.6],
            embedding_magnitude=1.0,
            ttl_seconds=3600,
        )

        with get_session() as session:
            expired_entry = session.query(RoutingCache).filter_by(selected_model="expired-model").first()
            expired_id = expired_entry.id

        set_expired_in_db(RoutingCache, expired_id)

        fresh_cache = SemanticCache(
            max_size=100,
            ttl_seconds=3600,
            similarity_threshold=0.85,
            persistent_cache_manager=semantic_cache.persistent_cache,
        )

        await fresh_cache.load_from_persistence()

        stats = await fresh_cache.get_stats()
        assert stats["routing"]["size"] == 1

        loaded_fresh = await fresh_cache.get(prompt_fresh, embedding=[0.1, 0.2, 0.3])
        assert loaded_fresh is not None
        assert loaded_fresh.selected_model == "fresh-model"

        loaded_expired = await fresh_cache.get(prompt_expired, embedding=[0.4, 0.5, 0.6])
        assert loaded_expired is None


class TestMemoryCacheTTLInvalidation:
    """Test TTL-based invalidation in memory cache (already tested in test_cache_invalidation.py)."""

    @pytest.mark.asyncio
    async def test_memory_cache_ttl_check(self):
        """Basic test to verify TTL checking works in memory cache."""
        cache = SemanticCache(
            max_size=10,
            ttl_seconds=1,
            similarity_threshold=0.85,
        )

        prompt = "test ttl"
        result = RoutingResult(selected_model="model-a", confidence=0.9, reasoning="test")

        await cache.set(prompt, result)

        cached = await cache.get(prompt)
        assert cached is not None

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_memory_entries(self):
        """Test that cleanup_expired() removes expired memory entries."""
        cache = SemanticCache(
            max_size=10,
            ttl_seconds=3600,
            similarity_threshold=0.85,
        )

        prompt = "test cleanup"
        result = RoutingResult(selected_model="model-a", confidence=0.9, reasoning="test")

        await cache.set(prompt, result)

        stats = await cache.get_stats()
        assert stats["routing"]["size"] == 1

        expired = await cache.cleanup_expired()
        assert expired["routing"] == 0

        async with cache._routing_lock:
            for key, value in cache.cache.items():
                cache.cache[key] = (value[0], 0, value[2], value[3], value[4])

        expired = await cache.cleanup_expired()
        assert expired["routing"] == 1

        stats = await cache.get_stats()
        assert stats["routing"]["size"] == 0
