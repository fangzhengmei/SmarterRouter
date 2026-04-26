"""
Tests for semantic cache invalidation strategies (TTL + similarity threshold).

Verifies that:
1. TTL-expired entries are no longer returned on exact match
2. TTL-expired entries are excluded from semantic similarity search
3. Proactive cleanup via cleanup_expired() works correctly
4. TTL evictions are properly recorded in statistics
"""
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from router.router import RoutingResult, SemanticCache


class _MockRoutingResult:
    """Mock RoutingResult for testing."""

    def __init__(self, model: str = "llama3"):
        self.selected_model = model
        self.confidence = 0.9
        self.reasoning = "test reasoning"


class TestCacheTTLInvalidation:
    """Tests for TTL-based cache invalidation."""

    @pytest.fixture
    def semantic_cache(self):
        """Fixture providing SemanticCache with short TTL for testing."""
        return SemanticCache(
            max_size=10,
            ttl_seconds=1,
            similarity_threshold=0.85,
            cache_stats_enabled=True,
            cache_stats_retention_hours=1,
        )

    @pytest.mark.asyncio
    async def test_exact_match_expires_after_ttl(self, semantic_cache):
        """Test that exact cache matches stop working after TTL expires."""
        prompt = "test prompt for ttl expiration"
        result = _MockRoutingResult(model="llama3")

        await semantic_cache.set(prompt, result)

        cached = await semantic_cache.get(prompt)
        assert cached is not None
        assert cached.selected_model == "llama3"
        assert semantic_cache.stats["routing_hits"] == 1

        await asyncio.sleep(1.1)

        cached = await semantic_cache.get(prompt)
        assert cached is None
        assert semantic_cache.stats["routing_misses"] == 1

    @pytest.mark.asyncio
    async def test_semantic_search_excludes_expired_entries(self, semantic_cache):
        """Test that semantic similarity search excludes TTL-expired entries."""
        prompt1 = "How do I write a Python function?"
        prompt2 = "What's the way to create Python functions?"

        result1 = _MockRoutingResult(model="codellama")
        embedding1 = [0.1, 0.2, 0.3, 0.4]

        await semantic_cache.set(prompt1, result1, embedding=embedding1)

        embedding2 = [0.11, 0.21, 0.31, 0.41]
        cached = await semantic_cache.get(prompt2, embedding=embedding2)
        assert cached is not None
        assert cached.selected_model == "codellama"
        assert semantic_cache.stats["routing_similarity_hits"] == 1

        await asyncio.sleep(1.1)

        cached = await semantic_cache.get(prompt2, embedding=embedding2)
        assert cached is None
        assert semantic_cache.stats["routing_misses"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_expired_entries(self, semantic_cache):
        """Test that cleanup_expired() proactively removes expired entries."""
        prompt1 = "expired prompt 1"
        prompt2 = "fresh prompt"
        result1 = _MockRoutingResult(model="llama3")
        result2 = _MockRoutingResult(model="codellama")

        await semantic_cache.set(prompt1, result1)

        await asyncio.sleep(0.6)

        await semantic_cache.set(prompt2, result2)

        stats = await semantic_cache.get_stats()
        assert stats["routing"]["size"] == 2

        await asyncio.sleep(0.6)

        expired = await semantic_cache.cleanup_expired()

        assert expired["routing"] == 1
        assert expired["response"] == 0
        assert expired["embedding"] == 0

        stats = await semantic_cache.get_stats()
        assert stats["routing"]["size"] == 1

        cached1 = await semantic_cache.get(prompt1)
        assert cached1 is None

        cached2 = await semantic_cache.get(prompt2)
        assert cached2 is not None
        assert cached2.selected_model == "codellama"

    @pytest.mark.asyncio
    async def test_ttl_eviction_recorded_in_stats(self, semantic_cache):
        """Test that TTL-based evictions are recorded with correct reason."""
        mock_record = AsyncMock()
        semantic_cache.time_series_stats.record_eviction = mock_record

        prompt = "prompt to expire"
        result = _MockRoutingResult(model="llama3")

        await semantic_cache.set(prompt, result)

        await asyncio.sleep(1.1)

        await semantic_cache.get(prompt)

        assert mock_record.called
        call_kwargs = mock_record.call_args[1]
        assert call_kwargs["reason"] == "ttl"
        assert call_kwargs["cache_type"] == "routing"

    @pytest.mark.asyncio
    async def test_cleanup_expired_records_ttl_evictions(self, semantic_cache):
        """Test that cleanup_expired() records TTL evictions."""
        mock_record = AsyncMock()
        semantic_cache.time_series_stats.record_eviction = mock_record

        for i in range(3):
            prompt = f"prompt {i}"
            result = _MockRoutingResult(model=f"model-{i}")
            await semantic_cache.set(prompt, result)

        stats = await semantic_cache.get_stats()
        assert stats["routing"]["size"] == 3

        await asyncio.sleep(1.1)

        expired = await semantic_cache.cleanup_expired()

        assert expired["routing"] == 3
        assert mock_record.call_count == 3
        for call in mock_record.call_args_list:
            assert call[1]["reason"] == "ttl"


class TestResponseCacheTTLInvalidation:
    """Tests for response cache TTL invalidation."""

    @pytest.fixture
    def semantic_cache(self):
        """Fixture providing SemanticCache with short response TTL."""
        cache = SemanticCache(
            max_size=10,
            ttl_seconds=1,
            similarity_threshold=0.85,
            cache_stats_enabled=True,
        )
        cache.response_ttl = 1
        return cache

    @pytest.mark.asyncio
    async def test_response_cache_expires_after_ttl(self, semantic_cache):
        """Test that response cache entries expire after TTL."""
        model = "llama3"
        prompt = "test response prompt"
        response = "This is a cached response"

        await semantic_cache.set_response(model=model, prompt=prompt, response=response)

        cached = await semantic_cache.get_response(model=model, prompt=prompt)
        assert cached == response
        assert semantic_cache.stats["response_hits"] == 1

        await asyncio.sleep(1.1)

        cached = await semantic_cache.get_response(model=model, prompt=prompt)
        assert cached is None
        assert semantic_cache.stats["response_misses"] == 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_expired_responses(self, semantic_cache):
        """Test that cleanup_expired() removes expired response cache entries."""
        semantic_cache.response_ttl = 1

        await semantic_cache.set_response(model="llama3", prompt="prompt1", response="resp1")
        await asyncio.sleep(0.6)
        await semantic_cache.set_response(model="codellama", prompt="prompt2", response="resp2")

        stats = await semantic_cache.get_stats()
        assert stats["response"]["size"] == 2

        await asyncio.sleep(0.6)

        expired = await semantic_cache.cleanup_expired()

        assert expired["response"] == 1

        stats = await semantic_cache.get_stats()
        assert stats["response"]["size"] == 1


class TestSimilarityThresholdIntegration:
    """Tests for similarity threshold working with TTL."""

    @pytest.fixture
    def semantic_cache(self):
        """Fixture providing SemanticCache with controlled similarity threshold."""
        return SemanticCache(
            max_size=10,
            ttl_seconds=3600,
            similarity_threshold=0.9,
            cache_stats_enabled=True,
        )

    @pytest.mark.asyncio
    async def test_similarity_below_threshold_not_hit(self, semantic_cache):
        """Test that entries with similarity below threshold are not returned."""
        prompt1 = "How to write Python code?"
        result1 = _MockRoutingResult(model="codellama")
        embedding1 = [1.0, 0.0, 0.0, 0.0]

        await semantic_cache.set(prompt1, result1, embedding=embedding1)

        prompt2 = "How to write Java code?"
        embedding2 = [0.0, 1.0, 0.0, 0.0]

        cached = await semantic_cache.get(prompt2, embedding=embedding2)
        assert cached is None
        assert semantic_cache.stats["routing_misses"] == 1

    @pytest.mark.asyncio
    async def test_similarity_above_threshold_is_hit(self, semantic_cache):
        """Test that entries with similarity above threshold are returned."""
        semantic_cache.similarity_threshold = 0.7

        prompt1 = "How to write Python functions?"
        result1 = _MockRoutingResult(model="codellama")
        embedding1 = [1.0, 0.8, 0.0, 0.0]

        await semantic_cache.set(prompt1, result1, embedding=embedding1)

        prompt2 = "How to create Python methods?"
        embedding2 = [0.9, 0.9, 0.0, 0.0]

        cached = await semantic_cache.get(prompt2, embedding=embedding2)
        assert cached is not None
        assert cached.selected_model == "codellama"
        assert semantic_cache.stats["routing_similarity_hits"] == 1

    @pytest.mark.asyncio
    async def test_adaptive_threshold_used_for_comparison(self, semantic_cache):
        """Test that adaptive threshold is used instead of base threshold."""
        semantic_cache.similarity_threshold = 0.85

        for i in range(20):
            prompt = f"prompt {i}"
            result = _MockRoutingResult(model="llama3")
            await semantic_cache.set(prompt, result)

        base_calls = [False]

        original_calculate = semantic_cache._calculate_adaptive_threshold

        def mock_calculate(cache_key=None):
            base_calls[0] = True
            return original_calculate(cache_key)

        semantic_cache._calculate_adaptive_threshold = mock_calculate

        prompt1 = "test adaptive threshold"
        result1 = _MockRoutingResult(model="codellama")
        embedding1 = [1.0, 0.5, 0.3, 0.2]
        await semantic_cache.set(prompt1, result1, embedding=embedding1)

        embedding2 = [0.95, 0.55, 0.35, 0.25]
        await semantic_cache.get("another prompt", embedding=embedding2)

        assert base_calls[0] is True


class TestEmbeddingCacheTTL:
    """Tests for embedding cache TTL invalidation."""

    @pytest.fixture
    def semantic_cache(self):
        """Fixture providing SemanticCache with short embedding TTL."""
        cache = SemanticCache(
            max_size=10,
            ttl_seconds=3600,
            embedding_ttl_seconds=1,
            cache_stats_enabled=True,
        )
        return cache

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_expired_embeddings(self, semantic_cache):
        """Test that cleanup_expired() removes expired embedding cache entries."""
        prompt = "test embedding prompt"
        key = semantic_cache._hash_prompt(prompt)

        async with semantic_cache._embedding_lock:
            semantic_cache.embedding_cache[key] = ([0.1, 0.2, 0.3], 1.0, time.time())

        stats = await semantic_cache.get_stats()
        assert stats["embedding"]["size"] == 1

        await asyncio.sleep(1.1)

        expired = await semantic_cache.cleanup_expired()

        assert expired["embedding"] == 1

        stats = await semantic_cache.get_stats()
        assert stats["embedding"]["size"] == 0


class TestCacheInvalidationEdgeCases:
    """Edge case tests for cache invalidation."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_on_empty_cache(self):
        """Test that cleanup_expired() works on empty cache."""
        cache = SemanticCache(ttl_seconds=3600)

        expired = await cache.cleanup_expired()

        assert expired["routing"] == 0
        assert expired["response"] == 0
        assert expired["embedding"] == 0

    @pytest.mark.asyncio
    async def test_mixed_expired_and_fresh_entries(self):
        """Test that cleanup correctly handles mix of expired and fresh entries."""
        cache = SemanticCache(ttl_seconds=2)

        for i in range(3):
            prompt = f"old prompt {i}"
            result = _MockRoutingResult(model=f"old-model-{i}")
            await cache.set(prompt, result)

        await asyncio.sleep(1)

        for i in range(2):
            prompt = f"fresh prompt {i}"
            result = _MockRoutingResult(model=f"fresh-model-{i}")
            await cache.set(prompt, result)

        stats = await cache.get_stats()
        assert stats["routing"]["size"] == 5

        await asyncio.sleep(1.1)

        expired = await cache.cleanup_expired()

        assert expired["routing"] == 3

        stats = await cache.get_stats()
        assert stats["routing"]["size"] == 2

    @pytest.mark.asyncio
    async def test_eviction_count_tracks_ttl_evictions(self):
        """Test that _eviction_counts tracks TTL-based evictions."""
        cache = SemanticCache(ttl_seconds=1, cache_stats_enabled=True)

        prompt = "test prompt"
        result = _MockRoutingResult(model="llama3")
        await cache.set(prompt, result)

        await asyncio.sleep(1.1)

        await cache.cleanup_expired()

        assert "ttl_routing" in cache._eviction_counts
        assert cache._eviction_counts["ttl_routing"] == 1


import asyncio
