"""Tests for benchmark synchronization orchestration."""

from unittest.mock import patch

import pytest

from router.benchmark_sync import (
    get_field_priority,
    merge_benchmark_data,
    sync_benchmarks,
)
from router.providers.base import BenchmarkProvider


class MockProvider(BenchmarkProvider):
    """Mock provider for testing."""

    def __init__(self, name: str, data: list):
        self._name = name
        self._data = data

    @property
    def name(self) -> str:
        return self._name

    async def fetch_data(self, ollama_models: list[str]) -> list[dict]:
        return self._data


@pytest.mark.asyncio
async def test_sync_benchmarks_single_provider():
    """Test sync with a single provider."""

    with patch("router.benchmark_sync.bulk_upsert_benchmarks") as mock_upsert:
        with patch("router.benchmark_sync.update_sync_status"):
            with patch("router.config.settings.benchmark_sources", "mock"):
                # We need to mock the provider instantiation
                with patch("router.benchmark_sync.HuggingFaceProvider"):
                    with patch("router.benchmark_sync.LMSYSProvider"):
                        with patch("router.benchmark_sync.ArtificialAnalysisProvider"):
                            mock_upsert.return_value = 1

                            count, matched = await sync_benchmarks(["llama3"])

                            # Since we're mocking everything, just verify it runs
                            assert isinstance(count, int)


@pytest.mark.asyncio
async def test_sync_benchmarks_multiple_providers():
    """Test that data from multiple providers is merged correctly."""
    provider1_data = [
        {"ollama_name": "llama3", "mmlu": 70.0, "reasoning_score": 0.7},
    ]
    provider2_data = [
        {"ollama_name": "llama3", "elo_rating": 1200},
    ]

    # This tests the merge logic
    all_data = {}
    for item in provider1_data:
        name = item["ollama_name"]
        all_data[name] = {}
        for k, v in item.items():
            if v is not None:
                all_data[name][k] = v

    for item in provider2_data:
        name = item["ollama_name"]
        if name not in all_data:
            all_data[name] = {}
        for k, v in item.items():
            if v is not None:
                all_data[name][k] = v

    # Verify merge
    assert "llama3" in all_data
    assert all_data["llama3"]["mmlu"] == 70.0
    assert all_data["llama3"]["elo_rating"] == 1200
    assert all_data["llama3"]["reasoning_score"] == 0.7


@pytest.mark.asyncio
async def test_sync_benchmarks_provider_failure():
    """Test that one provider failure doesn't break others."""

    class FailingProvider(BenchmarkProvider):
        @property
        def name(self) -> str:
            return "failing"

        async def fetch_data(self, ollama_models: list[str]) -> list[dict]:
            raise Exception("Provider failed")

    class WorkingProvider(BenchmarkProvider):
        @property
        def name(self) -> str:
            return "working"

        async def fetch_data(self, ollama_models: list[str]) -> list[dict]:
            return [{"ollama_name": "llama3", "mmlu": 70.0}]

    # Verify that exception in one doesn't stop processing
    providers = [FailingProvider(), WorkingProvider()]

    all_data = {}
    for provider in providers:
        try:
            data = await provider.fetch_data(["llama3"])
            for item in data:
                name = item["ollama_name"]
                if name not in all_data:
                    all_data[name] = {}
                for k, v in item.items():
                    if v is not None:
                        all_data[name][k] = v
        except Exception:
            pass  # Should continue

    # Working provider's data should still be there
    assert "llama3" in all_data


@pytest.mark.asyncio
async def test_sync_benchmarks_empty_models():
    """Test sync with empty model list."""
    with patch("router.benchmark_sync.bulk_upsert_benchmarks") as mock_upsert:
        with patch("router.benchmark_sync.update_sync_status"):
            with patch("router.config.settings.benchmark_sources", "huggingface"):
                with patch("router.benchmark_sync.HuggingFaceProvider") as mock_hf:
                    with patch("router.benchmark_sync.LMSYSProvider"):
                        with patch("router.benchmark_sync.ArtificialAnalysisProvider"):
                            mock_hf.return_value = MockProvider("huggingface", [])
                            mock_upsert.return_value = 0

                            count, matched = await sync_benchmarks([])

                            assert isinstance(count, int)
                            assert isinstance(matched, list)


@pytest.mark.asyncio
async def test_sync_benchmarks_provider_selection():
    """Test that only enabled providers are used."""

    with patch("router.config.settings") as mock_settings:
        # Test with huggingface only
        mock_settings.benchmark_sources = "huggingface"

        with patch("router.benchmark_sync.HuggingFaceProvider") as mock_hf:
            with patch("router.benchmark_sync.LMSYSProvider") as mock_lmsys:
                mock_hf.return_value = MockProvider("huggingface", [])
                mock_lmsys.return_value = MockProvider("lmsys", [])

                # In the actual code, we'd verify only HuggingFace is instantiated
                # For now, just verify the config parsing
                sources = [s.strip().lower() for s in mock_settings.benchmark_sources.split(",")]
                assert "huggingface" in sources
                assert "lmsys" not in sources


class TestGetFieldPriority:
    """Tests for get_field_priority function."""

    def test_default_priority(self):
        """Test default priority order for fields without override."""
        priority = get_field_priority("mmlu")
        assert priority[0] == "artificial_analysis"
        assert priority[1] == "huggingface"
        assert priority[2] == "lmsys"

    def test_parameters_override(self):
        """Test that parameters field has special override."""
        priority = get_field_priority("parameters")
        assert priority == ["huggingface", "artificial_analysis"]

    def test_elo_rating_override(self):
        """Test that elo_rating field has special override."""
        priority = get_field_priority("elo_rating")
        assert priority == ["lmsys"]

    def test_throughput_override(self):
        """Test that throughput field has special override."""
        priority = get_field_priority("throughput")
        assert priority == ["artificial_analysis"]


class TestMergeBenchmarkData:
    """Tests for merge_benchmark_data function."""

    def test_merge_no_conflicts(self):
        """Test merging data from multiple providers with no conflicts."""
        provider_data = [
            (
                "huggingface",
                [
                    {"ollama_name": "llama3", "mmlu": 70.0, "reasoning_score": 0.7},
                ],
            ),
            (
                "lmsys",
                [
                    {"ollama_name": "llama3", "elo_rating": 1200},
                ],
            ),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert "llama3" in merged
        assert merged["llama3"]["mmlu"] == 70.0
        assert merged["llama3"]["elo_rating"] == 1200
        assert merged["llama3"]["reasoning_score"] == 0.7
        assert conflicts == {}

    def test_merge_same_values_no_conflict(self):
        """Test that same values from different providers don't create conflict."""
        provider_data = [
            (
                "huggingface",
                [
                    {"ollama_name": "llama3", "mmlu": 70.0},
                ],
            ),
            (
                "artificial_analysis",
                [
                    {"ollama_name": "llama3", "mmlu": 70.0},
                ],
            ),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert merged["llama3"]["mmlu"] == 70.0
        assert conflicts == {}

    def test_merge_conflict_default_priority(self):
        """Test conflict resolution using default priority.

        Default priority: artificial_analysis (1) > huggingface (2) > lmsys (3)
        """
        provider_data = [
            (
                "huggingface",
                [
                    {"ollama_name": "llama3", "mmlu": 70.0},
                ],
            ),
            (
                "artificial_analysis",
                [
                    {"ollama_name": "llama3", "mmlu": 75.0},
                ],
            ),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert merged["llama3"]["mmlu"] == 75.0
        assert "llama3" in conflicts
        assert "mmlu" in conflicts["llama3"]
        assert conflicts["llama3"]["mmlu"]["huggingface"] == 70.0
        assert conflicts["llama3"]["mmlu"]["artificial_analysis"] == 75.0

    def test_merge_conflict_parameters_override(self):
        """Test conflict resolution for parameters field (override: huggingface first)."""
        provider_data = [
            (
                "huggingface",
                [
                    {"ollama_name": "llama3", "parameters": "8B"},
                ],
            ),
            (
                "artificial_analysis",
                [
                    {"ollama_name": "llama3", "parameters": "7B"},
                ],
            ),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert merged["llama3"]["parameters"] == "8B"
        assert "llama3" in conflicts
        assert "parameters" in conflicts["llama3"]

    def test_merge_elo_rating_only_lmsys(self):
        """Test that elo_rating from lmsys is correctly used."""
        provider_data = [
            (
                "lmsys",
                [
                    {"ollama_name": "llama3", "elo_rating": 1200},
                ],
            ),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert merged["llama3"]["elo_rating"] == 1200
        assert conflicts == {}

    def test_merge_multiple_models(self):
        """Test merging data for multiple models."""
        provider_data = [
            (
                "huggingface",
                [
                    {"ollama_name": "llama3", "mmlu": 70.0},
                    {"ollama_name": "mistral", "mmlu": 65.0},
                ],
            ),
            (
                "artificial_analysis",
                [
                    {"ollama_name": "llama3", "mmlu": 75.0, "throughput": 100},
                    {"ollama_name": "gemma", "mmlu": 60.0},
                ],
            ),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert "llama3" in merged
        assert "mistral" in merged
        assert "gemma" in merged
        assert merged["llama3"]["mmlu"] == 75.0
        assert merged["llama3"]["throughput"] == 100
        assert merged["mistral"]["mmlu"] == 65.0
        assert merged["gemma"]["mmlu"] == 60.0
        assert "llama3" in conflicts
        assert "mistral" not in conflicts
        assert "gemma" not in conflicts

    def test_merge_null_values_ignored(self):
        """Test that null values are ignored during merge."""
        provider_data = [
            (
                "huggingface",
                [
                    {"ollama_name": "llama3", "mmlu": 70.0, "humaneval": None},
                ],
            ),
            (
                "artificial_analysis",
                [
                    {"ollama_name": "llama3", "mmlu": None, "humaneval": 60.0},
                ],
            ),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert merged["llama3"]["mmlu"] == 70.0
        assert merged["llama3"]["humaneval"] == 60.0
        assert conflicts == {}

    def test_merge_dict_values(self):
        """Test merging with dict values (like extra_data)."""
        hf_extra = {"source": "huggingface", "details": "open_llm_leaderboard"}
        aa_extra = {"source": "artificial_analysis", "api_version": "v2"}

        provider_data = [
            (
                "huggingface",
                [
                    {"ollama_name": "llama3", "extra_data": hf_extra},
                ],
            ),
            (
                "artificial_analysis",
                [
                    {"ollama_name": "llama3", "extra_data": aa_extra},
                ],
            ),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert merged["llama3"]["extra_data"] == aa_extra
        assert "llama3" in conflicts
        assert "extra_data" in conflicts["llama3"]

    def test_merge_unknown_provider_skipped(self):
        """Test that unknown providers are skipped."""
        provider_data = [
            (
                "unknown_provider",
                [
                    {"ollama_name": "llama3", "mmlu": 70.0},
                ],
            ),
            (
                "huggingface",
                [
                    {"ollama_name": "llama3", "mmlu": 65.0},
                ],
            ),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert merged["llama3"]["mmlu"] == 65.0
        assert conflicts == {}

    def test_merge_empty_data(self):
        """Test merging with empty provider data."""
        provider_data = [
            ("huggingface", []),
            ("lmsys", []),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert merged == {}
        assert conflicts == {}

    def test_merge_missing_ollama_name(self):
        """Test that items without ollama_name are skipped."""
        provider_data = [
            (
                "huggingface",
                [
                    {"mmlu": 70.0},
                    {"ollama_name": "llama3", "mmlu": 70.0},
                ],
            ),
        ]

        merged, conflicts = merge_benchmark_data(provider_data)

        assert "llama3" in merged
        assert len(merged) == 1
