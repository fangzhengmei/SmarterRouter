"""
Tests for backend health-aware routing (Item #54).

Tests that:
- Backends are tracked for consecutive failures
- Backends are marked unhealthy after failure threshold
- Unhealthy backends are filtered during routing
- Automatic recovery via probing mechanism
- Fail-open strategy when all backends are unhealthy
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from router.backends.health import (
    BackendHealthConfig,
    BackendHealthManager,
    BackendHealthStatus,
    BackendHealthTracker,
    BackendUnhealthyError,
    get_backend_identifier,
    get_health_manager,
    reset_health_manager,
)
from router.backends import resilience
from router.backends.resilience import (
    build_backend_id_for_model,
    get_health_manager_from_config,
    with_backend_resilience,
)


class TestBackendHealthTracker:
    """Tests for BackendHealthTracker class."""

    @pytest.mark.asyncio
    async def test_initial_state_is_healthy(self):
        """Tracker should start in HEALTHY state."""
        tracker = BackendHealthTracker("test-backend")
        assert tracker.status == BackendHealthStatus.HEALTHY
        assert tracker.is_healthy is True
        assert tracker.can_accept_request() is True

    @pytest.mark.asyncio
    async def test_consecutive_failures_mark_unhealthy(self):
        """Consecutive failures should mark backend as UNHEALTHY."""
        config = BackendHealthConfig(failure_threshold=3)
        tracker = BackendHealthTracker("test-backend", config)

        for i in range(2):
            await tracker.record_failure()
            assert tracker.status == BackendHealthStatus.HEALTHY
            assert tracker.can_accept_request() is True

        await tracker.record_failure()
        assert tracker.status == BackendHealthStatus.UNHEALTHY
        assert tracker.is_healthy is False
        assert tracker.can_accept_request() is False

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        """Consecutive successes should reset failure count."""
        config = BackendHealthConfig(failure_threshold=3, success_streak_reset=2)
        tracker = BackendHealthTracker("test-backend", config)

        await tracker.record_failure()
        await tracker.record_failure()

        await tracker.record_success()
        await tracker.record_success()

        await tracker.record_failure()
        assert tracker.status == BackendHealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_backend_rejects_requests(self):
        """Unhealthy backend should reject requests."""
        config = BackendHealthConfig(failure_threshold=2, recovery_interval=60.0)
        tracker = BackendHealthTracker("test-backend", config)

        await tracker.record_failure()
        await tracker.record_failure()

        assert tracker.status == BackendHealthStatus.UNHEALTHY
        assert tracker.can_accept_request() is False

    @pytest.mark.asyncio
    async def test_probing_after_recovery_interval(self):
        """After recovery interval, backend should enter PROBING state."""
        config = BackendHealthConfig(failure_threshold=2, recovery_interval=0.1)
        tracker = BackendHealthTracker("test-backend", config)

        await tracker.record_failure()
        await tracker.record_failure()
        assert tracker.status == BackendHealthStatus.UNHEALTHY

        await asyncio.sleep(0.15)

        can_accept = tracker.can_accept_request()
        assert can_accept is True
        assert tracker.status == BackendHealthStatus.PROBING

    @pytest.mark.asyncio
    async def test_probing_success_recovers_to_healthy(self):
        """Successful probes should recover backend to HEALTHY."""
        config = BackendHealthConfig(failure_threshold=2, probe_attempts=2)
        tracker = BackendHealthTracker("test-backend", config)

        await tracker.record_failure()
        await tracker.record_failure()
        tracker._transition_to(BackendHealthStatus.PROBING)
        tracker._state.probe_attempts_remaining = 2

        await tracker.record_success()
        assert tracker.status == BackendHealthStatus.PROBING

        await tracker.record_success()
        assert tracker.status == BackendHealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_probing_failure_returns_to_unhealthy(self):
        """Failure during probing should return backend to UNHEALTHY."""
        config = BackendHealthConfig(failure_threshold=2)
        tracker = BackendHealthTracker("test-backend", config)

        tracker._transition_to(BackendHealthStatus.PROBING)

        await tracker.record_failure()
        assert tracker.status == BackendHealthStatus.UNHEALTHY

    def test_get_stats_returns_correct_info(self):
        """get_stats should return correct health information."""
        config = BackendHealthConfig(failure_threshold=3)
        tracker = BackendHealthTracker("test-backend", config)

        stats = tracker.get_stats()
        assert stats["backend_id"] == "test-backend"
        assert stats["status"] == "healthy"
        assert stats["consecutive_failures"] == 0
        assert stats["is_healthy"] is True


class TestBackendHealthManager:
    """Tests for BackendHealthManager class."""

    @pytest.mark.asyncio
    async def test_record_success_and_failure(self):
        """Manager should track health for multiple backends."""
        manager = BackendHealthManager(BackendHealthConfig(failure_threshold=2))

        await manager.record_success("backend-a")
        await manager.record_success("backend-a")

        await manager.record_failure("backend-b")
        await manager.record_failure("backend-b")

        assert manager.is_backend_healthy("backend-a") is True
        assert manager.can_accept_request("backend-a") is True

        assert manager.is_backend_healthy("backend-b") is False
        assert manager.can_accept_request("backend-b") is False

    def test_filter_healthy_backends(self):
        """filter_healthy_backends should return only healthy backends."""
        manager = BackendHealthManager(BackendHealthConfig(failure_threshold=2))

        manager._get_tracker("backend-a")
        manager._get_tracker("backend-b")
        manager._get_tracker("backend-c")

        manager._trackers["backend-b"]._transition_to(BackendHealthStatus.UNHEALTHY)

        result = manager.filter_healthy_backends(["backend-a", "backend-b", "backend-c"])
        assert "backend-a" in result
        assert "backend-b" not in result
        assert "backend-c" in result

    def test_filter_healthy_backends_fail_open(self):
        """When all backends are unhealthy, should return all (fail-open)."""
        manager = BackendHealthManager(BackendHealthConfig(failure_threshold=2))

        manager._get_tracker("backend-a")
        manager._get_tracker("backend-b")

        manager._trackers["backend-a"]._transition_to(BackendHealthStatus.UNHEALTHY)
        manager._trackers["backend-b"]._transition_to(BackendHealthStatus.UNHEALTHY)

        result = manager.filter_healthy_backends(["backend-a", "backend-b"])
        assert len(result) == 2
        assert "backend-a" in result
        assert "backend-b" in result

    @pytest.mark.asyncio
    async def test_reset_backend(self):
        """reset_backend should restore backend to HEALTHY."""
        manager = BackendHealthManager(BackendHealthConfig(failure_threshold=2))

        await manager.record_failure("backend-a")
        await manager.record_failure("backend-a")
        assert manager.can_accept_request("backend-a") is False

        await manager.reset_backend("backend-a")
        assert manager.can_accept_request("backend-a") is True

    def test_get_all_stats(self):
        """get_all_stats should return stats for all tracked backends."""
        manager = BackendHealthManager()

        manager._get_tracker("backend-a")
        manager._get_tracker("backend-b")

        stats = manager.get_all_stats()
        assert "backend-a" in stats
        assert "backend-b" in stats


class TestBackendIdentifier:
    """Tests for backend identifier generation."""

    def test_get_backend_identifier_local(self):
        """Local backends should have correct identifier format."""
        result = get_backend_identifier("local", "default")
        assert result == "local:local"

    def test_get_backend_identifier_external(self):
        """External backends should have correct identifier format."""
        result = get_backend_identifier("external", "openai")
        assert result == "external:openai"

        result = get_backend_identifier("external", "anthropic")
        assert result == "external:anthropic"

    def test_build_backend_id_for_model_local(self):
        """Local models (no slash) should use local identifier."""
        result = build_backend_id_for_model("llama2")
        assert result == "local:local"

    def test_build_backend_id_for_model_external(self):
        """External models (with slash) should extract provider."""
        result = build_backend_id_for_model("openai/gpt-4", backend_type="external")
        assert result == "external:openai"

        result = build_backend_id_for_model("anthropic/claude-3-opus", backend_type="external")
        assert result == "external:anthropic"

    def test_build_backend_id_for_model_with_provider(self):
        """Explicit provider should override model name parsing."""
        result = build_backend_id_for_model("any-model", provider="mistral")
        assert result == "external:mistral"


class TestHealthIntegrationWithResilience:
    """Tests for integration between health tracking and resilience module."""

    def setup_method(self):
        """Reset global health managers before each test."""
        resilience._health_manager = None
        reset_health_manager()

    @pytest.mark.asyncio
    async def test_with_backend_resilience_tracks_health(self):
        """with_backend_resilience should record success/failure to health manager."""
        config = MagicMock()
        config.backend_retry_enabled = False
        config.backend_circuit_breaker_enabled = False
        config.backend_health_enabled = True
        config.backend_health_failure_threshold = 2
        config.backend_health_recovery_interval = 30.0
        config.backend_health_probe_attempts = 2
        config.backend_health_success_streak_reset = 3

        async def successful_operation():
            return {"result": "success"}

        result = await with_backend_resilience(
            operation_name="test",
            operation=successful_operation,
            config=config,
            backend_id="test-backend",
        )
        assert result["result"] == "success"

        manager = get_health_manager_from_config(config)
        stats = manager.get_backend_stats("test-backend")
        assert stats is not None
        assert stats["total_successes"] == 1

    @pytest.mark.asyncio
    async def test_with_backend_resilience_records_failure(self):
        """Failed operations should be recorded to health manager."""
        config = MagicMock()
        config.backend_retry_enabled = False
        config.backend_circuit_breaker_enabled = False
        config.backend_health_enabled = True
        config.backend_health_failure_threshold = 2
        config.backend_health_recovery_interval = 30.0
        config.backend_health_probe_attempts = 2
        config.backend_health_success_streak_reset = 3

        async def failing_operation():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await with_backend_resilience(
                operation_name="test",
                operation=failing_operation,
                config=config,
                backend_id="test-backend",
            )

        manager = get_health_manager_from_config(config)
        stats = manager.get_backend_stats("test-backend")
        assert stats is not None
        assert stats["total_failures"] == 1

    @pytest.mark.asyncio
    async def test_with_backend_resilience_rejects_unhealthy_backend(self):
        """Unhealthy backends should raise BackendUnhealthyError."""
        class TestConfig:
            backend_retry_enabled = False
            backend_circuit_breaker_enabled = False
            backend_health_enabled = True
            backend_health_failure_threshold = 1
            backend_health_recovery_interval = 30.0
            backend_health_probe_attempts = 2
            backend_health_success_streak_reset = 3

        config = TestConfig()
        manager = get_health_manager_from_config(config)
        await manager.record_failure("unhealthy-backend")

        stats = manager.get_backend_stats("unhealthy-backend")
        assert stats["status"] == "unhealthy"

        async def operation():
            return {"result": "should not be called"}

        with pytest.raises(BackendUnhealthyError) as exc_info:
            await with_backend_resilience(
                operation_name="test",
                operation=operation,
                config=config,
                backend_id="unhealthy-backend",
            )

        assert "unhealthy-backend" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_with_backend_resilience_skips_health_without_backend_id(self):
        """Without backend_id, health tracking should be skipped."""
        config = MagicMock()
        config.backend_retry_enabled = False
        config.backend_circuit_breaker_enabled = False
        config.backend_health_enabled = True

        async def successful_operation():
            return {"result": "success"}

        result = await with_backend_resilience(
            operation_name="test",
            operation=successful_operation,
            config=config,
            backend_id=None,
        )
        assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_with_backend_resilience_respects_health_enabled_flag(self):
        """When backend_health_enabled is False, health checks should be skipped."""
        config = MagicMock()
        config.backend_retry_enabled = False
        config.backend_circuit_breaker_enabled = False
        config.backend_health_enabled = False

        async def successful_operation():
            return {"result": "success"}

        result = await with_backend_resilience(
            operation_name="test",
            operation=successful_operation,
            config=config,
            backend_id="test-backend",
        )
        assert result["result"] == "success"


class TestHealthRoutingScenarios:
    """Integration tests for health-aware routing scenarios."""

    @pytest.mark.asyncio
    async def test_health_based_failover_scenario(self):
        """Test complete failover scenario: healthy -> failing -> unhealthy -> recover."""
        config = BackendHealthConfig(
            failure_threshold=2,
            recovery_interval=0.1,
            probe_attempts=1,
        )
        manager = BackendHealthManager(config)

        assert manager.can_accept_request("backend-a") is True
        assert manager.can_accept_request("backend-b") is True

        await manager.record_failure("backend-a")
        await manager.record_failure("backend-a")

        healthy = manager.filter_healthy_backends(["backend-a", "backend-b"])
        assert "backend-a" not in healthy
        assert "backend-b" in healthy

        await asyncio.sleep(0.15)

        healthy = manager.filter_healthy_backends(["backend-a", "backend-b"])
        assert "backend-a" in healthy
        assert "backend-b" in healthy

        await manager.record_success("backend-a")

        stats = manager.get_backend_stats("backend-a")
        assert stats["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_all_backends_unhealthy_fail_open(self):
        """When all backends are unhealthy, should fail open to allow requests."""
        config = BackendHealthConfig(failure_threshold=1)
        manager = BackendHealthManager(config)

        await manager.record_failure("backend-a")
        await manager.record_failure("backend-b")
        await manager.record_failure("backend-c")

        healthy = manager.filter_healthy_backends(["backend-a", "backend-b", "backend-c"])

        assert len(healthy) == 3
        assert "backend-a" in healthy
        assert "backend-b" in healthy
        assert "backend-c" in healthy

    def test_untracked_backends_considered_healthy(self):
        """Backends not yet tracked should be considered healthy."""
        manager = BackendHealthManager()

        assert manager.is_backend_healthy("new-backend") is True
        assert manager.can_accept_request("new-backend") is True

        stats = manager.get_backend_stats("new-backend")
        assert stats is None


class TestBackendUnhealthyError:
    """Tests for BackendUnhealthyError exception."""

    def test_exception_creation(self):
        """Exception should store backend_id and message."""
        error = BackendUnhealthyError("test-backend", "Custom message")

        assert error.backend_id == "test-backend"
        assert "test-backend" in str(error)
        assert "Custom message" in str(error)

    def test_exception_default_message(self):
        """Exception should have a default message."""
        error = BackendUnhealthyError("test-backend")

        assert "Backend is unhealthy" in str(error)
