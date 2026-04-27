"""
Backend Health Management - Health-aware routing for SmarterRouter.

This module provides health tracking for backends with automatic failover:
- Track consecutive failures per backend
- Automatically mark backends as unhealthy when failure threshold is reached
- Filter unhealthy backends during routing
- Automatic recovery via probing mechanism
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class BackendHealthStatus(Enum):
    """Health status for a backend."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    PROBING = "probing"


@dataclass
class BackendHealthConfig:
    """Configuration for backend health tracking."""

    failure_threshold: int = 3
    recovery_interval: float = 30.0
    probe_attempts: int = 2
    success_streak_reset: int = 3


@dataclass
class BackendHealthState:
    """Per-backend health state."""

    backend_id: str
    status: BackendHealthStatus = BackendHealthStatus.HEALTHY
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: float | None = None
    last_state_change_time: float = field(default_factory=time.monotonic)
    probe_attempts_remaining: int = 0
    total_failures: int = 0
    total_successes: int = 0


class BackendHealthTracker:
    """
    Tracks health state for a single backend.

    Manages:
    - Consecutive failure counting
    - Health state transitions (HEALTHY -> UNHEALTHY -> PROBING -> HEALTHY)
    - Probing mechanism for recovery
    """

    def __init__(
        self,
        backend_id: str,
        config: BackendHealthConfig | None = None,
    ):
        self.backend_id = backend_id
        self.config = config or BackendHealthConfig()
        self._state = BackendHealthState(backend_id=backend_id)
        self._lock = asyncio.Lock()

    @property
    def status(self) -> BackendHealthStatus:
        """Current health status."""
        return self._state.status

    @property
    def is_healthy(self) -> bool:
        """Check if backend is healthy (HEALTHY or PROBING)."""
        return self._state.status in (BackendHealthStatus.HEALTHY, BackendHealthStatus.PROBING)

    def can_accept_request(self) -> bool:
        """
        Check if this backend can accept a request.

        Returns:
            True if HEALTHY or PROBING, False if UNHEALTHY.
            For UNHEALTHY backends, automatically transitions to PROBING
            if recovery_interval has elapsed.
        """
        state = self._state

        if state.status == BackendHealthStatus.HEALTHY:
            return True

        if state.status == BackendHealthStatus.PROBING:
            return True

        if state.status == BackendHealthStatus.UNHEALTHY:
            if state.last_failure_time is not None:
                elapsed = time.monotonic() - state.last_failure_time
                if elapsed >= self.config.recovery_interval:
                    self._transition_to(BackendHealthStatus.PROBING)
                    state.probe_attempts_remaining = self.config.probe_attempts
                    logger.info(
                        f"Backend {self.backend_id} entering PROBING state after "
                        f"{elapsed:.1f}s recovery interval"
                    )
                    return True
            return False

        return False

    async def record_success(self) -> None:
        """Record a successful request to this backend."""
        async with self._lock:
            state = self._state
            state.consecutive_failures = 0
            state.consecutive_successes += 1
            state.total_successes += 1

            if state.status == BackendHealthStatus.PROBING:
                state.probe_attempts_remaining -= 1
                logger.debug(
                    f"Backend {self.backend_id} probe success. "
                    f"Remaining attempts: {state.probe_attempts_remaining}"
                )
                if state.probe_attempts_remaining <= 0:
                    self._transition_to(BackendHealthStatus.HEALTHY)
                    logger.info(f"Backend {self.backend_id} recovered to HEALTHY state")

            elif state.status == BackendHealthStatus.HEALTHY:
                if state.consecutive_successes >= self.config.success_streak_reset:
                    state.consecutive_failures = 0

    async def record_failure(self) -> None:
        """Record a failed request to this backend."""
        async with self._lock:
            state = self._state
            state.consecutive_failures += 1
            state.consecutive_successes = 0
            state.total_failures += 1
            state.last_failure_time = time.monotonic()

            if state.status == BackendHealthStatus.PROBING:
                self._transition_to(BackendHealthStatus.UNHEALTHY)
                logger.warning(
                    f"Backend {self.backend_id} failed during probe, "
                    f"returning to UNHEALTHY state"
                )

            elif state.status == BackendHealthStatus.HEALTHY:
                if state.consecutive_failures >= self.config.failure_threshold:
                    self._transition_to(BackendHealthStatus.UNHEALTHY)
                    logger.warning(
                        f"Backend {self.backend_id} marked UNHEALTHY after "
                        f"{state.consecutive_failures} consecutive failures"
                    )

    def _transition_to(self, new_status: BackendHealthStatus) -> None:
        """Transition to a new health state."""
        old_status = self._state.status
        if old_status == new_status:
            return

        self._state.status = new_status
        self._state.last_state_change_time = time.monotonic()

        if new_status == BackendHealthStatus.HEALTHY:
            self._state.consecutive_failures = 0
            self._state.probe_attempts_remaining = 0
        elif new_status == BackendHealthStatus.UNHEALTHY:
            self._state.consecutive_successes = 0
            self._state.probe_attempts_remaining = 0
        elif new_status == BackendHealthStatus.PROBING:
            self._state.probe_attempts_remaining = self.config.probe_attempts

    def get_stats(self) -> dict[str, Any]:
        """Get health statistics for this backend."""
        state = self._state
        return {
            "backend_id": state.backend_id,
            "status": state.status.value,
            "consecutive_failures": state.consecutive_failures,
            "consecutive_successes": state.consecutive_successes,
            "last_failure_time": state.last_failure_time,
            "last_state_change_time": state.last_state_change_time,
            "probe_attempts_remaining": state.probe_attempts_remaining,
            "total_failures": state.total_failures,
            "total_successes": state.total_successes,
            "is_healthy": self.is_healthy,
            "can_accept_request": self.can_accept_request(),
        }


class BackendHealthManager:
    """
    Manages health tracking for all backends.

    Provides:
    - Global registry of backend health trackers
    - Filtering of unhealthy backends
    - Health status reporting
    """

    def __init__(self, config: BackendHealthConfig | None = None):
        self._config = config or BackendHealthConfig()
        self._trackers: dict[str, BackendHealthTracker] = {}
        self._lock = asyncio.Lock()

    def _get_tracker(self, backend_id: str) -> BackendHealthTracker:
        """Get or create a health tracker for a backend."""
        if backend_id not in self._trackers:
            self._trackers[backend_id] = BackendHealthTracker(
                backend_id=backend_id,
                config=self._config,
            )
        return self._trackers[backend_id]

    async def record_success(self, backend_id: str) -> None:
        """Record a successful request to a backend."""
        async with self._lock:
            tracker = self._get_tracker(backend_id)
        await tracker.record_success()

    async def record_failure(self, backend_id: str) -> None:
        """Record a failed request to a backend."""
        async with self._lock:
            tracker = self._get_tracker(backend_id)
        await tracker.record_failure()

    def is_backend_healthy(self, backend_id: str) -> bool:
        """Check if a backend is healthy."""
        if backend_id not in self._trackers:
            return True
        return self._trackers[backend_id].is_healthy

    def can_accept_request(self, backend_id: str) -> bool:
        """Check if a backend can accept requests."""
        if backend_id not in self._trackers:
            return True
        return self._trackers[backend_id].can_accept_request()

    def filter_healthy_backends(self, backend_ids: list[str]) -> list[str]:
        """
        Filter a list of backend IDs, returning only those that can accept requests.

        If all backends are unhealthy, returns all backends (fail-open).
        """
        if not backend_ids:
            return []

        healthy = [bid for bid in backend_ids if self.can_accept_request(bid)]

        if healthy:
            return healthy

        logger.warning(
            f"All backends are unhealthy: {backend_ids}. "
            f"Failing open to allow requests through."
        )
        return backend_ids

    def get_backend_stats(self, backend_id: str) -> dict[str, Any] | None:
        """Get health statistics for a specific backend."""
        if backend_id not in self._trackers:
            return None
        return self._trackers[backend_id].get_stats()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get health statistics for all tracked backends."""
        return {bid: tracker.get_stats() for bid, tracker in self._trackers.items()}

    async def reset_backend(self, backend_id: str) -> None:
        """Reset a backend's health state to HEALTHY."""
        async with self._lock:
            if backend_id in self._trackers:
                tracker = self._trackers[backend_id]
                tracker._state.status = BackendHealthStatus.HEALTHY
                tracker._state.consecutive_failures = 0
                tracker._state.consecutive_successes = 0
                tracker._state.last_failure_time = None
                tracker._state.probe_attempts_remaining = 0
                logger.info(f"Backend {backend_id} health state reset to HEALTHY")

    async def reset_all(self) -> None:
        """Reset all backends' health states to HEALTHY."""
        async with self._lock:
            for tracker in self._trackers.values():
                tracker._state.status = BackendHealthStatus.HEALTHY
                tracker._state.consecutive_failures = 0
                tracker._state.consecutive_successes = 0
                tracker._state.last_failure_time = None
                tracker._state.probe_attempts_remaining = 0
            logger.info("All backend health states reset to HEALTHY")


_global_health_manager: BackendHealthManager | None = None


def get_health_manager(config: BackendHealthConfig | None = None) -> BackendHealthManager:
    """Get the global backend health manager instance.

    Args:
        config: Optional configuration. Only used when creating a new manager.
            If the manager already exists, this config is ignored.

    Returns:
        The global BackendHealthManager instance.
    """
    global _global_health_manager
    if _global_health_manager is None:
        _global_health_manager = BackendHealthManager(config)
    return _global_health_manager


def reset_health_manager() -> None:
    """Reset the global health manager.

    This is primarily used in tests to ensure isolation between test cases.
    """
    global _global_health_manager
    _global_health_manager = None


def get_backend_identifier(backend_type: str, provider: str | None = None) -> str:
    """
    Generate a unique identifier for a backend.

    Args:
        backend_type: "local" or "external"
        provider: Provider name for external backends (e.g., "openai", "anthropic")

    Returns:
        Unique backend identifier string
    """
    if backend_type == "external" and provider:
        return f"external:{provider}"
    return f"local:{backend_type}"


class BackendUnhealthyError(Exception):
    """Raised when a backend is unhealthy and cannot accept requests."""

    def __init__(self, backend_id: str, message: str = "Backend is unhealthy"):
        super().__init__(f"{backend_id}: {message}")
        self.backend_id = backend_id
