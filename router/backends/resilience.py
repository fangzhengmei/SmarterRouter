from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from router.backends.health import (
    BackendHealthConfig,
    BackendHealthManager,
    get_backend_identifier,
    get_health_manager,
)
from router.circuit_breaker import CircuitBreakerConfig, get_circuit_breaker

T = TypeVar("T")

_health_manager: BackendHealthManager | None = None


def _get_health_manager_for_config(config: Any) -> BackendHealthManager:
    """Get or create health manager with config-specific settings."""
    global _health_manager
    if _health_manager is None:
        health_config = BackendHealthConfig(
            failure_threshold=getattr(config, "backend_health_failure_threshold", 3),
            recovery_interval=getattr(config, "backend_health_recovery_interval", 30.0),
            probe_attempts=getattr(config, "backend_health_probe_attempts", 2),
            success_streak_reset=getattr(config, "backend_health_success_streak_reset", 3),
        )
        _health_manager = get_health_manager(health_config)
    return _health_manager


async def with_backend_resilience(
    operation_name: str,
    operation: Callable[[], Awaitable[T]],
    config: Any,
    backend_id: str | None = None,
) -> T:
    """Apply circuit breaker, retry policy, and health tracking for backend operations.

    Order matters:
    - Circuit breaker wraps the retried operation so one logical operation
      counts as one success/failure in breaker state.
    - Retry still handles transient errors inside the breaker execution.
    - Health tracking records overall success/failure for the backend.

    Args:
        operation_name: Name of the operation for circuit breaker.
        operation: The async operation to execute.
        config: Configuration settings.
        backend_id: Optional backend identifier for health tracking.
            If not provided, health tracking is skipped.
    """
    from router.backends.retry import retry_operation

    health_enabled = getattr(config, "backend_health_enabled", True)
    health_manager = None
    if health_enabled and backend_id:
        health_manager = _get_health_manager_for_config(config)

    if health_manager and backend_id:
        if not health_manager.can_accept_request(backend_id):
            from router.backends.health import BackendUnhealthyError

            raise BackendUnhealthyError(
                backend_id,
                f"Backend {backend_id} is unhealthy and cannot accept requests",
            )

    async def with_retry() -> T:
        if config.backend_retry_enabled:
            return await retry_operation(
                operation,
                max_retries=config.backend_max_retries,
                base_delay=config.backend_retry_base_delay,
                max_delay=config.backend_retry_max_delay,
            )
        return await operation()

    async def execute_with_health_tracking() -> T:
        try:
            if not config.backend_circuit_breaker_enabled:
                result = await with_retry()
            else:
                breaker = await get_circuit_breaker(
                    operation_name,
                    CircuitBreakerConfig(
                        failure_threshold=config.backend_circuit_breaker_failure_threshold,
                        reset_timeout=config.backend_circuit_breaker_reset_timeout,
                        half_open_max_attempts=config.backend_circuit_breaker_half_open_max_attempts,
                        sliding_window_size=config.backend_circuit_breaker_sliding_window_size,
                    ),
                )
                result = await breaker.execute(with_retry)

            if health_manager and backend_id:
                await health_manager.record_success(backend_id)

            return result

        except Exception:
            if health_manager and backend_id:
                await health_manager.record_failure(backend_id)
            raise

    return await execute_with_health_tracking()


def get_health_manager_from_config(config: Any) -> BackendHealthManager:
    """Get the health manager instance for the given config.

    This is used by routing logic to filter unhealthy backends.
    """
    return _get_health_manager_for_config(config)


def build_backend_id_for_model(
    model_name: str,
    backend_type: str | None = None,
    provider: str | None = None,
) -> str:
    """Build a backend identifier for a model.

    Args:
        model_name: The model name (may contain provider prefix like "openai/gpt-4").
        backend_type: Optional explicit backend type ("local" or "external").
        provider: Optional explicit provider name.

    Returns:
        Backend identifier string.
    """
    if provider:
        return get_backend_identifier("external", provider)

    if backend_type == "external":
        if "/" in model_name:
            provider_from_model = model_name.split("/")[0]
            return get_backend_identifier("external", provider_from_model)
        return get_backend_identifier("external", "unknown")

    return get_backend_identifier("local", "default")
