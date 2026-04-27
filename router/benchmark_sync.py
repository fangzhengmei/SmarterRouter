import asyncio
import logging
from typing import Any

from router.benchmark_db import (
    bulk_upsert_benchmarks,
    update_sync_status,
)
from router.providers.artificial_analysis import ArtificialAnalysisProvider
from router.providers.base import BenchmarkProvider
from router.providers.huggingface import HuggingFaceProvider
from router.providers.lmsys import LMSYSProvider

logger = logging.getLogger(__name__)

PROVIDER_PRIORITY: dict[str, int] = {
    "artificial_analysis": 1,
    "huggingface": 2,
    "lmsys": 3,
}

FIELD_PRIORITY_OVERRIDES: dict[str, list[str]] = {
    "parameters": ["huggingface", "artificial_analysis"],
    "elo_rating": ["lmsys"],
    "throughput": ["artificial_analysis"],
    "extra_data": ["artificial_analysis"],
}

EXCLUDE_FIELDS_FROM_MERGE = {"ollama_name"}


def get_field_priority(field: str) -> list[str]:
    """Get the priority order of providers for a specific field.

    Returns a list of provider names in priority order (highest first).
    If no override exists, uses the default PROVIDER_PRIORITY order.
    """
    if field in FIELD_PRIORITY_OVERRIDES:
        return FIELD_PRIORITY_OVERRIDES[field]
    return sorted(PROVIDER_PRIORITY.keys(), key=lambda p: PROVIDER_PRIORITY[p])


def merge_benchmark_data(
    provider_data: list[tuple[str, list[dict[str, Any]]]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Merge benchmark data from multiple providers with conflict detection.

    Args:
        provider_data: List of (provider_name, data_list) tuples

    Returns:
        Tuple of:
        - merged_data: Dict keyed by model name containing merged benchmark data
        - conflicts: Dict keyed by model name containing conflict details
          Format: {model_name: {field_name: {provider_name: value}}}
    """
    all_data: dict[str, dict[str, Any]] = {}
    field_sources: dict[str, dict[str, dict[str, Any]]] = {}
    conflicts: dict[str, dict[str, dict[str, Any]]] = {}

    for provider_name, data in provider_data:
        if provider_name not in PROVIDER_PRIORITY:
            logger.warning(f"Unknown provider: {provider_name}, skipping")
            continue

        for item in data:
            model_name = item.get("ollama_name")
            if not model_name:
                continue

            if model_name not in all_data:
                all_data[model_name] = {}
                field_sources[model_name] = {}

            for field, value in item.items():
                if field in EXCLUDE_FIELDS_FROM_MERGE:
                    continue

                if value is None:
                    continue

                if field not in field_sources[model_name]:
                    field_sources[model_name][field] = {}

                field_sources[model_name][field][provider_name] = value

    for model_name, fields in field_sources.items():
        for field, sources in fields.items():
            if len(sources) == 1:
                provider = list(sources.keys())[0]
                all_data[model_name][field] = sources[provider]
                continue

            unique_values = {}
            for provider, value in sources.items():
                value_key = str(value) if not isinstance(value, dict) else str(sorted(value.items()))
                if value_key not in unique_values:
                    unique_values[value_key] = []
                unique_values[value_key].append((provider, value))

            if len(unique_values) > 1:
                if model_name not in conflicts:
                    conflicts[model_name] = {}
                conflicts[model_name][field] = dict(sources.items())

                conflict_details = []
                for value_key, providers_with_value in unique_values.items():
                    provider_names = [p for p, _ in providers_with_value]
                    sample_value = providers_with_value[0][1]
                    conflict_details.append(f"value={sample_value} from {provider_names}")

                logger.debug(
                    f"Conflict detected for model '{model_name}', field '{field}': {'; '.join(conflict_details)}"
                )

            priority_order = get_field_priority(field)

            selected_value = None
            selected_provider = None
            for provider in priority_order:
                if provider in sources:
                    selected_value = sources[provider]
                    selected_provider = provider
                    break

            if selected_value is not None:
                all_data[model_name][field] = selected_value

                if len(unique_values) > 1:
                    logger.info(
                        f"Resolved conflict for '{model_name}'.'{field}': "
                        f"selected value from '{selected_provider}' (priority: {PROVIDER_PRIORITY.get(selected_provider, 'unknown')})"
                    )

    for model_name in list(all_data.keys()):
        all_data[model_name]["ollama_name"] = model_name

    return all_data, conflicts


async def sync_benchmarks(ollama_models: list[str]) -> tuple[int, list[str]]:
    """Sync benchmarks from all enabled providers in parallel.

    Args:
        ollama_models: List of model names to match against benchmarks

    Returns:
        Tuple of (count of synced models, list of matched model names)
    """
    logger.info("Starting benchmark sync")
    update_sync_status("running", 0)

    from router.config import settings

    enabled_sources = [s.strip().lower() for s in settings.benchmark_sources.split(",")]

    providers: list[BenchmarkProvider] = []
    if "huggingface" in enabled_sources:
        providers.append(HuggingFaceProvider())
    if "lmsys" in enabled_sources:
        providers.append(LMSYSProvider())
    if "artificial_analysis" in enabled_sources:
        providers.append(ArtificialAnalysisProvider())

    async def fetch_provider_data(provider: BenchmarkProvider) -> tuple[str, list[dict[str, Any]]]:
        """Fetch data from a single provider with timeout protection.

        Returns:
            Tuple of (provider_name, data_list)
        """
        try:
            data = await asyncio.wait_for(provider.fetch_data(ollama_models), timeout=120.0)
            return (provider.name, data)
        except TimeoutError:
            logger.error(f"Provider {provider.name} timed out after 120s")
            return (provider.name, [])
        except Exception as e:
            logger.error(f"Provider {provider.name} failed: {e}")
            return (provider.name, [])

    results = await asyncio.gather(*[fetch_provider_data(p) for p in providers])

    for provider_name, data in results:
        logger.info(f"Provider {provider_name} returned {len(data)} records")

    all_data, conflicts = merge_benchmark_data(results)

    if conflicts:
        total_conflicts = sum(len(fields) for fields in conflicts.values())
        logger.info(
            f"Detected {total_conflicts} field conflicts across {len(conflicts)} models during merge"
        )
        for model_name, field_conflicts in conflicts.items():
            for field_name, sources in field_conflicts.items():
                logger.debug(
                    f"Conflict: model='{model_name}', field='{field_name}', sources={sources}"
                )

    matched_models = set(all_data.keys())
    final_benchmarks = list(all_data.values())

    count = bulk_upsert_benchmarks(final_benchmarks)

    update_sync_status("completed", count)
    logger.info(f"Benchmark sync completed: {count} models synced")

    # Invalidate merged benchmarks cache to force refresh on next request
    try:
        from router.router import _MERGED_BENCHMARKS_CACHE
        _MERGED_BENCHMARKS_CACHE.clear()
    except ImportError:
        pass  # Module not loaded yet, cache will be clear on next use

    return count, list(matched_models)
