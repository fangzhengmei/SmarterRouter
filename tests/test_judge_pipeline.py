"""Tests for multi-stage evaluation pipeline in Judge."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from router.judge import (
    ACCURACY_PROMPT,
    CLARITY_PROMPT,
    CONCISENESS_PROMPT,
    DEFAULT_DIMENSION_WEIGHTS,
    DIMENSION_PROMPTS,
    DimensionScore,
    EvaluationDimension,
    EvaluationPipeline,
    EvaluationResult,
    FastFailureDetector,
    JudgeClient,
    _extract_json_from_content,
)


class TestEvaluationDimension:
    """Tests for EvaluationDimension enum."""

    def test_dimension_values(self):
        """Test that all evaluation dimensions have correct values."""
        assert EvaluationDimension.ACCURACY.value == "accuracy"
        assert EvaluationDimension.HELPFULNESS.value == "helpfulness"
        assert EvaluationDimension.CLARITY.value == "clarity"
        assert EvaluationDimension.CONCISENESS.value == "conciseness"
        assert EvaluationDimension.INSTRUCTION_FOLLOWING.value == "instruction_following"

    def test_dimension_count(self):
        """Test that there are exactly 5 evaluation dimensions."""
        assert len(list(EvaluationDimension)) == 5

    def test_dimension_iteration(self):
        """Test iterating over EvaluationDimension enum."""
        dimensions = list(EvaluationDimension)
        expected = [
            EvaluationDimension.ACCURACY,
            EvaluationDimension.HELPFULNESS,
            EvaluationDimension.CLARITY,
            EvaluationDimension.CONCISENESS,
            EvaluationDimension.INSTRUCTION_FOLLOWING,
        ]
        assert dimensions == expected


class TestDimensionScore:
    """Tests for DimensionScore dataclass."""

    def test_creation_defaults(self):
        """Test creating DimensionScore with minimal parameters."""
        score = DimensionScore(
            dimension=EvaluationDimension.ACCURACY,
            score=0.85,
        )
        assert score.dimension == EvaluationDimension.ACCURACY
        assert score.score == 0.85
        assert score.reasoning == ""
        assert score.metadata == {}
        assert isinstance(score.timestamp, datetime)

    def test_creation_full(self):
        """Test creating DimensionScore with all parameters."""
        now = datetime.now()
        score = DimensionScore(
            dimension=EvaluationDimension.HELPFULNESS,
            score=0.9,
            reasoning="Very helpful response",
            metadata={"attempts": 1, "model": "gpt-4o"},
            timestamp=now,
        )
        assert score.dimension == EvaluationDimension.HELPFULNESS
        assert score.score == 0.9
        assert score.reasoning == "Very helpful response"
        assert score.metadata == {"attempts": 1, "model": "gpt-4o"}
        assert score.timestamp == now

    def test_to_dict(self):
        """Test converting DimensionScore to dict."""
        score = DimensionScore(
            dimension=EvaluationDimension.CLARITY,
            score=0.7,
            reasoning="Clear but could be better",
            metadata={"source": "test"},
        )
        result = score.to_dict()

        assert result["dimension"] == "clarity"
        assert result["score"] == 0.7
        assert result["reasoning"] == "Clear but could be better"
        assert result["metadata"] == {"source": "test"}
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation_minimal(self):
        """Test creating EvaluationResult with minimal parameters."""
        dimension_scores = {
            EvaluationDimension.ACCURACY: DimensionScore(
                dimension=EvaluationDimension.ACCURACY, score=0.8
            ),
        }
        result = EvaluationResult(
            overall_score=0.8,
            dimension_scores=dimension_scores,
        )
        assert result.overall_score == 0.8
        assert len(result.dimension_scores) == 1
        assert result.is_fast_failure is False
        assert result.failure_reason == ""
        assert result.metadata == {}

    def test_creation_full(self):
        """Test creating EvaluationResult with all parameters."""
        dimension_scores = {
            dim: DimensionScore(dimension=dim, score=0.5)
            for dim in EvaluationDimension
        }
        now = datetime.now()
        result = EvaluationResult(
            overall_score=0.5,
            dimension_scores=dimension_scores,
            is_fast_failure=True,
            failure_reason="Response too short",
            metadata={"source": "test", "model": "test-model"},
            timestamp=now,
        )
        assert result.overall_score == 0.5
        assert len(result.dimension_scores) == 5
        assert result.is_fast_failure is True
        assert result.failure_reason == "Response too short"
        assert result.metadata == {"source": "test", "model": "test-model"}
        assert result.timestamp == now

    def test_to_dict(self):
        """Test converting EvaluationResult to dict."""
        dimension_scores = {
            EvaluationDimension.ACCURACY: DimensionScore(
                dimension=EvaluationDimension.ACCURACY, score=0.9, reasoning="Good accuracy"
            ),
            EvaluationDimension.HELPFULNESS: DimensionScore(
                dimension=EvaluationDimension.HELPFULNESS, score=0.8, reasoning="Very helpful"
            ),
        }
        result = EvaluationResult(
            overall_score=0.85,
            dimension_scores=dimension_scores,
            is_fast_failure=False,
            metadata={"test": True},
        )
        result_dict = result.to_dict()

        assert result_dict["overall_score"] == 0.85
        assert "accuracy" in result_dict["dimension_scores"]
        assert "helpfulness" in result_dict["dimension_scores"]
        assert result_dict["dimension_scores"]["accuracy"]["score"] == 0.9
        assert result_dict["dimension_scores"]["helpfulness"]["reasoning"] == "Very helpful"
        assert result_dict["is_fast_failure"] is False
        assert result_dict["metadata"]["test"] is True


class TestFastFailureDetector:
    """Tests for FastFailureDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a FastFailureDetector instance."""
        return FastFailureDetector()

    def test_empty_response(self, detector):
        """Test detecting empty response."""
        is_failure, reason = detector.check("")
        assert is_failure is True
        assert "empty" in reason.lower()

    def test_short_response(self, detector):
        """Test detecting very short response."""
        is_failure, reason = detector.check("hi")
        assert is_failure is True
        assert "short" in reason.lower()

    def test_valid_response(self, detector):
        """Test that valid response passes detection."""
        is_failure, reason = detector.check(
            "This is a valid response that is long enough to pass the check."
        )
        assert is_failure is False
        assert reason == ""

    def test_apology_pattern(self, detector):
        """Test detecting apology patterns."""
        is_failure, reason = detector.check(
            "I'm sorry, I cannot help with that request at this time."
        )
        assert is_failure is True

    def test_cannot_help_pattern(self, detector):
        """Test detecting 'cannot help' patterns."""
        is_failure, reason = detector.check(
            "I cannot assist you with that particular question."
        )
        assert is_failure is True

    def test_ai_language_model_pattern(self, detector):
        """Test detecting 'as an AI language model' pattern."""
        is_failure, reason = detector.check(
            "As an AI language model, I can help you with various tasks."
        )
        assert is_failure is True

    def test_no_access_pattern(self, detector):
        """Test detecting 'no access' patterns."""
        is_failure, reason = detector.check(
            "I don't have access to that information right now."
        )
        assert is_failure is True

    def test_unable_to_process_pattern(self, detector):
        """Test detecting 'unable to process' patterns."""
        is_failure, reason = detector.check(
            "I am unable to generate a response for this request."
        )
        assert is_failure is True

    def test_context_window_pattern(self, detector):
        """Test detecting context window exceeded patterns."""
        is_failure, reason = detector.check(
            "Context window exceeded. Please reduce input size."
        )
        assert is_failure is True

    def test_rate_limit_pattern(self, detector):
        """Test detecting rate limit patterns."""
        is_failure, reason = detector.check(
            "Rate limit exceeded. Please try again later."
        )
        assert is_failure is True

    def test_failure_patterns_case_insensitive(self, detector):
        """Test that pattern matching is case-insensitive."""
        is_failure, reason = detector.check(
            "I'M SORRY, BUT I CANNOT HELP WITH THAT."
        )
        assert is_failure is True

    def test_boundary_just_long_enough(self, detector):
        """Test response that is just long enough (20 chars)."""
        response = "This is exac 20chars"
        assert len(response) == 20
        is_failure, reason = detector.check(response)
        assert is_failure is False

    def test_boundary_too_short(self, detector):
        """Test response that is one char too short."""
        response = "This is just 19 cha"
        assert len(response) == 19
        is_failure, reason = detector.check(response)
        assert is_failure is True


class TestEvaluationPipeline:
    """Tests for EvaluationPipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create an EvaluationPipeline with default settings."""
        from router.judge import BaseEvaluator, ACCURACY_PROMPT, HELPFULNESS_PROMPT

        evaluators = [
            BaseEvaluator(ACCURACY_PROMPT, EvaluationDimension.ACCURACY),
            BaseEvaluator(HELPFULNESS_PROMPT, EvaluationDimension.HELPFULNESS),
        ]
        weights = {
            EvaluationDimension.ACCURACY: 0.6,
            EvaluationDimension.HELPFULNESS: 0.4,
        }
        return EvaluationPipeline(evaluators=evaluators, weights=weights)

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        from router.judge import BaseEvaluator, ACCURACY_PROMPT, HELPFULNESS_PROMPT

        evaluators = [
            BaseEvaluator(ACCURACY_PROMPT, EvaluationDimension.ACCURACY),
            BaseEvaluator(HELPFULNESS_PROMPT, EvaluationDimension.HELPFULNESS),
        ]
        weights = {
            EvaluationDimension.ACCURACY: 60,
            EvaluationDimension.HELPFULNESS: 40,
        }
        pipeline = EvaluationPipeline(evaluators=evaluators, weights=weights)

        assert pipeline.weights[EvaluationDimension.ACCURACY] == 0.6
        assert pipeline.weights[EvaluationDimension.HELPFULNESS] == 0.4
        total = sum(pipeline.weights.values())
        assert abs(total - 1.0) < 0.0001

    def test_compute_overall_score(self, pipeline):
        """Test computing overall score from dimension scores."""
        dimension_scores = {
            EvaluationDimension.ACCURACY: DimensionScore(
                dimension=EvaluationDimension.ACCURACY, score=1.0
            ),
            EvaluationDimension.HELPFULNESS: DimensionScore(
                dimension=EvaluationDimension.HELPFULNESS, score=0.5
            ),
        }
        overall = pipeline.compute_overall_score(dimension_scores)

        expected = 1.0 * 0.6 + 0.5 * 0.4
        assert overall == expected

    def test_compute_overall_score_with_missing_dimensions(self):
        """Test computing score when some dimensions are missing."""
        from router.judge import BaseEvaluator, ACCURACY_PROMPT

        evaluators = [BaseEvaluator(ACCURACY_PROMPT, EvaluationDimension.ACCURACY)]
        weights = {EvaluationDimension.ACCURACY: 1.0}
        pipeline = EvaluationPipeline(evaluators=evaluators, weights=weights)

        dimension_scores = {
            EvaluationDimension.ACCURACY: DimensionScore(
                dimension=EvaluationDimension.ACCURACY, score=0.8
            ),
        }
        overall = pipeline.compute_overall_score(dimension_scores)
        assert overall == 0.8

    def test_create_fast_failure_result(self, pipeline):
        """Test creating fast failure result."""
        result = pipeline.create_fast_failure_result("Test failure reason")

        assert result.is_fast_failure is True
        assert result.failure_reason == "Test failure reason"
        assert result.overall_score == 0.1
        assert len(result.dimension_scores) == len(list(EvaluationDimension))

        for dim_score in result.dimension_scores.values():
            assert dim_score.score == 0.0
            assert "fast failure" in dim_score.reasoning.lower()
            assert dim_score.metadata.get("fast_failure") is True

    def test_evaluator_mapping(self, pipeline):
        """Test that evaluators are properly mapped by dimension."""
        assert EvaluationDimension.ACCURACY in pipeline.evaluators
        assert EvaluationDimension.HELPFULNESS in pipeline.evaluators
        assert len(pipeline.evaluators) == 2

    def test_default_weights_used_when_none_provided(self):
        """Test that default weights are used when no weights provided."""
        from router.judge import BaseEvaluator, ACCURACY_PROMPT, HELPFULNESS_PROMPT

        evaluators = [
            BaseEvaluator(ACCURACY_PROMPT, EvaluationDimension.ACCURACY),
            BaseEvaluator(HELPFULNESS_PROMPT, EvaluationDimension.HELPFULNESS),
        ]
        pipeline = EvaluationPipeline(evaluators=evaluators, weights=None)

        for dim in DEFAULT_DIMENSION_WEIGHTS:
            assert dim in pipeline.weights
            assert pipeline.weights[dim] == DEFAULT_DIMENSION_WEIGHTS[dim]


class TestBaseEvaluator:
    """Tests for BaseEvaluator class."""

    def test_build_prompt(self):
        """Test building evaluation prompt."""
        from router.judge import BaseEvaluator

        template = "Evaluate this: {prompt} and response: {response}"
        evaluator = BaseEvaluator(template, EvaluationDimension.ACCURACY)

        prompt = "What is 2+2?"
        response = "The answer is 4."
        result = evaluator.build_prompt(prompt, response)

        assert "What is 2+2?" in result
        assert "The answer is 4." in result
        assert evaluator.dimension == EvaluationDimension.ACCURACY

    def test_all_dimension_prompts_exist(self):
        """Test that all evaluation dimensions have corresponding prompts."""
        assert EvaluationDimension.ACCURACY in DIMENSION_PROMPTS
        assert EvaluationDimension.HELPFULNESS in DIMENSION_PROMPTS
        assert EvaluationDimension.CLARITY in DIMENSION_PROMPTS
        assert EvaluationDimension.CONCISENESS in DIMENSION_PROMPTS
        assert EvaluationDimension.INSTRUCTION_FOLLOWING in DIMENSION_PROMPTS

    def test_prompt_templates_have_placeholders(self):
        """Test that all dimension prompts have required placeholders."""
        for dim, prompt in DIMENSION_PROMPTS.items():
            assert "{prompt}" in prompt, f"Missing {{prompt}} in {dim.value} prompt"
            assert "{response}" in prompt, f"Missing {{response}} in {dim.value} prompt"
            assert "score" in prompt.lower(), f"Missing 'score' in {dim.value} prompt"

    def test_accuracy_prompt_content(self):
        """Test accuracy prompt contains expected content."""
        assert "factual" in ACCURACY_PROMPT.lower()
        assert "accuracy" in ACCURACY_PROMPT.lower()
        assert "correct" in ACCURACY_PROMPT.lower()

    def test_clarity_prompt_content(self):
        """Test clarity prompt contains expected content."""
        assert "clear" in CLARITY_PROMPT.lower()
        assert "understand" in CLARITY_PROMPT.lower()
        assert "structure" in CLARITY_PROMPT.lower()

    def test_conciseness_prompt_content(self):
        """Test conciseness prompt contains expected content."""
        assert "concise" in CONCISENESS_PROMPT.lower()
        assert "verbose" in CONCISENESS_PROMPT.lower()
        assert "filler" in CONCISENESS_PROMPT.lower()


class TestJudgeClientNewAPI:
    """Tests for new multi-stage evaluation APIs in JudgeClient."""

    @pytest.fixture
    def judge_enabled(self):
        """Create a JudgeClient with judge enabled."""
        with patch("router.judge.settings") as mock_settings:
            mock_settings.judge_enabled = True
            mock_settings.judge_model = "gpt-4o"
            mock_settings.judge_base_url = "https://api.openai.com/v1"
            mock_settings.judge_api_key = "test-key"
            mock_settings.judge_max_retries = 3
            mock_settings.judge_retry_base_delay = 1.0
            return JudgeClient()

    @pytest.fixture
    def judge_disabled(self):
        """Create a JudgeClient with judge disabled."""
        with patch("router.judge.settings") as mock_settings:
            mock_settings.judge_enabled = False
            mock_settings.judge_model = "gpt-4o"
            mock_settings.judge_base_url = "https://api.openai.com/v1"
            mock_settings.judge_api_key = None
            return JudgeClient()

    @pytest.mark.asyncio
    async def test_evaluate_response_fast_failure(self, judge_enabled):
        """Test that evaluate_response detects fast failures."""
        result = await judge_enabled.evaluate_response(
            "Test prompt",
            "I'm sorry, I cannot help with that."
        )

        assert result.is_fast_failure is True
        assert result.overall_score == 0.1
        assert result.failure_reason != ""
        assert len(result.dimension_scores) == 5

    @pytest.mark.asyncio
    async def test_evaluate_response_disabled(self, judge_disabled):
        """Test evaluate_response when judge is disabled."""
        result = await judge_disabled.evaluate_response(
            "Test prompt",
            "This is a valid response that is long enough."
        )

        assert result.is_fast_failure is False
        assert result.overall_score == 0.5
        assert result.metadata.get("judge_disabled") is True
        for dim_score in result.dimension_scores.values():
            assert dim_score.score == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_response_disabled_fast_failure(self, judge_disabled):
        """Test evaluate_response with disabled judge and fast failure."""
        result = await judge_disabled.evaluate_response(
            "Test prompt",
            "I'm sorry, I cannot help."
        )

        assert result.is_fast_failure is True
        assert result.overall_score == 0.1

    @pytest.mark.asyncio
    async def test_evaluate_response_with_mock(self, judge_enabled):
        """Test evaluate_response with mocked API calls."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"score": 0.9, "reasoning": "Good"})}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await judge_enabled.evaluate_response(
                "What is 2+2?",
                "The answer is 4, which is correct and helpful."
            )

            assert result.is_fast_failure is False
            assert 0.0 <= result.overall_score <= 1.0
            assert len(result.dimension_scores) == 5
            for dim in EvaluationDimension:
                assert dim in result.dimension_scores

    @pytest.mark.asyncio
    async def test_evaluate_responses_batch_empty(self, judge_enabled):
        """Test batch evaluation with empty input."""
        results = await judge_enabled.evaluate_responses_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_evaluate_responses_batch_disabled(self, judge_disabled):
        """Test batch evaluation when judge is disabled."""
        pairs = [
            ("Prompt 1", "This is a valid response with enough content."),
            ("Prompt 2", "I'm sorry, I cannot help."),
        ]

        results = await judge_disabled.evaluate_responses_batch(pairs)

        assert len(results) == 2

        assert results[0].is_fast_failure is False
        assert results[0].overall_score == 0.5

        assert results[1].is_fast_failure is True
        assert results[1].overall_score == 0.1

    @pytest.mark.asyncio
    async def test_evaluate_responses_batch_with_mock(self, judge_enabled):
        """Test batch evaluation with mocked API calls."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"score": 0.75, "reasoning": "Okay"})}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            pairs = [
                ("What is 2+2?", "The answer is 4, which is correct."),
                ("What is 3+5?", "The answer is 8, correct and clear."),
            ]

            results = await judge_enabled.evaluate_responses_batch(pairs, max_concurrent=1)

            assert len(results) == 2
            for result in results:
                assert result.is_fast_failure is False
                assert 0.0 <= result.overall_score <= 1.0

    def test_evaluation_pipeline_initialized(self, judge_enabled):
        """Test that JudgeClient initializes evaluation pipeline."""
        assert hasattr(judge_enabled, "evaluation_pipeline")
        assert judge_enabled.evaluation_pipeline is not None
        assert len(judge_enabled.evaluation_pipeline.evaluators) == 5

    def test_default_dimension_weights(self):
        """Test default dimension weights sum to 1.0 and have expected values."""
        total = sum(DEFAULT_DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.0001

        assert DEFAULT_DIMENSION_WEIGHTS[EvaluationDimension.ACCURACY] == 0.30
        assert DEFAULT_DIMENSION_WEIGHTS[EvaluationDimension.HELPFULNESS] == 0.30
        assert DEFAULT_DIMENSION_WEIGHTS[EvaluationDimension.CLARITY] == 0.15
        assert DEFAULT_DIMENSION_WEIGHTS[EvaluationDimension.CONCISENESS] == 0.10
        assert DEFAULT_DIMENSION_WEIGHTS[EvaluationDimension.INSTRUCTION_FOLLOWING] == 0.15


class TestJudgeIntegrationWithPipeline:
    """Integration tests for JudgeClient with multi-stage pipeline."""

    @pytest.fixture
    def judge_enabled(self):
        """Create a JudgeClient with judge enabled."""
        with patch("router.judge.settings") as mock_settings:
            mock_settings.judge_enabled = True
            mock_settings.judge_model = "gpt-4o"
            mock_settings.judge_base_url = "https://api.openai.com/v1"
            mock_settings.judge_api_key = "test-key"
            mock_settings.judge_max_retries = 3
            mock_settings.judge_retry_base_delay = 1.0
            return JudgeClient()

    @pytest.mark.asyncio
    async def test_fast_failure_short_circuits_evaluation(self, judge_enabled):
        """Test that fast failure prevents API calls."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock()
            mock_client.return_value.__aenter__.return_value.post = mock_post

            result = await judge_enabled.evaluate_response(
                "Test prompt",
                "I'm sorry, cannot help."
            )

            assert result.is_fast_failure is True
            mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_response_triggers_evaluation(self, judge_enabled):
        """Test that valid response triggers multi-dimensional evaluation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"score": 0.85, "reasoning": "Good"})}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            valid_response = (
                "This is a sufficiently long response that should not trigger fast failure. "
                "It contains multiple sentences and useful information."
            )

            result = await judge_enabled.evaluate_response(
                "What is the meaning of life?",
                valid_response
            )

            assert result.is_fast_failure is False
            assert len(result.dimension_scores) == 5
            assert all(isinstance(s, DimensionScore) for s in result.dimension_scores.values())

    @pytest.mark.asyncio
    async def test_overall_score_is_weighted_sum(self, judge_enabled):
        """Test that overall score uses same value for all dimensions when mock returns fixed score."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({"score": 0.8, "reasoning": "Good"})}}
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = MagicMock()
            mock_client_instance.post = AsyncMock(return_value=mock_response)
            mock_client_instance.is_closed = False
            mock_client.return_value = mock_client_instance
            mock_client_instance.__aenter__.return_value = mock_client_instance

            valid_response = (
                "This is a valid response that is long enough to pass the fast failure check. "
                "It has multiple sentences and provides useful information."
            )

            result = await judge_enabled.evaluate_response("Test prompt", valid_response)

            assert abs(result.overall_score - 0.8) < 0.0001
            for dim_score in result.dimension_scores.values():
                assert dim_score.score == 0.8


class TestJsonExtractionEdgeCases:
    """Additional edge case tests for JSON extraction."""

    def test_extract_json_with_extra_whitespace(self):
        """Test extraction with extra whitespace around JSON."""
        content = "   \n\n  {\"score\": 0.75}  \n\n   "
        result = _extract_json_from_content(content)
        assert result == '{"score": 0.75}'

    def test_extract_json_multiple_objects(self):
        """Test extraction when multiple JSON objects exist."""
        content = 'First: {"score": 0.5} Second: {"score": 1.0}'
        result = _extract_json_from_content(content)
        assert '"score": 0.5' in result

    def test_extract_json_with_braces_in_reasoning(self):
        """Test extraction when reasoning contains braces."""
        content = '{"score": 0.8, "reasoning": "It used {example} which is good"}'
        result = _extract_json_from_content(content)
        parsed = json.loads(result)
        assert parsed["score"] == 0.8
        assert "{example}" in parsed["reasoning"]
