import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

from router.config import settings
from router.encryption import get_encryption_manager

logger = logging.getLogger(__name__)


class EvaluationDimension(Enum):
    """评估维度枚举"""
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    CLARITY = "clarity"
    CONCISENESS = "conciseness"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass
class DimensionScore:
    """单个维度的评分结果"""
    dimension: EvaluationDimension
    score: float
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvaluationResult:
    """完整的多维度评估结果"""
    overall_score: float
    dimension_scores: dict[EvaluationDimension, DimensionScore]
    is_fast_failure: bool = False
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "dimension_scores": {
                dim.value: score.to_dict()
                for dim, score in self.dimension_scores.items()
            },
            "is_fast_failure": self.is_fast_failure,
            "failure_reason": self.failure_reason,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


JUDGE_PROMPT_TEMPLATE = """You are an impartial judge evaluating the quality of an AI model's response to a specific prompt.
Your goal is to provide a quality score between 0.0 and 1.0, where 1.0 is a perfect response.

Consider the following criteria:
1. Accuracy: Is the information correct?
2. Helpfulness: Does it directly answer the user's request?
3. Clarity: Is it easy to understand and well-structured?
4. Conciseness: Is it free of unnecessary filler?
5. Instruction Following: Did it follow all specific constraints in the prompt?

Prompt:
{prompt}

Model Response:
{response}

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any other text, markdown formatting, code blocks, or explanations outside the JSON.

Required JSON format (respond with EXACTLY this, no markdown):
{{"score": 0.85, "reasoning": "Your brief explanation here"}}

The score must be a number between 0.0 and 1.0.
"""


ACCURACY_PROMPT = """Evaluate the FACTUAL ACCURACY of an AI model's response.

Your task: Determine if the response contains correct and accurate information.
Score 0.0 = Completely incorrect or contains dangerous misinformation
Score 0.3 = Mostly incorrect with some minor correct elements
Score 0.5 = Mixed accuracy - some correct, some incorrect
Score 0.7 = Mostly accurate with minor errors
Score 1.0 = Completely accurate with no errors

Prompt:
{prompt}

Model Response:
{response}

Respond with ONLY a JSON object:
{{"score": 0.0, "reasoning": "Brief explanation of accuracy assessment"}}
"""


HELPFULNESS_PROMPT = """Evaluate the HELPFULNESS of an AI model's response.

Your task: Determine how well the response directly addresses the user's request.
Score 0.0 = Does not address the prompt at all
Score 0.3 = Barely related, mostly unhelpful
Score 0.5 = Partially helpful but incomplete or misses key points
Score 0.7 = Mostly helpful with minor gaps
Score 1.0 = Fully addresses all aspects of the user's request

Prompt:
{prompt}

Model Response:
{response}

Respond with ONLY a JSON object:
{{"score": 0.0, "reasoning": "Brief explanation of helpfulness assessment"}}
"""


CLARITY_PROMPT = """Evaluate the CLARITY of an AI model's response.

Your task: Assess how easy it is to understand the response.
Consider: logical flow, structure, grammar, readability, organization.
Score 0.0 = Completely incoherent or unreadable
Score 0.3 = Very confusing, hard to follow
Score 0.5 = Somewhat clear but poorly structured
Score 0.7 = Clear and well-structured with minor issues
Score 1.0 = Crystal clear, excellently structured, easy to follow

Prompt:
{prompt}

Model Response:
{response}

Respond with ONLY a JSON object:
{{"score": 0.0, "reasoning": "Brief explanation of clarity assessment"}}
"""


CONCISENESS_PROMPT = """Evaluate the CONCISENESS of an AI model's response.

Your task: Determine if the response is free of unnecessary filler.
Score 0.0 = Extremely verbose, full of redundant content
Score 0.3 = Quite verbose with unnecessary repetition
Score 0.5 = Somewhat verbose but contains necessary info
Score 0.7 = Mostly concise with minor filler
Score 1.0 = Perfectly concise - no unnecessary words

Prompt:
{prompt}

Model Response:
{response}

Respond with ONLY a JSON object:
{{"score": 0.0, "reasoning": "Brief explanation of conciseness assessment"}}
"""


INSTRUCTION_FOLLOWING_PROMPT = """Evaluate how well the AI model FOLLOWED INSTRUCTIONS.

Your task: Check if the response followed all constraints and requirements in the prompt.
Look for: format requirements, output constraints, tone requirements, specific instructions.
Score 0.0 = Ignored all instructions
Score 0.3 = Followed very few instructions
Score 0.5 = Followed some instructions but missed key ones
Score 0.7 = Followed most instructions with minor deviations
Score 1.0 = Followed all instructions perfectly

Prompt:
{prompt}

Model Response:
{response}

Respond with ONLY a JSON object:
{{"score": 0.0, "reasoning": "Brief explanation of instruction following assessment"}}
"""


DEFAULT_DIMENSION_WEIGHTS: dict[EvaluationDimension, float] = {
    EvaluationDimension.ACCURACY: 0.30,
    EvaluationDimension.HELPFULNESS: 0.30,
    EvaluationDimension.CLARITY: 0.15,
    EvaluationDimension.CONCISENESS: 0.10,
    EvaluationDimension.INSTRUCTION_FOLLOWING: 0.15,
}


def _extract_json_from_content(content: str) -> str:
    """Extract JSON from content, handling markdown code blocks and extra text."""
    content = content.strip()

    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1 :]

        closing_fence = content.rfind("```")
        if closing_fence != -1:
            content = content[:closing_fence]

        content = content.strip()

    start_idx = content.find('{"score"')
    if start_idx != -1:
        content = content[start_idx:]
        brace_count = 0
        end_idx = 0
        for i, char in enumerate(content):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        content = content[:end_idx]

    return content.strip()


class BaseEvaluator(ABC):
    """评估器基类"""

    def __init__(self, prompt_template: str, dimension: EvaluationDimension):
        self.prompt_template = prompt_template
        self.dimension = dimension

    def build_prompt(self, prompt: str, response: str) -> str:
        return self.prompt_template.format(prompt=prompt, response=response)


class FastFailureDetector:
    """快速失败检测器 - 基于规则的快速过滤"""

    FAILURE_PATTERNS = [
        re.compile(r"(?i)^\s*($|i'?m\s+sorry)"),
        re.compile(r"(?i)i\s+(cannot|can't|am\s+unable\s+to)\s+(help|assist)"),
        re.compile(r"(?i)as\s+an\s+ai\s+language\s+model"),
        re.compile(r"(?i)i\s+don'?t\s+have\s+(access|information)"),
        re.compile(r"(?i)unable\s+to\s+(process|generate|complete)"),
        re.compile(r"(?i)context\s+window\s+(exceeded|too\s+long)"),
        re.compile(r"(?i)rate\s+limit"),
    ]

    def check(self, response: str) -> tuple[bool, str]:
        """检查是否为明显失败的响应

        Returns:
            (is_failure, reason) - 是否失败及原因
        """
        if not response or len(response.strip()) < 20:
            return True, "Response is empty or too short"

        for pattern in self.FAILURE_PATTERNS:
            if pattern.search(response):
                return True, f"Matched failure pattern: {pattern.pattern}"

        return False, ""


class EvaluationPipeline:
    """多阶段评估管线"""

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
        weights: dict[EvaluationDimension, float] | None = None,
        use_fast_failure: bool = True,
    ):
        self.evaluators = {e.dimension: e for e in evaluators}
        self.weights = weights or DEFAULT_DIMENSION_WEIGHTS.copy()
        self.use_fast_failure = use_fast_failure
        self.fast_failure_detector = FastFailureDetector()
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """确保权重和为1.0"""
        total = sum(self.weights.values())
        if total != 1.0 and total > 0:
            self.weights = {
                dim: w / total for dim, w in self.weights.items()
            }

    def compute_overall_score(
        self, dimension_scores: dict[EvaluationDimension, DimensionScore]
    ) -> float:
        """根据各维度评分计算综合分数"""
        total = 0.0
        total_weight = 0.0

        for dim, score in dimension_scores.items():
            weight = self.weights.get(dim, 0.0)
            total += score.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        return total / total_weight

    def create_fast_failure_result(self, reason: str) -> EvaluationResult:
        """创建快速失败的评估结果"""
        dimension_scores = {}
        for dim in EvaluationDimension:
            dimension_scores[dim] = DimensionScore(
                dimension=dim,
                score=0.0,
                reasoning="Skipped due to fast failure detection",
                metadata={"fast_failure": True},
            )

        return EvaluationResult(
            overall_score=0.1,
            dimension_scores=dimension_scores,
            is_fast_failure=True,
            failure_reason=reason,
            metadata={"evaluation_pipeline": "fast_failure"},
        )


class JudgeClient:
    """JudgeClient 客户端 - 支持多阶段评估"""

    def __init__(self):
        self.enabled = settings.judge_enabled
        self.model = settings.judge_model
        self.base_url = settings.judge_base_url
        self.api_key = get_encryption_manager().maybe_decrypt(settings.judge_api_key)
        self.http_referer = settings.judge_http_referer
        self.x_title = settings.judge_x_title
        self.max_retries = settings.judge_max_retries
        self.base_delay = settings.judge_retry_base_delay

        self._client: httpx.AsyncClient | None = None
        self._request_times: deque[float] = deque(maxlen=100)
        self._rate_limit_per_minute = 50
        self._max_concurrent = 3

        self._init_evaluation_pipeline()

    def _init_evaluation_pipeline(self) -> None:
        """初始化评估管线"""
        evaluators = [
            BaseEvaluator(ACCURACY_PROMPT, EvaluationDimension.ACCURACY),
            BaseEvaluator(HELPFULNESS_PROMPT, EvaluationDimension.HELPFULNESS),
            BaseEvaluator(CLARITY_PROMPT, EvaluationDimension.CLARITY),
            BaseEvaluator(CONCISENESS_PROMPT, EvaluationDimension.CONCISENESS),
            BaseEvaluator(INSTRUCTION_FOLLOWING_PROMPT, EvaluationDimension.INSTRUCTION_FOLLOWING),
        ]

        self.evaluation_pipeline = EvaluationPipeline(
            evaluators=evaluators,
            weights=DEFAULT_DIMENSION_WEIGHTS.copy(),
            use_fast_failure=True,
        )

    def _is_obvious_failure(self, response: str) -> bool:
        """检查是否为明显失败的响应（保持向后兼容）"""
        is_failure, _ = self.evaluation_pipeline.fast_failure_detector.check(response)
        return is_failure

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建共享 HTTP 客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                verify=settings.verify_tls,
            )
        return self._client

    async def _respect_rate_limit(self) -> None:
        """速率限制控制"""
        now = time.time()
        one_minute_ago = now - 60

        while self._request_times and self._request_times[0] < one_minute_ago:
            self._request_times.popleft()

        if len(self._request_times) >= self._rate_limit_per_minute * 0.8:
            delay = 60 - (now - self._request_times[0])
            if delay > 0:
                logger.debug(f"Rate limit approaching, pausing {delay:.1f}s")
                await asyncio.sleep(min(delay, 10))

    async def _evaluate_single_dimension(
        self, prompt: str, response: str, evaluator: BaseEvaluator
    ) -> DimensionScore:
        """评估单个维度"""
        judge_prompt = evaluator.build_prompt(prompt, response)
        client = await self._get_client()

        for attempt in range(self.max_retries):
            try:
                score_value, reasoning = await self._try_score(
                    client, judge_prompt, prompt, response
                )
                self._request_times.append(time.time())
                return DimensionScore(
                    dimension=evaluator.dimension,
                    score=max(0.0, min(1.0, score_value)),
                    reasoning=reasoning,
                    metadata={"attempts": attempt + 1},
                )
            except (httpx.HTTPStatusError, httpx.NetworkError, httpx.TimeoutException) as e:
                if attempt == self.max_retries - 1:
                    break
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 400:
                    break
                delay = self.base_delay * (2**attempt)
                await asyncio.sleep(delay)
            except Exception:
                break

        return DimensionScore(
            dimension=evaluator.dimension,
            score=0.5,
            reasoning="Evaluation failed, using fallback score",
            metadata={"fallback": True},
        )

    async def _try_score(
        self, client: httpx.AsyncClient | None, judge_prompt: str, prompt: str, response: str
    ) -> tuple[float, str]:
        """执行单次评分尝试"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that evaluates AI responses.",
                },
                {"role": "user", "content": judge_prompt},
            ],
        }

        async def _make_request(c: httpx.AsyncClient) -> tuple[float, str]:
            res = await c.post(
                f"{self.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers
            )
            res.raise_for_status()
            data = res.json()

            if not data.get("choices") or len(data["choices"]) == 0:
                raise ValueError("Empty choices in judge response")

            message = data["choices"][0].get("message", {})
            content = message.get("content", "")

            if not content or not content.strip():
                raise ValueError("Empty content in judge response")

            json_content = _extract_json_from_content(content)
            try:
                result = json.loads(json_content)
            except json.JSONDecodeError as json_err:
                logger.warning(f"Judge returned invalid JSON: {content[:200]}... Error: {json_err}")
                raise ValueError(f"Invalid JSON from judge: {json_err}") from None

            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")

            logger.debug(f"Judge score: {score} (Reasoning: {reasoning})")
            return score, reasoning

        if client is None:
            async with httpx.AsyncClient(timeout=30.0, verify=settings.verify_tls) as c:
                return await _make_request(c)
        else:
            return await _make_request(client)

    async def evaluate_response(
        self, prompt: str, response: str
    ) -> EvaluationResult:
        """执行多阶段评估，返回完整的评估结果"""
        if self.evaluation_pipeline.use_fast_failure:
            is_failure, reason = self.evaluation_pipeline.fast_failure_detector.check(response)
            if is_failure:
                logger.debug(f"Fast failure detected: {reason}")
                return self.evaluation_pipeline.create_fast_failure_result(reason)

        if not self.enabled:
            dimension_scores = {}
            for dim in EvaluationDimension:
                dimension_scores[dim] = DimensionScore(
                    dimension=dim,
                    score=0.5,
                    reasoning="Judge disabled, using neutral score",
                    metadata={"disabled": True},
                )
            return EvaluationResult(
                overall_score=0.5,
                dimension_scores=dimension_scores,
                metadata={"judge_disabled": True},
            )

        tasks = []
        for evaluator in self.evaluation_pipeline.evaluators.values():
            task = self._evaluate_single_dimension(prompt, response, evaluator)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        dimension_scores: dict[EvaluationDimension, DimensionScore] = {}
        for result in results:
            dimension_scores[result.dimension] = result

        overall_score = self.evaluation_pipeline.compute_overall_score(dimension_scores)

        return EvaluationResult(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            metadata={
                "num_evaluators": len(dimension_scores),
                "evaluation_model": self.model,
            },
        )

    async def evaluate_responses_batch(
        self, prompt_response_pairs: list[tuple[str, str]], max_concurrent: int = 3
    ) -> list[EvaluationResult]:
        """批量执行多阶段评估"""
        if not prompt_response_pairs:
            return []

        if not self.enabled:
            results = []
            for _, response in prompt_response_pairs:
                if self._is_obvious_failure(response):
                    result = self.evaluation_pipeline.create_fast_failure_result(
                        "Obvious failure pattern detected"
                    )
                else:
                    dimension_scores = {}
                    for dim in EvaluationDimension:
                        dimension_scores[dim] = DimensionScore(
                            dimension=dim,
                            score=0.5,
                            reasoning="Judge disabled, using neutral score",
                        )
                    result = EvaluationResult(
                        overall_score=0.5,
                        dimension_scores=dimension_scores,
                        metadata={"judge_disabled": True},
                    )
                results.append(result)
            return results

        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_single(prompt: str, response: str) -> EvaluationResult:
            async with semaphore:
                await self._respect_rate_limit()
                return await self.evaluate_response(prompt, response)

        tasks = [evaluate_single(p, r) for p, r in prompt_response_pairs]
        return await asyncio.gather(*tasks)

    async def score_response(self, prompt: str, response: str) -> float:
        """评分单个响应（保持向后兼容）"""
        if not self.enabled:
            if response and len(response.strip()) > 0:
                return 0.5
            return 0.0

        if not response or len(response.strip()) < 5:
            return 0.0

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response=response)

        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                score, _ = await self._try_score(
                    client=None, judge_prompt=judge_prompt, prompt=prompt, response=response
                )
                self._request_times.append(time.time())
                return max(0.0, min(1.0, score))
            except (httpx.HTTPStatusError, httpx.NetworkError, httpx.TimeoutException) as e:
                last_exception = e

                if isinstance(e, httpx.HTTPStatusError):
                    status_code = e.response.status_code
                    if status_code == 400:
                        break
                    if status_code == 429:
                        logger.warning(
                            f"Judge rate limited (429) on attempt {attempt + 1}/{self.max_retries}"
                        )
                    elif status_code >= 500:
                        logger.warning(
                            f"Judge server error {status_code} on attempt {attempt + 1}/{self.max_retries}"
                        )
                    else:
                        break
                else:
                    logger.warning(
                        f"Judge network/timeout error on attempt {attempt + 1}/{self.max_retries}: {e}"
                    )

                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2**attempt)
                    logger.info(f"Retrying judge request in {delay:.1f}s...")
                    await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Unexpected judge error: {e}")
                last_exception = e
                break

        if last_exception:
            error_msg = (
                str(last_exception) if str(last_exception) else type(last_exception).__name__
            )
            logger.warning(
                f"Judge scoring failed after {self.max_retries} attempts: {error_msg}. Falling back."
            )

        if response and len(response.strip()) > 0:
            return 0.5
        return 0.0

    async def score_responses_batch(
        self, prompt_response_pairs: list[tuple[str, str]], max_concurrent: int = 3
    ) -> list[float]:
        """批量评分响应（保持向后兼容）"""
        if not prompt_response_pairs:
            return []

        if not self.enabled:
            return [0.1 if self._is_obvious_failure(r) else 0.5 for _, r in prompt_response_pairs]

        filtered_pairs = []
        pre_assigned_scores: dict[int, float] = {}

        for i, (prompt, response) in enumerate(prompt_response_pairs):
            if self._is_obvious_failure(response):
                pre_assigned_scores[i] = 0.1
                logger.debug(f"Skipped judge API call for obvious failure at index {i}")
            else:
                filtered_pairs.append((i, prompt, response))

        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_single(index: int, prompt: str, response: str) -> tuple[int, float]:
            async with semaphore:
                await self._respect_rate_limit()

                try:
                    score = await self.score_response(prompt, response)
                    self._request_times.append(time.time())
                    return index, score
                except Exception as e:
                    logger.debug(f"Judge failed for index {index}: {e}")
                    return index, 0.5

        tasks = [score_single(idx, prompt, response) for idx, prompt, response in filtered_pairs]
        scored_results = await asyncio.gather(*tasks)

        final_scores: list[float] = [0.0] * len(prompt_response_pairs)

        for idx, score in pre_assigned_scores.items():
            final_scores[idx] = score

        for idx, score in scored_results:
            final_scores[idx] = score

        logger.debug(
            f"Batch scored {len(prompt_response_pairs)} responses, "
            f"saved {len(pre_assigned_scores)} API calls via pre-filter"
        )

        return final_scores

    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
