"""Likelihood-engine protocol and lazy in-process vLLM implementation."""

from __future__ import annotations

import math
import platform
from collections.abc import Sequence
from typing import Protocol, TypedDict, cast

from inference_experiments.wikitext.models import (
    EngineProvenance,
    InferenceError,
    PromptLikelihood,
    PromptTokenLogprob,
    ScoringWindow,
)


class LikelihoodEngine(Protocol):
    """Minimal inference boundary used by the perplexity evaluator."""

    def score(
        self, requests: tuple[ScoringWindow, ...]
    ) -> tuple[PromptLikelihood, ...]: ...

    def provenance(self) -> EngineProvenance: ...

    def close(self) -> None: ...


class _TokensPrompt(TypedDict):
    prompt_token_ids: list[int]


class _Logprob(Protocol):
    logprob: float


class _Completion(Protocol):
    token_ids: list[int]


class _RequestOutput(Protocol):
    prompt_token_ids: list[int] | None
    prompt_logprobs: Sequence[dict[int, _Logprob] | None] | None
    outputs: list[_Completion]


class _Runner(Protocol):
    def generate(
        self,
        prompts: list[_TokensPrompt],
        sampling_params: object,
        *,
        use_tqdm: bool,
    ) -> list[_RequestOutput]: ...


class VLLMLikelihoodEngine:
    """Two-GPU vLLM prompt-logprob scorer for pre-tokenized windows."""

    def __init__(
        self,
        *,
        model: str,
        tokenizer: str,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        quantization: str | None,
        seed: int,
    ) -> None:
        from vllm import LLM, SamplingParams

        runner = LLM(
            model=model,
            tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            seed=seed,
        )
        self._runner: _Runner | None = cast(_Runner, runner)
        self._sampling_params = SamplingParams(
            prompt_logprobs=0,
            max_tokens=1,
            temperature=0.0,
        )

    def score(
        self, requests: tuple[ScoringWindow, ...]
    ) -> tuple[PromptLikelihood, ...]:
        """Return the observed-token log-probability at every prompt position."""
        if not requests:
            return ()
        prompts = [
            _TokensPrompt(prompt_token_ids=list(request.prompt_token_ids))
            for request in requests
        ]
        outputs = self._active_runner().generate(
            prompts,
            self._sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != len(requests):
            raise InferenceError("vLLM returned a different number of responses")
        return tuple(
            self._parse_output(request, output)
            for request, output in zip(requests, outputs, strict=True)
        )

    def provenance(self) -> EngineProvenance:
        """Report volatile engine and accelerator versions."""
        import torch
        import vllm

        return EngineProvenance(
            python_version=platform.python_version(),
            vllm_version=vllm.__version__,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            gpu_names=tuple(
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ),
        )

    def close(self) -> None:
        """Release the vLLM runner and its workers."""
        self._runner = None

    def _active_runner(self) -> _Runner:
        if self._runner is None:
            raise InferenceError("vLLM likelihood engine is closed")
        return self._runner

    def _parse_output(
        self, request: ScoringWindow, output: _RequestOutput
    ) -> PromptLikelihood:
        expected_ids = list(request.prompt_token_ids)
        if output.prompt_token_ids != expected_ids:
            raise InferenceError(f"{request.window_id}: prompt token IDs changed")
        if len(output.outputs) != 1 or len(output.outputs[0].token_ids) != 1:
            raise InferenceError(
                f"{request.window_id}: expected one ignored generated token"
            )
        raw_scores = output.prompt_logprobs
        if raw_scores is None or len(raw_scores) != len(expected_ids):
            raise InferenceError(
                f"{request.window_id}: prompt log-probability length mismatch"
            )
        if raw_scores[0] is not None:
            raise InferenceError(
                f"{request.window_id}: first prompt log-probability must be None"
            )

        scores: list[PromptTokenLogprob | None] = [None]
        for offset, (token_id, values) in enumerate(
            zip(expected_ids[1:], raw_scores[1:], strict=True), start=1
        ):
            if values is None or token_id not in values:
                raise InferenceError(
                    f"{request.window_id}: missing observed token score at {offset}"
                )
            logprob = values[token_id].logprob
            if not math.isfinite(logprob) or logprob > 0:
                raise InferenceError(
                    f"{request.window_id}: invalid log-probability at {offset}"
                )
            scores.append(PromptTokenLogprob(token_id=token_id, logprob=logprob))
        return PromptLikelihood(
            window_id=request.window_id,
            prompt_token_ids=request.prompt_token_ids,
            prompt_logprobs=tuple(scores),
        )
