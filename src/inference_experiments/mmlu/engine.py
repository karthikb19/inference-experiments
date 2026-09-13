"""Inference protocol and lazy vLLM implementation."""

from __future__ import annotations

import platform
from typing import Protocol, cast

from inference_experiments.mmlu.models import (
    CHOICES,
    Choice,
    EngineProvenance,
    InferenceAnswer,
    InferenceError,
    InferenceRequest,
)


class InferenceEngine(Protocol):
    """Minimal engine boundary used by the evaluator."""

    def predict(
        self, requests: tuple[InferenceRequest, ...]
    ) -> tuple[InferenceAnswer, ...]: ...

    def provenance(self) -> EngineProvenance: ...

    def close(self) -> None: ...


class _Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class _Logprob(Protocol):
    logprob: float


class _Completion(Protocol):
    token_ids: list[int]
    logprobs: list[dict[int, _Logprob]] | None


class _RequestOutput(Protocol):
    prompt_token_ids: list[int]
    outputs: list[_Completion]


class _Runner(Protocol):
    def get_tokenizer(self) -> _Tokenizer: ...

    def generate(
        self,
        prompts: list[str],
        sampling_params: object,
        *,
        use_tqdm: bool,
    ) -> list[_RequestOutput]: ...


class VLLMEngine:
    """Two-GPU, in-process vLLM engine for constrained MMLU scoring."""

    def __init__(
        self,
        *,
        model: str,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        quantization: str | None,
        seed: int,
    ) -> None:
        from vllm import LLM, SamplingParams

        runner = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
            seed=seed,
        )
        self._runner: _Runner | None = cast(_Runner, runner)
        self._answer_token_ids = _answer_token_ids(runner.get_tokenizer())
        self._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=4,
            logprob_token_ids=list(self._answer_token_ids),
            allowed_token_ids=list(self._answer_token_ids),
        )

    def predict(
        self, requests: tuple[InferenceRequest, ...]
    ) -> tuple[InferenceAnswer, ...]:
        """Score A-D directly for a batch of prompts."""
        if not requests:
            return ()
        outputs = self._active_runner().generate(
            [request.prompt for request in requests],
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
        """Report versions and visible GPUs separately from quality output."""
        import torch
        import vllm

        gpu_names = tuple(
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        )
        return EngineProvenance(
            python_version=platform.python_version(),
            vllm_version=vllm.__version__,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            gpu_names=gpu_names,
        )

    def close(self) -> None:
        """Release the runner so vLLM can stop its worker processes."""
        self._runner = None

    def _active_runner(self) -> _Runner:
        if self._runner is None:
            raise InferenceError("vLLM engine is closed")
        return self._runner

    def _parse_output(
        self, request: InferenceRequest, output: _RequestOutput
    ) -> InferenceAnswer:
        if len(output.outputs) != 1:
            raise InferenceError(f"{request.row_id}: expected one completion")
        completion = output.outputs[0]
        if len(completion.token_ids) != 1 or not completion.logprobs:
            raise InferenceError(f"{request.row_id}: expected one scored answer token")
        logprobs = completion.logprobs[0]
        try:
            scores = tuple(
                logprobs[token_id].logprob for token_id in self._answer_token_ids
            )
        except KeyError as error:
            raise InferenceError(
                f"{request.row_id}: vLLM omitted an A-D log probability"
            ) from error
        typed_scores = cast(tuple[float, float, float, float], scores)
        try:
            selected_index = self._answer_token_ids.index(completion.token_ids[0])
        except ValueError as error:
            raise InferenceError(
                f"{request.row_id}: selected token is not an A-D answer"
            ) from error
        selected: Choice = CHOICES[selected_index]
        # Preserve vLLM's selected answer when multiple choices share the maximum.
        if typed_scores[selected_index] < max(typed_scores):
            raise InferenceError(
                f"{request.row_id}: selected token and scores disagree"
            )
        return InferenceAnswer(
            row_id=request.row_id,
            choice=selected,
            choice_logprobs=typed_scores,
            prompt_tokens=len(output.prompt_token_ids),
            output_tokens=len(completion.token_ids),
        )


def _answer_token_ids(tokenizer: _Tokenizer) -> tuple[int, int, int, int]:
    token_ids: list[int] = []
    for choice in CHOICES:
        encoded = tokenizer.encode(f" {choice}", add_special_tokens=False)
        if len(encoded) != 1:
            raise InferenceError(f"answer {choice!r} is not one tokenizer token")
        token_ids.append(encoded[0])
    if len(set(token_ids)) != 4:
        raise InferenceError("A-D answer tokens are not distinct")
    return cast(tuple[int, int, int, int], tuple(token_ids))
