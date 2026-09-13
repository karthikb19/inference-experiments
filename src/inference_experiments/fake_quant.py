"""vLLM plugins for symmetric per-channel INT4 and INT8 fake quantization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

FAKE_QUANTIZATION = "int8-fake-quant"
INT4_FAKE_QUANTIZATION = "int4-fake-quant"
INT8_MAX = 127
INT4_MAX = 7


def fake_quantize_rows(
    weight: torch.Tensor, *, row_absmax: torch.Tensor | None = None
) -> torch.Tensor:
    """Quantize each output row to INT8 and dequantize to the input dtype."""
    return _fake_quantize_rows(weight, row_absmax=row_absmax, quantization_max=INT8_MAX)


def int4_fake_quantize_rows(
    weight: torch.Tensor, *, row_absmax: torch.Tensor | None = None
) -> torch.Tensor:
    """Quantize each output row to signed INT4 and dequantize to the input dtype."""
    return _fake_quantize_rows(weight, row_absmax=row_absmax, quantization_max=INT4_MAX)


def _fake_quantize_rows(
    weight: torch.Tensor,
    *,
    row_absmax: torch.Tensor | None,
    quantization_max: int,
) -> torch.Tensor:
    if weight.ndim != 2 or weight.shape[0] == 0 or weight.shape[1] == 0:
        raise ValueError("weight must be a non-empty two-dimensional tensor")
    if not weight.is_floating_point():
        raise TypeError("weight must use a floating-point dtype")

    float_weight = weight.float()
    if row_absmax is None:
        row_absmax = float_weight.abs().amax(dim=1)
    if row_absmax.shape != (weight.shape[0],):
        raise ValueError("row_absmax must contain one value per output row")
    if not torch.isfinite(row_absmax).all() or (row_absmax < 0).any():
        raise ValueError("row_absmax values must be finite and non-negative")

    scales = row_absmax.float() / quantization_max
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = (
        torch.round(float_weight / safe_scales[:, None])
        .clamp(-quantization_max, quantization_max)
        .to(torch.int8)
    )
    return (quantized.float() * safe_scales[:, None]).to(weight.dtype)


class _FakeQuantLinearMethod(LinearMethodBase):
    """Load BF16 weights, fake-quantize them once, and use BF16 GEMM."""

    quantization_max: int

    def __init__(self, quantization_max: int) -> None:
        self.quantization_max = quantization_max
        self._unquantized = UnquantizedLinearMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: object,
    ) -> None:
        """Create the ordinary BF16 weight expected by checkpoint loaders."""
        if params_dtype != torch.bfloat16:
            raise ValueError("fake quantization requires BF16 model weights")
        self._unquantized.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Replace loaded weights with their per-row quantized round-trip values."""
        weight = getattr(layer, "weight", None)
        if not isinstance(weight, torch.nn.Parameter):
            raise TypeError("fake-quantized linear layer has no weight parameter")

        row_absmax = weight.detach().float().abs().amax(dim=1)
        if isinstance(layer, RowParallelLinear) and layer.tp_size > 1:
            torch.distributed.all_reduce(
                row_absmax,
                op=torch.distributed.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
        with torch.no_grad():
            weight.copy_(
                _fake_quantize_rows(
                    weight,
                    row_absmax=row_absmax,
                    quantization_max=self.quantization_max,
                )
            )
        self._unquantized.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multiply activations by the dequantized BF16 weight."""
        return self._unquantized.apply(layer, x, bias)


class _FakeQuantConfig(QuantizationConfig):
    """Apply weight-only fake quantization to every vLLM linear layer."""

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return FAKE_QUANTIZATION

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, object]) -> _FakeQuantConfig:
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        del prefix
        if isinstance(layer, LinearBase):
            return self.linear_method()
        return None

    @classmethod
    def linear_method(cls) -> LinearMethodBase:
        raise NotImplementedError


@register_quantization_config(FAKE_QUANTIZATION)
class Int8FakeQuantConfig(_FakeQuantConfig):
    """Apply weight-only fake quantization to every vLLM linear layer."""

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return FAKE_QUANTIZATION

    @classmethod
    def linear_method(cls) -> LinearMethodBase:
        return _FakeQuantLinearMethod(INT8_MAX)


@register_quantization_config(INT4_FAKE_QUANTIZATION)
class Int4FakeQuantConfig(_FakeQuantConfig):
    """Apply signed INT4 weight-only fake quantization to every linear layer."""

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return INT4_FAKE_QUANTIZATION

    @classmethod
    def linear_method(cls) -> LinearMethodBase:
        return _FakeQuantLinearMethod(INT4_MAX)


def register_plugin() -> None:
    """Provide the callable required by vLLM's general-plugin loader."""
