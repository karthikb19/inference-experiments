"""vLLM plugin for symmetric per-channel INT8 weight fake quantization."""

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
INT8_MAX = 127


def fake_quantize_rows(
    weight: torch.Tensor, *, row_absmax: torch.Tensor | None = None
) -> torch.Tensor:
    """Quantize each output row to INT8 and dequantize to the input dtype."""
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

    scales = row_absmax.float() / INT8_MAX
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = (
        torch.round(float_weight / safe_scales[:, None])
        .clamp(-INT8_MAX, INT8_MAX)
        .to(torch.int8)
    )
    return (quantized.float() * safe_scales[:, None]).to(weight.dtype)


class Int8FakeQuantLinearMethod(LinearMethodBase):
    """Load BF16 weights, fake-quantize them once, and use BF16 GEMM."""

    def __init__(self) -> None:
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
            raise ValueError("int8-fake-quant requires BF16 model weights")
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
        """Replace loaded weights with their per-row INT8 round-trip values."""
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
            weight.copy_(fake_quantize_rows(weight, row_absmax=row_absmax))
        self._unquantized.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multiply activations by the dequantized BF16 weight."""
        return self._unquantized.apply(layer, x, bias)


@register_quantization_config(FAKE_QUANTIZATION)
class Int8FakeQuantConfig(QuantizationConfig):
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
    def from_config(cls, config: dict[str, object]) -> Int8FakeQuantConfig:
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        del prefix
        if isinstance(layer, LinearBase):
            return Int8FakeQuantLinearMethod()
        return None


def register_plugin() -> None:
    """Provide the callable required by vLLM's general-plugin loader."""
