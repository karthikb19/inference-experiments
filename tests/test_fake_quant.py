"""Tests for per-channel weight fake quantization."""

import pytest
import torch

from inference_experiments.fake_quant import (
    INT4_MAX,
    INT8_MAX,
    FakeQuantization,
    fake_quantize_rows,
    int4_fake_quantize_rows,
)


def test_fake_quantization_enum_uses_vllm_method_names() -> None:
    assert FakeQuantization.INT_8_FAKE_QUANT.value == "int8-fake-quant"
    assert FakeQuantization.INT_4_FAKE_QUANT.value == "int4-fake-quant"


def test_int4_fake_quantize_rows_uses_signed_four_bit_levels() -> None:
    weight = torch.tensor([[-2.0, -0.9, 0.0, 2.0]], dtype=torch.bfloat16)

    result = int4_fake_quantize_rows(weight)

    scale = 2.0 / INT4_MAX
    expected = torch.tensor(
        [[-2.0, round(-0.9 / scale) * scale, 0.0, 2.0]], dtype=torch.bfloat16
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == torch.bfloat16


def test_int4_fake_quantize_rows_rounds_more_coarsely_than_int8() -> None:
    weight = torch.tensor([[1.0, 1.1]], dtype=torch.bfloat16)

    int4_result = int4_fake_quantize_rows(weight)
    int8_result = fake_quantize_rows(weight)

    assert not torch.equal(int4_result, int8_result)


def test_fake_quantize_rows_uses_independent_symmetric_scales() -> None:
    weight = torch.tensor(
        [[-2.0, -1.0, 0.0, 2.0], [-0.25, 0.125, 0.25, 0.0]],
        dtype=torch.bfloat16,
    )

    result = fake_quantize_rows(weight)

    first_scale = 2.0 / INT8_MAX
    second_scale = 0.25 / INT8_MAX
    expected = torch.tensor(
        [
            [-2.0, round(-1.0 / first_scale) * first_scale, 0.0, 2.0],
            [-0.25, round(0.125 / second_scale) * second_scale, 0.25, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == torch.bfloat16


def test_fake_quantize_rows_preserves_zero_rows() -> None:
    weight = torch.zeros((2, 3), dtype=torch.bfloat16)

    result = fake_quantize_rows(weight)

    torch.testing.assert_close(result, weight, rtol=0, atol=0)


def test_fake_quantize_rows_uses_supplied_global_row_maximum() -> None:
    weight = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)

    local = fake_quantize_rows(weight)
    global_result = fake_quantize_rows(weight, row_absmax=torch.tensor([4.0]))

    assert not torch.equal(local, global_result)
    expected_scale = 4.0 / INT8_MAX
    expected = torch.tensor(
        [
            [
                round(1.0 / expected_scale) * expected_scale,
                round(2.0 / expected_scale) * expected_scale,
            ]
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(global_result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "weight",
    [torch.tensor([1.0]), torch.empty((0, 2)), torch.ones((1, 1), dtype=torch.int8)],
)
def test_fake_quantize_rows_rejects_invalid_weights(weight: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError), match="weight"):
        fake_quantize_rows(weight)


@pytest.mark.parametrize(
    "row_absmax", [torch.tensor([1.0, 2.0]), torch.tensor([float("nan")])]
)
def test_fake_quantize_rows_rejects_invalid_scales(row_absmax: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="row_absmax"):
        fake_quantize_rows(torch.ones((1, 2)), row_absmax=row_absmax)
