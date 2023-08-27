from typing import Tuple

import pytest
import torch

from normalizing_flows.src.utils import get_batch_shape, vjp_tensor


@pytest.mark.parametrize('event_shape', [(1,), (2,), (3, 5), (5, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize('batch_shape', [(1,), (2,), (3, 5), (5, 3), (1, 2, 3, 4)])
def test_batch_shape(event_shape: Tuple, batch_shape: Tuple):
    torch.manual_seed(0)
    event_shape = torch.Size(event_shape)
    batch_shape = torch.Size(batch_shape)
    x = torch.randn(*batch_shape, *event_shape)
    assert get_batch_shape(x, event_shape) == batch_shape


@pytest.mark.parametrize('event_shape', [(1,), (2,), (3, 5), (1, 3, 5, 7)])
def test_vjp_function_quadratic(event_shape):
    # f(x) = x^2

    torch.manual_seed(0)
    x = torch.randn(size=event_shape)
    v = torch.randn(size=event_shape)

    fval, vjp = torch.autograd.functional.vjp(lambda _in: _in ** 2, x, v)

    assert torch.allclose(vjp, 2 * x * v)


@pytest.mark.parametrize('event_shape', [(1,), (2,), (3, 5), (1, 3, 5, 7)])
def test_vjp_tensor_quadratic(event_shape):
    # f(x) = x^2

    torch.manual_seed(0)
    x = torch.randn(size=event_shape)
    v = torch.randn(size=event_shape)

    # print()
    # print(f'{x = }')
    # print(f'{v = }')

    x_clone = x.clone().requires_grad_(True)
    output = vjp_tensor(v, x_clone ** 2, x_clone)

    assert torch.allclose(output, 2 * x * v)


def test_vjp_tensor_batched_quadratic():
    # f(x) = x^2
    batch_shape = (3,)
    event_shape = (2,)

    torch.manual_seed(0)
    x = torch.randn(size=(*batch_shape, *event_shape))  # data
    v = torch.randn(size=(*batch_shape, *event_shape))  # noise

    fval, vjp = torch.autograd.functional.vjp(lambda _in: _in ** 2, x, v)

    assert torch.allclose(vjp, 2 * x * v)
