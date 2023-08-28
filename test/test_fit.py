import pytest
import torch
from normalizing_flows import Flow
from normalizing_flows.bijections import NICE, RealNVP, MAF, ElementwiseAffine, ElementwiseShift, ElementwiseRQSpline, \
    CouplingRQNSF, MaskedAutoregressiveRQNSF, LowerTriangular, ElementwiseScale, QR, LU


@pytest.mark.parametrize('bijection_class', [
    LowerTriangular,
    ElementwiseScale,
    LU,
    QR,
    ElementwiseAffine,
    ElementwiseShift,
    ElementwiseRQSpline,
    NICE,
    RealNVP,
    MAF,
    CouplingRQNSF,
    MaskedAutoregressiveRQNSF
])
def test_standard_gaussian(bijection_class):
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    x = torch.randn(size=(n_data, n_dim))
    bijection = bijection_class(event_shape=(n_dim,))
    flow = Flow(bijection)
    flow.fit(x, n_epochs=15)
    x_flow = flow.sample(100_000)
    x_mean = torch.mean(x_flow, dim=0)
    x_var = torch.var(x_flow, dim=0)

    assert torch.allclose(x_mean, torch.zeros(size=(n_dim,)), atol=0.1)
    assert torch.allclose(x_var, torch.ones(size=(n_dim,)), atol=0.1)


def test_diagonal_gaussian_elementwise_affine():
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    sigma = torch.tensor([[0.1, 1.0, 10.0]])
    x = torch.randn(size=(n_data, n_dim)) * sigma
    bijection = ElementwiseAffine(event_shape=(n_dim,))
    flow = Flow(bijection)
    flow.fit(x, n_epochs=15)
    x_flow = flow.sample(100_000)
    x_std = torch.std(x_flow, dim=0)
    relative_error = max((x_std - sigma.ravel()).abs() / sigma.ravel())

    assert relative_error < 0.1


@pytest.mark.parametrize('bijection_class',
                         [
                             LowerTriangular,
                             ElementwiseScale,
                             LU,
                             QR,
                             MaskedAutoregressiveRQNSF,
                             ElementwiseRQSpline,
                             ElementwiseAffine,
                             RealNVP,
                             MAF,
                             CouplingRQNSF
                         ])
def test_diagonal_gaussian_1(bijection_class):
    torch.manual_seed(0)

    n_data = 10_000
    n_dim = 3
    sigma = torch.tensor([[0.1, 1.0, 10.0]])
    x = torch.randn(size=(n_data, n_dim)) * sigma
    bijection = bijection_class(event_shape=(n_dim,))
    flow = Flow(bijection)
    if isinstance(bijection, LowerTriangular):
        flow.fit(x, n_epochs=100)
    else:
        flow.fit(x, n_epochs=25)
    x_flow = flow.sample(100_000)
    x_std = torch.std(x_flow, dim=0)
    relative_error = max((x_std - sigma.ravel()).abs() / sigma.ravel())

    assert relative_error < 0.1
