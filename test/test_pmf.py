import pytest
import torch

from normalizing_flows import RealNVP, PrincipalManifoldFlow


def test_constructor():
    torch.manual_seed(0)
    PrincipalManifoldFlow(
        RealNVP(event_shape=(2,)),
        record_Ihat_P=True,
        record_log_px=True,
        debug=False,
        method="vectorized"
    )


def test_log_prob():
    torch.manual_seed(0)
    x = torch.randn(size=(10, 2))
    flow = PrincipalManifoldFlow(
        RealNVP(event_shape=(2,)),
        record_Ihat_P=True,
        record_log_px=True,
        debug=False,
        method="vectorized"
    )
    log_prob = flow.log_prob(x)
    assert log_prob.shape == (10,)


def test_sample():
    torch.manual_seed(0)
    flow = PrincipalManifoldFlow(
        RealNVP(event_shape=(2,)),
        record_Ihat_P=True,
        record_log_px=True,
        debug=False,
        method="vectorized"
    )
    x = flow.sample(10)
    assert x.shape == (10, 2)


def test_autograd():
    torch.manual_seed(0)
    x = torch.randn(size=(10, 2))
    flow = PrincipalManifoldFlow(
        RealNVP(event_shape=(2,)),
        record_Ihat_P=True,
        record_log_px=True,
        debug=False,
        method="vectorized"
    )
    loss = -flow.log_prob(x).mean()
    loss.backward()


@pytest.mark.parametrize('objective', ['unbiased', 'brute_force'])
@pytest.mark.parametrize('method', ['default', 'einsum', 'vjp', 'vectorized'])
def test_fit(objective: str, method: str):
    torch.manual_seed(0)
    x = torch.randn(size=(10, 2))
    flow = PrincipalManifoldFlow(
        RealNVP(event_shape=(2,)),
        record_Ihat_P=False,
        record_log_px=False,
        debug=False,
        method=method,
        objective=objective
    )
    flow.fit(x, n_epochs=2, show_progress=False)


@pytest.mark.parametrize('objective', ['unbiased', 'brute_force'])
@pytest.mark.parametrize('method', ['default', 'einsum', 'vjp', 'vectorized'])
def test_broad(objective: str, method: str):
    torch.manual_seed(0)
    x = torch.randn(size=(10, 2))
    flow = PrincipalManifoldFlow(
        RealNVP(event_shape=(2,)),
        record_Ihat_P=False,
        record_log_px=False,
        debug=False,
        method=method,
        objective=objective
    )
    flow.fit(x, n_epochs=2, show_progress=False)

    x_new = flow.sample(5)
    lp = flow.log_prob(x_new)

    assert x_new.shape == (5, 2)
    assert lp.shape == (5,)
    assert torch.all(torch.isfinite(x))
    assert torch.all(torch.isfinite(lp))
