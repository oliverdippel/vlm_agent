import torch
import pytest
import torch.nn.functional as F
from src.models.flow_head import FlowMatchingHead, VectorFieldNetwork
from src.entity.config_entity import FlowConfig


@pytest.fixture
def cfg():
    return FlowConfig(
        action_dim=7,
        hidden_dim=64,   # small for fast tests
        n_layers=2,
        conditioning_dim=128,
        n_euler_steps=5,
    )


@pytest.fixture
def flow(cfg):
    return FlowMatchingHead(cfg)


def test_vector_field_output_shape(cfg, flow):
    B = 4
    x_t = torch.randn(B, cfg.action_dim)
    t = torch.rand(B)
    cond = torch.randn(B, cfg.conditioning_dim)
    v = flow.net(x_t, t, cond)
    assert v.shape == (B, cfg.action_dim)


def test_cfm_loss_is_scalar(cfg, flow):
    B = 8
    x_1 = torch.randn(B, cfg.action_dim)
    cond = torch.randn(B, cfg.conditioning_dim)

    loss = flow.cfm_loss(x_1, cond)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_vector_field_can_overfit_fixed_flow_target(cfg, flow):
    """Sanity check: network should fit a fixed flow-matching target."""
    torch.manual_seed(0)

    flow.net.eval()  # disable dropout for deterministic unit test

    optimizer = torch.optim.Adam(flow.net.parameters(), lr=1e-2)

    B = 32
    x_1 = torch.randn(B, cfg.action_dim)
    x_0 = torch.randn_like(x_1)
    cond = torch.randn(B, cfg.conditioning_dim)
    t = torch.rand(B)

    t_exp = t[:, None]
    x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
    target = x_1 - x_0

    losses = []

    for _ in range(100):
        optimizer.zero_grad()

        pred = flow.net(x_t, t, cond)
        loss = F.mse_loss(pred, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_sample_output_shape(cfg, flow):
    B = 4
    cond = torch.randn(B, cfg.conditioning_dim)
    action = flow.sample(cond)
    assert action.shape == (B, cfg.action_dim)


def test_ema_update_changes_weights(cfg, flow):
    """EMA weights should slowly track net weights."""
    before = flow.ema_net.output_proj[-1].bias.clone()
    # do a gradient step on net
    optimizer = torch.optim.Adam(flow.net.parameters(), lr=1e-2)
    x_1 = torch.randn(8, cfg.action_dim)
    cond = torch.randn(8, cfg.conditioning_dim)
    loss = flow.cfm_loss(x_1, cond)
    loss.backward()
    optimizer.step()
    flow.update_ema()
    after = flow.ema_net.output_proj[-1].bias.clone()
    assert not torch.allclose(before, after)


def test_nfe_ablation_same_shape(cfg, flow):
    """Different NFE should all return same shape."""
    cond = torch.randn(2, cfg.conditioning_dim)
    for nfe in [5, 10, 20]:
        action = flow.sample(cond, n_steps=nfe)
        assert action.shape == (2, cfg.action_dim), f"Failed at NFE={nfe}"

def test_ema_parameters_are_frozen(flow):
    for param in flow.ema_net.parameters():
        assert param.requires_grad is False


def test_net_parameters_are_trainable(flow):
    for param in flow.net.parameters():
        assert param.requires_grad is True


def test_sample_runs_without_gradients(cfg, flow):
    cond = torch.randn(4, cfg.conditioning_dim)

    action = flow.sample(cond)

    assert action.requires_grad is False


def test_cfm_loss_produces_gradients(cfg, flow):
    x_1 = torch.randn(8, cfg.action_dim)
    cond = torch.randn(8, cfg.conditioning_dim)

    loss = flow.cfm_loss(x_1, cond)
    loss.backward()

    grads = [
        p.grad
        for p in flow.net.parameters()
        if p.requires_grad
    ]

    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_ema_update_uses_expected_formula(cfg):
    flow = FlowMatchingHead(cfg, ema_decay=0.5)

    ema_param = next(flow.ema_net.parameters())
    net_param = next(flow.net.parameters())

    before_ema = ema_param.clone()

    with torch.no_grad():
        net_param.add_(1.0)

    expected = 0.5 * before_ema + 0.5 * net_param

    flow.update_ema()

    assert torch.allclose(ema_param, expected)


def test_sample_without_ema_output_shape(cfg, flow):
    cond = torch.randn(4, cfg.conditioning_dim)

    action = flow.sample(cond, use_ema=False)

    assert action.shape == (4, cfg.action_dim)


def test_sample_outputs_are_finite(cfg, flow):
    cond = torch.randn(4, cfg.conditioning_dim)

    action = flow.sample(cond)

    assert torch.isfinite(action).all()


def test_cfm_loss_is_finite(cfg, flow):
    x_1 = torch.randn(8, cfg.action_dim)
    cond = torch.randn(8, cfg.conditioning_dim)

    loss = flow.cfm_loss(x_1, cond)

    assert torch.isfinite(loss)


def test_sample_with_one_step(cfg, flow):
    cond = torch.randn(2, cfg.conditioning_dim)

    action = flow.sample(cond, n_steps=1)

    assert action.shape == (2, cfg.action_dim)
    assert torch.isfinite(action).all()