import torch

from llm_build.config import ModelConfig
from llm_build.model.mamba3_block import Mamba3Block
from llm_build.model.mamba3_ref import selective_scan_parallel, selective_scan_sequential


def test_selective_scan_parallel_matches_sequential():
    torch.manual_seed(0)
    b, t, d = 2, 128, 32
    u = torch.randn(b, t, d)
    delta = torch.randn(b, t, d)
    a = torch.randn(b, t, d)
    b_coef = torch.randn(b, t, d)
    c = torch.randn(b, t, d)
    y_ref = selective_scan_sequential(u, delta, a, b_coef, c)
    y_par = selective_scan_parallel(u, delta, a, b_coef, c)
    assert torch.allclose(y_ref, y_par, rtol=1e-4, atol=1e-5)


def test_mamba3_backward():
    cfg = ModelConfig(d_model=64, d_state=16, expand=2, mlp_ratio=2.0)
    m = Mamba3Block(cfg)
    x = torch.randn(2, 16, 64, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert x.grad is not None
