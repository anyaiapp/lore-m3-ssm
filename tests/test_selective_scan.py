import torch

from llm_build.model.mamba3_ref import selective_scan_parallel, selective_scan_sequential


def test_parallel_matches_sequential():
    torch.manual_seed(0)
    b, t, d = 2, 128, 32
    u = torch.randn(b, t, d)
    delta = torch.randn(b, t, d)
    a = torch.randn(b, t, d)
    bvec = torch.randn(b, t, d)
    c = torch.randn(b, t, d)
    y_ref = selective_scan_sequential(u, delta, a, bvec, c)
    y_par = selective_scan_parallel(u, delta, a, bvec, c)
    torch.testing.assert_close(y_par, y_ref, rtol=1e-4, atol=1e-4)


def test_parallel_grad():
    b, t, d = 1, 64, 16
    u = torch.randn(b, t, d, requires_grad=True)
    delta = torch.randn(b, t, d, requires_grad=True)
    a = torch.randn(b, t, d, requires_grad=True)
    bvec = torch.randn(b, t, d, requires_grad=True)
    c = torch.randn(b, t, d, requires_grad=True)
    y = selective_scan_parallel(u, delta, a, bvec, c)
    y.sum().backward()
    assert u.grad is not None
