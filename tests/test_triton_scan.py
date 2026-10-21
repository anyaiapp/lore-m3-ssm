import pytest
import torch

from llm_build.model.mamba3_ref import selective_scan_sequential
from llm_build.model.mamba3_scan_triton import selective_scan_triton, triton_scan_available


@pytest.mark.skipif(not triton_scan_available(), reason="CUDA+Triton required")
def test_triton_forward_matches_sequential():
    torch.manual_seed(0)
    b, t, d = 2, 128, 32
    u = torch.randn(b, t, d, device="cuda")
    delta = torch.randn(b, t, d, device="cuda")
    a = torch.randn(b, t, d, device="cuda")
    bvec = torch.randn(b, t, d, device="cuda")
    c = torch.randn(b, t, d, device="cuda")
    y_ref = selective_scan_sequential(u, delta, a, bvec, c)
    y_tri = selective_scan_triton(u, delta, a, bvec, c)
    torch.testing.assert_close(y_tri, y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not triton_scan_available(), reason="CUDA+Triton required")
def test_triton_grad_matches_sequential():
    torch.manual_seed(1)
    b, t, d = 2, 64, 16
    specs = [
        torch.randn(b, t, d, device="cuda")
        for _ in range(5)
    ]
    dy = torch.randn(b, t, d, device="cuda")

    def grad_ref():
        u, delta, a, bvec, c = [x.detach().requires_grad_(True) for x in specs]
        y = selective_scan_sequential(u, delta, a, bvec, c)
        return torch.autograd.grad(y, (u, delta, a, bvec, c), grad_outputs=dy)

    def grad_tri():
        u, delta, a, bvec, c = [x.detach().requires_grad_(True) for x in specs]
        y = selective_scan_triton(u, delta, a, bvec, c)
        return torch.autograd.grad(y, (u, delta, a, bvec, c), grad_outputs=dy)

    ref = grad_ref()
    tri = grad_tri()
    for gr, gt in zip(ref, tri):
        torch.testing.assert_close(gt, gr, rtol=2e-3, atol=2e-3)
