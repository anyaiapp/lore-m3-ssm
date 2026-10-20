import torch

from llm_build.model.mlp import SwiGLU
from llm_build.model.norms import RMSNorm


def test_rmsnorm_shape():
    x = torch.randn(2, 8, 16)
    y = RMSNorm(16)(x)
    assert y.shape == x.shape


def test_swiglu_shape():
    x = torch.randn(2, 8, 32)
    y = SwiGLU(32, 2.0)(x)
    assert y.shape == x.shape
