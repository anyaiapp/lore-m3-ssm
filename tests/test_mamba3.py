import torch

from llm_build.config import ModelConfig
from llm_build.model.mamba3_block import Mamba3Block


def test_mamba3_backward():
    cfg = ModelConfig(d_model=64, d_state=16, expand=2, mlp_ratio=2.0)
    m = Mamba3Block(cfg)
    x = torch.randn(2, 16, 64, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert x.grad is not None
