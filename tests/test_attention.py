import torch

from llm_build.config import ModelConfig
from llm_build.model.transformer import TransformerBlock


def test_attention_forward():
    cfg = ModelConfig(d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, max_seq_len=128)
    m = TransformerBlock(cfg)
    x = torch.randn(2, 16, 64)
    y = m(x)
    assert y.shape == x.shape
