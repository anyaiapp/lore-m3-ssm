import torch

from llm_build.config import ModelConfig
from llm_build.model.lore_m3 import LoreM3


def test_lore_m3_forward():
    cfg = ModelConfig(n_layers=4, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_state=16)
    m = LoreM3(cfg, vocab_size=1000)
    x = torch.randint(0, 1000, (2, 32))
    logits, loss = m(x, x.clone())
    assert logits.shape == (2, 32, 1000)
    assert loss is not None
