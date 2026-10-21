import torch

from llm_build.config import ModelConfig
from llm_build.model.lore_m3 import LoreM3


def test_grad_checkpoint_forward_backward():
    cfg = ModelConfig(n_layers=4, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_state=16)
    model = LoreM3(cfg, vocab_size=256, grad_checkpoint=True)
    model.train()
    x = torch.randint(0, 256, (2, 32))
    labels = x.clone()
    labels[:, :8] = -100
    logits, loss = model(x, labels)
    loss.backward()
    assert logits.shape == (2, 32, 256)
    assert any(p.grad is not None for p in model.parameters())
