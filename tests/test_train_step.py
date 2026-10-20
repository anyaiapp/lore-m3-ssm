import torch
from torch.utils.data import DataLoader, TensorDataset

from llm_build.config import load_config
from llm_build.model.lore_m3 import LoreM3


def test_one_train_step_cpu():
    cfg = load_config("configs/experiment_100mb.yaml")
    cfg.train.device = "cpu"
    cfg.model.n_layers = 4
    cfg.model.d_model = 64
    cfg.model.n_heads = 4
    cfg.model.n_kv_heads = 2
    cfg.model.head_dim = 16
    cfg.model.d_state = 16
    model = LoreM3(cfg.model, vocab_size=512)
    x = torch.randint(0, 512, (2, 32))
    labels = x.clone()
    labels[:, :16] = -100
    logits, loss = model(x, labels)
    loss.backward()
    assert logits.shape[-1] == 512
