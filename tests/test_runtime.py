import torch

from llm_build.train.runtime import resolve_ssm_backend


def test_ssm_backend_auto_cuda():
    assert resolve_ssm_backend("auto", torch.device("cuda")) == "compiled"


def test_ssm_backend_auto_mps():
    assert resolve_ssm_backend("auto", torch.device("mps")) == "pytorch"
