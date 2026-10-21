import torch

from llm_build.model.mamba3_ref import resolve_ssm_scan_mode


def test_auto_mps_uses_sequential():
    assert resolve_ssm_scan_mode("auto", 1024, torch.device("mps")) == "sequential"


def test_auto_cuda_without_gpu_falls_back_to_sequential():
    # No CUDA device → auto should not pick triton/parallel
    assert resolve_ssm_scan_mode("auto", 1024, torch.device("cpu")) == "sequential"
