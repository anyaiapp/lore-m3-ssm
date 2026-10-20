import torch

from llm_build.data.encode import batch_to_causal_labels


def test_causal_labels_mask_user():
    input_ids = torch.tensor([[10, 11, 12, 13]])
    mask = torch.tensor([[0, 0, 1, 1]])
    labels = batch_to_causal_labels(input_ids, mask)
    assert labels[0, 0].item() == -100
    assert labels[0, 1].item() == 12
    assert labels[0, 2].item() == 13
    assert labels[0, 3].item() == -100
