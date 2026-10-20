from llm_build.model.layer_schedule import attention_layer_indices


def test_attention_indices_l20():
    assert attention_layer_indices(20, "mamba3_5to1") == [5, 11, 17]
