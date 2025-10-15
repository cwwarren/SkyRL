from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import safetensors.numpy
from cloudpathlib import CloudPath, implementation_registry
from cloudpathlib.local import local_s3_implementation
from flax import nnx
from peft import LoraConfig as PEFTLoraConfig
from transformers import AutoConfig

from tx.layers.lora import update_adapter_config
from tx.models import Qwen3ForCausalLM
from tx.tinker.types import LoraConfig
from tx.utils import models
from tx.utils.storage import download_and_unpack


def create_test_model():
    """Create a small Qwen3 model for testing with LoRA enabled."""
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    # Make it smaller for testing
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.max_lora_adapters = 5
    config.max_lora_rank = 32

    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        update_adapter_config(model, adapter_index=0, lora_rank=8, lora_alpha=16)

    return config, model


@pytest.mark.parametrize("storage_type", ["local", "cloud"])
def test_save_load_lora_checkpoint(storage_type: str, monkeypatch, tmp_path: Path):
    # Setup output path for tar.gz file based on storage type
    if storage_type == "cloud":
        monkeypatch.setitem(implementation_registry, "s3", local_s3_implementation)
        client = local_s3_implementation.client_class(local_storage_dir=tmp_path)
        output_path = CloudPath("s3://bucket/checkpoint.tar.gz", client=client)
    else:
        output_path = tmp_path / "checkpoint.tar.gz"

    # Create a small Qwen3 model
    config, original_model = create_test_model()
    adapter_config = LoraConfig(rank=8, alpha=16)

    # Modify LoRA weights to specific values for testing
    original_model.model.layers[0].self_attn.q_proj.lora_A.value = jnp.ones_like(
        original_model.model.layers[0].self_attn.q_proj.lora_A.value
    )
    original_model.model.layers[0].self_attn.q_proj.lora_B.value = (
        jnp.ones_like(original_model.model.layers[0].self_attn.q_proj.lora_B.value) * 2.0
    )

    # Save the LoRA checkpoint as tar.gz
    models.save_lora_checkpoint(original_model, adapter_config, adapter_index=0, output_path=output_path)

    # Verify tar.gz file was created
    assert output_path.exists()

    # Verify the checkpoint by extracting and loading it with safetensors
    with download_and_unpack(output_path) as extracted_dir:
        # Load the PEFT config
        peft_config = PEFTLoraConfig.from_pretrained(extracted_dir)
        assert peft_config.r == adapter_config.rank
        assert peft_config.lora_alpha == adapter_config.alpha

        # Load the adapter weights directly from safetensors
        adapter_weights = safetensors.numpy.load_file(extracted_dir / "adapter_model.safetensors")

        # Verify the adapter weights match what we set
        lora_A = adapter_weights["model.layers.0.self_attn.q_proj.lora_A.weight"]
        lora_B = adapter_weights["model.layers.0.self_attn.q_proj.lora_B.weight"]

        # Note: Our JAX model has shape (adapter_index, ..., features, rank) but we extract
        # a single adapter, so the saved weights are (features, rank) and need to be transposed
        # because save_safetensors does param.T
        assert np.allclose(lora_A.T, np.ones_like(lora_A.T))
        assert np.allclose(lora_B.T, np.ones_like(lora_B.T) * 2.0)
