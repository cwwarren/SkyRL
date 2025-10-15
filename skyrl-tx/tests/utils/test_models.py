import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from cloudpathlib import CloudPath, implementation_registry
from cloudpathlib.local import local_s3_implementation
from flax import nnx
from transformers import AutoConfig

from tx.models import Qwen3ForCausalLM
from tx.tinker.types import LoraConfig
from tx.utils import models


def create_test_model():
    """Create a small Qwen3 model for testing."""
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    # Make it smaller for testing
    config.num_hidden_layers = 1
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 2
    config.num_key_value_heads = 2

    mesh = jax.make_mesh((1, 1), ("dp", "tp"))
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
    return config, model


@pytest.mark.parametrize("storage_type", ["local", "cloud"])
def test_save_load_lora_checkpoint(storage_type: str, monkeypatch, tmp_path: Path):
    # Setup output directory based on storage type
    if storage_type == "cloud":
        monkeypatch.setitem(implementation_registry, "s3", local_s3_implementation)
        client = local_s3_implementation.client_class(local_storage_dir=tmp_path)
        output_dir = CloudPath("s3://bucket/checkpoint", client=client)
    else:
        output_dir = tmp_path / "checkpoint"

    # Create a small Qwen3 model
    config, original_model = create_test_model()
    adapter_config = LoraConfig(rank=8, alpha=16)

    # Save the LoRA checkpoint
    models.save_lora_checkpoint(config, adapter_config, original_model, output_dir)

    # Verify adapter_model.safetensors was created
    model_path = output_dir / "adapter_model.safetensors"
    assert model_path.exists()

    # Verify adapter_config.json was created and has correct content
    config_path = output_dir / "adapter_config.json"
    assert config_path.exists()

    saved_config = json.loads(config_path.read_text())
    assert saved_config["r"] == adapter_config.rank
    assert saved_config["lora_alpha"] == adapter_config.alpha
    assert saved_config["peft_type"] == "LORA"

    # Round trip: Load the checkpoint into a new model
    _, loaded_model = create_test_model()
    loaded_config = models.load_lora_checkpoint(output_dir, config, loaded_model)

    # Verify the config was loaded correctly
    assert loaded_config.rank == adapter_config.rank
    assert loaded_config.alpha == adapter_config.alpha

    # Verify some weights match the original
    original_state = nnx.state(original_model)
    loaded_state = nnx.state(loaded_model)

    # Compare a few key parameters
    assert jnp.allclose(
        original_state.model.layers[0].self_attn.q_proj.kernel.value,
        loaded_state.model.layers[0].self_attn.q_proj.kernel.value
    )
    assert jnp.allclose(
        original_state.model.layers[0].self_attn.o_proj.kernel.value,
        loaded_state.model.layers[0].self_attn.o_proj.kernel.value
    )
