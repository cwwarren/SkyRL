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

    # Save the LoRA checkpoint as tar.gz
    models.save_lora_checkpoint(config, adapter_config, original_model, output_path)

    # Verify tar.gz file was created
    assert output_path.exists()

    # Verify the tar.gz contains the expected files
    import tarfile
    if isinstance(output_path, CloudPath):
        # Download to verify contents
        temp_tar = tmp_path / "temp.tar.gz"
        output_path.download_to(temp_tar)
        tar_path = temp_tar
    else:
        tar_path = output_path

    with tarfile.open(tar_path, "r:gz") as tar:
        names = [member.name for member in tar.getmembers()]
        assert "adapter_model.safetensors" in names
        assert "adapter_config.json" in names

        # Verify adapter_config.json has correct content
        config_file = tar.extractfile("adapter_config.json")
        saved_config = json.loads(config_file.read())
        assert saved_config["r"] == adapter_config.rank
        assert saved_config["lora_alpha"] == adapter_config.alpha
        assert saved_config["peft_type"] == "LORA"

    # Round trip: Load the checkpoint from tar.gz into a new model
    _, loaded_model = create_test_model()
    loaded_config = models.load_lora_checkpoint(output_path, config, loaded_model)

    # Verify the config was loaded correctly
    assert loaded_config.rank == adapter_config.rank
    assert loaded_config.alpha == adapter_config.alpha

    # Verify some weights match the original
    original_state = nnx.state(original_model)
    loaded_state = nnx.state(loaded_model)

    # Compare a few key parameters
    assert jnp.allclose(
        original_state.model.layers[0].self_attn.q_proj.kernel.value,
        loaded_state.model.layers[0].self_attn.q_proj.kernel.value,
    )
    assert jnp.allclose(
        original_state.model.layers[0].self_attn.o_proj.kernel.value,
        loaded_state.model.layers[0].self_attn.o_proj.kernel.value,
    )
