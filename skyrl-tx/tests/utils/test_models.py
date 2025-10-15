import json
from pathlib import Path

import pytest
from cloudpathlib import CloudPath, implementation_registry
from cloudpathlib.local import local_s3_implementation
from flax import nnx
import jax.numpy as jnp
from transformers import PretrainedConfig

from tx.tinker.types import LoraConfig
from tx.utils import models


class MockModel(nnx.Module):
    """Minimal mock model for testing save_lora_checkpoint."""

    def __init__(self):
        # Create a simple linear layer with LoRA-like parameters
        self.weight = nnx.Param(jnp.ones((10, 10)))
        self.lora_A = nnx.Param(jnp.ones((10, 8)))
        self.lora_B = nnx.Param(jnp.ones((8, 10)))


@pytest.mark.usefixtures("monkeypatch")
def test_save_lora_checkpoint_cloud(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(implementation_registry, "s3", local_s3_implementation)

    client = local_s3_implementation.client_class(local_storage_dir=tmp_path)
    output_dir = CloudPath("s3://bucket/checkpoint", client=client)

    # Create mock model and config
    model = MockModel()
    config = PretrainedConfig()
    adapter_config = LoraConfig(rank=8, alpha=16)

    # Save the LoRA checkpoint
    models.save_lora_checkpoint(config, adapter_config, model, output_dir)

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


def test_save_lora_checkpoint_local(tmp_path: Path):
    output_dir = tmp_path / "checkpoint"

    # Create mock model and config
    model = MockModel()
    config = PretrainedConfig()
    adapter_config = LoraConfig(rank=4, alpha=12)

    # Save the LoRA checkpoint
    models.save_lora_checkpoint(config, adapter_config, model, output_dir)

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
