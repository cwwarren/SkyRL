import json
from pathlib import Path

import pytest
from cloudpathlib import CloudPath, implementation_registry
from cloudpathlib.local import local_s3_implementation

from tx.tinker.types import LoraConfig
from tx.utils import models


@pytest.mark.usefixtures("monkeypatch")
def test_save_adapter_config_cloud(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(implementation_registry, "s3", local_s3_implementation)

    client = local_s3_implementation.client_class(local_storage_dir=tmp_path)
    output_dir = CloudPath("s3://bucket/checkpoint", client=client)
    adapter_config = LoraConfig(rank=8, alpha=16)

    models.save_adapter_config(adapter_config, output_dir)

    config_path = output_dir / "adapter_config.json"
    assert config_path.exists()

    saved_config = json.loads(config_path.read_text())
    assert saved_config["r"] == adapter_config.rank
    assert saved_config["lora_alpha"] == adapter_config.alpha
    assert saved_config["peft_type"] == "LORA"


def test_save_adapter_config_local(tmp_path: Path):
    output_dir = tmp_path / "checkpoint"
    output_dir.mkdir()
    adapter_config = LoraConfig(rank=4, alpha=12)

    models.save_adapter_config(adapter_config, output_dir)

    config_path = output_dir / "adapter_config.json"
    assert config_path.exists()

    saved_config = json.loads(config_path.read_text())
    assert saved_config["r"] == adapter_config.rank
    assert saved_config["lora_alpha"] == adapter_config.alpha
    assert saved_config["peft_type"] == "LORA"
