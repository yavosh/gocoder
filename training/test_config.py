#!/usr/bin/env python3
"""Basic tests for training config loading and validation."""
import os
import yaml


def test_config_loads():
    config_path = os.path.join(os.path.dirname(__file__), "config", "nemotron-cascade-2.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert "model" in cfg, "config must have 'model' section"
    assert "lora" in cfg, "config must have 'lora' section"
    assert "training" in cfg, "config must have 'training' section"
    assert "dataset" in cfg, "config must have 'dataset' section"

    assert cfg["model"]["name"] == "nvidia/Nemotron-Cascade-2-30B-A3B"
    assert cfg["model"]["load_in_4bit"] is False
    assert cfg["training"]["bf16"] is True
    assert cfg["lora"]["r"] in (16, 32)
    assert 0 < cfg["dataset"]["fim_ratio"] < 1

    print("All config tests passed.")


if __name__ == "__main__":
    test_config_loads()
