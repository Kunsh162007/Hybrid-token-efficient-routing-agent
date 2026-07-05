"""Config loading and validation tests."""

import pytest

from routing_agent.config import AppConfig, ConfigError, get_api_key, load_config


def test_load_config_defaults_when_no_file(tmp_path, monkeypatch):
    # Arrange: point default lookup at an empty directory
    monkeypatch.chdir(tmp_path)

    # Act
    cfg = load_config()

    # Assert
    assert isinstance(cfg, AppConfig)
    assert cfg.ladder.per_task_token_budget == 2000


def test_load_config_reads_yaml_overrides(tmp_path):
    # Arrange
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "ladder:\n  per_task_token_budget: 500\nremote:\n  base_url: 'https://x.test/v1/'\n",
        encoding="utf-8",
    )

    # Act
    cfg = load_config(cfg_file)

    # Assert
    assert cfg.ladder.per_task_token_budget == 500
    assert cfg.remote.base_url == "https://x.test/v1"  # trailing slash stripped


def test_load_config_explicit_missing_path_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_rejects_non_mapping(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)


def test_load_config_rejects_invalid_values(tmp_path):
    from pydantic import ValidationError

    bad = tmp_path / "config.yaml"
    bad.write_text("ladder:\n  confidence_threshold: 5.0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_get_api_key_missing_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # isolate from any real .env in the repo root
    with pytest.raises(ConfigError, match="FIREWORKS_API_KEY"):
        get_api_key()


def test_get_api_key_reads_env(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test123")
    assert get_api_key() == "fw_test123"


def test_get_api_key_falls_back_to_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "# comment\nFIREWORKS_API_KEY='fw_from_file'\n", encoding="utf-8"
    )

    assert get_api_key() == "fw_from_file"


def test_dotenv_never_overrides_real_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_real")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("FIREWORKS_API_KEY=fw_file\n", encoding="utf-8")

    assert get_api_key() == "fw_real"
