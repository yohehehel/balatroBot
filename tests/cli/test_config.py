"""Tests for balatrobot.config module."""

from argparse import Namespace

import pytest

from balatrobot.config import Config, _parse_env_value


class TestParseEnvValue:
    """Tests for _parse_env_value type conversion."""

    def test_bool_true_values(self):
        """Boolean fields convert '1' and 'true' to True."""
        assert _parse_env_value("fast", "1") is True
        assert _parse_env_value("fast", "true") is True
        assert _parse_env_value("headless", "1") is True

    def test_bool_false_values(self):
        """Boolean fields convert other values to False."""
        assert _parse_env_value("fast", "0") is False
        assert _parse_env_value("fast", "false") is False
        assert _parse_env_value("fast", "yes") is False

    def test_int_valid(self):
        """Integer fields parse valid numbers."""
        assert _parse_env_value("port", "12346") == 12346
        assert _parse_env_value("port", "9999") == 9999

    def test_int_invalid(self):
        """Integer fields raise ValueError for invalid input."""
        with pytest.raises(ValueError):
            _parse_env_value("port", "abc")

    def test_string_passthrough(self):
        """String fields pass through unchanged."""
        assert _parse_env_value("host", "localhost") == "localhost"
        assert _parse_env_value("balatro_path", "/path/to/game") == "/path/to/game"


class TestConfigDefaults:
    """Tests for Config default values."""

    def test_defaults(self, clean_env):
        """Config has correct default values."""
        config = Config()

        assert config.host == "127.0.0.1"
        assert config.port == 12346
        assert config.fast is False
        assert config.headless is False
        assert config.logs_path == "logs"
        assert config.balatro_path is None


class TestConfigFromArgs:
    """Tests for Config.from_args() method."""

    def test_cli_args_used(self, clean_env):
        """CLI arguments are used when provided."""
        args = Namespace(
            host="0.0.0.0",
            port=9999,
            fast=True,
            headless=None,
            render_on_api=None,
            audio=None,
            debug=None,
            no_shaders=None,
            balatro_path=None,
            lovely_path=None,
            love_path=None,
            platform=None,
            logs_path=None,
        )
        config = Config.from_args(args)

        assert config.host == "0.0.0.0"
        assert config.port == 9999
        assert config.fast is True

    def test_cli_overrides_env(self, clean_env, monkeypatch):
        """CLI args override environment variables."""
        monkeypatch.setenv("BALATROBOT_PORT", "8888")

        args = Namespace(
            host=None,
            port=9999,
            fast=None,
            headless=None,
            render_on_api=None,
            audio=None,
            debug=None,
            no_shaders=None,
            balatro_path=None,
            lovely_path=None,
            love_path=None,
            platform=None,
            logs_path=None,
        )
        config = Config.from_args(args)

        assert config.port == 9999  # CLI wins over env

    def test_env_fallback(self, clean_env, monkeypatch):
        """Environment variables used when CLI args are None."""
        monkeypatch.setenv("BALATROBOT_PORT", "8888")
        monkeypatch.setenv("BALATROBOT_FAST", "1")

        args = Namespace(
            host=None,
            port=None,
            fast=None,
            headless=None,
            render_on_api=None,
            audio=None,
            debug=None,
            no_shaders=None,
            balatro_path=None,
            lovely_path=None,
            love_path=None,
            platform=None,
            logs_path=None,
        )
        config = Config.from_args(args)

        assert config.port == 8888
        assert config.fast is True


class TestConfigFromEnv:
    """Tests for Config.from_env() method."""

    def test_loads_env_vars(self, clean_env, monkeypatch):
        """Loads configuration from environment variables."""
        monkeypatch.setenv("BALATROBOT_PORT", "9999")
        monkeypatch.setenv("BALATROBOT_HOST", "0.0.0.0")
        monkeypatch.setenv("BALATROBOT_FAST", "1")

        config = Config.from_env()

        assert config.port == 9999
        assert config.host == "0.0.0.0"
        assert config.fast is True

    def test_defaults_when_no_env(self, clean_env):
        """Uses defaults when no env vars set."""
        config = Config.from_env()

        assert config.port == 12346
        assert config.host == "127.0.0.1"


class TestConfigToEnv:
    """Tests for Config.to_env() method."""

    def test_serializes_values(self):
        """Serializes config to environment dict."""
        config = Config(port=9999, fast=True, host="0.0.0.0")
        env = config.to_env()

        assert env["BALATROBOT_PORT"] == "9999"
        assert env["BALATROBOT_FAST"] == "1"
        assert env["BALATROBOT_HOST"] == "0.0.0.0"

    def test_skips_none_values(self):
        """None values are not included."""
        config = Config(balatro_path=None)
        env = config.to_env()

        assert "BALATROBOT_BALATRO_PATH" not in env

    def test_skips_false_bools(self):
        """False boolean values are not included."""
        config = Config(fast=False, headless=False)
        env = config.to_env()

        assert "BALATROBOT_FAST" not in env
        assert "BALATROBOT_HEADLESS" not in env
