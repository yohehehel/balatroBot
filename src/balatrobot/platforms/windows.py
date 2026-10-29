"""Windows platform launcher."""

import os
from pathlib import Path

from balatrobot.config import Config
from balatrobot.platforms.base import BaseLauncher


class WindowsLauncher(BaseLauncher):
    """Windows-specific Balatro launcher via Steam."""

    def validate_paths(self, config: Config) -> None:
        """Validate paths, apply Windows defaults if None."""
        if config.love_path is None:
            config.love_path = (
                r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe"
            )

        love = Path(config.love_path)
        if not love.exists():
            raise RuntimeError(f"Balatro executable not found: {love}")

        if config.lovely_path is None:
            base_dir = love.parent
            for candidate in ("version.dll", "winmm.dll", "dwmapi.dll"):
                p = base_dir / candidate
                if p.exists():
                    config.lovely_path = str(p)
                    break
            if config.lovely_path is None:
                config.lovely_path = str(base_dir / "version.dll")

        lovely = Path(config.lovely_path)
        if not lovely.exists():
            raise RuntimeError(f"version.dll not found: {lovely}")

    def build_env(self, config: Config) -> dict[str, str]:
        """Build environment."""
        env = os.environ.copy()
        env.update(config.to_env())
        return env

    def build_cmd(self, config: Config) -> list[str]:
        """Build Windows launch command."""
        assert config.love_path is not None
        return [config.love_path]
