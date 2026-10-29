"""macOS platform launcher."""

import os
from pathlib import Path

from balatrobot.config import Config
from balatrobot.platforms.base import BaseLauncher


class MacOSLauncher(BaseLauncher):
    """macOS-specific Balatro launcher."""

    def validate_paths(self, config: Config) -> None:
        """Validate paths, apply macOS defaults if None."""
        if config.love_path is None:
            config.love_path = str(
                Path.home()
                / "Library/Application Support/Steam/steamapps/common/Balatro"
                / "Balatro.app/Contents/MacOS/love"
            )
        if config.lovely_path is None:
            config.lovely_path = str(
                Path.home()
                / "Library/Application Support/Steam/steamapps/common/Balatro"
                / "liblovely.dylib"
            )

        love = Path(config.love_path)
        lovely = Path(config.lovely_path)

        if not love.exists():
            raise RuntimeError(f"LOVE executable not found: {love}")
        if not lovely.exists():
            raise RuntimeError(f"liblovely.dylib not found: {lovely}")

    def build_env(self, config: Config) -> dict[str, str]:
        """Build environment with DYLD_INSERT_LIBRARIES."""
        assert config.lovely_path is not None
        env = os.environ.copy()
        env["DYLD_INSERT_LIBRARIES"] = config.lovely_path
        env.update(config.to_env())
        return env

    def build_cmd(self, config: Config) -> list[str]:
        """Build macOS launch command."""
        assert config.love_path is not None
        return [config.love_path]
