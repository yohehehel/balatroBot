"""Terminal progress bar display and SB3 callback for Macro-RL training.

Renders a live, pinned progress bar at the bottom of the terminal window
displaying current vs total timesteps (e.g. `2,500 / 5,000`), percentage,
steps/s, and ETA, while regular instance logs scroll cleanly above it.
"""

from __future__ import annotations

import os
import sys
import time

from stable_baselines3.common.callbacks import BaseCallback

# Enable Windows Virtual Terminal Processing if running on Windows
if sys.platform == "win32":
    try:
        os.system("")  # Activates VT100/ANSI mode in Windows console
    except Exception:
        pass


def _format_eta(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    if seconds < 0 or seconds > 360000:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class ProgressDisplay:
    """Manages the in-place terminal progress bar pinned at the bottom."""

    _instance: ProgressDisplay | None = None

    def __init__(self, total_timesteps: int, initial_timesteps: int = 0, bar_length: int = 24) -> None:
        ProgressDisplay._instance = self
        self.initial_timesteps = initial_timesteps
        self.total_timesteps = max(1, total_timesteps)
        self.bar_length = bar_length
        self.current_timesteps = initial_timesteps
        self.start_time = time.time()
        self.is_tty = sys.stdout.isatty()
        self._max_line_len = 95
        self._last_non_tty_print = initial_timesteps

    @classmethod
    def get_instance(cls) -> ProgressDisplay | None:
        return cls._instance

    def update(self, current_timesteps: int) -> None:
        """Update the progress counter and re-render the status bar."""
        self.current_timesteps = current_timesteps
        self.render()

    def render(self) -> None:
        """Draw or refresh the status line."""
        completed = max(0, self.current_timesteps - self.initial_timesteps)
        needed = max(1, self.total_timesteps - self.initial_timesteps)
        pct = min(100.0, (completed / needed) * 100.0)
        elapsed = time.time() - self.start_time
        fps = completed / max(0.001, elapsed)
        remaining = max(0, self.total_timesteps - self.current_timesteps)
        eta_str = _format_eta(remaining / max(0.001, fps))

        filled = min(self.bar_length, int(self.bar_length * (completed / needed)))
        bar = "=" * filled + "-" * (self.bar_length - filled)
        status = (
            f"[Progression: [{bar}] {self.current_timesteps:,} / {self.total_timesteps:,} "
            f"({pct:5.1f}%) | {fps:.1f} steps/s | ETA: {eta_str}]"
        )

        self._max_line_len = max(self._max_line_len, len(status))

        if self.is_tty:
            sys.stdout.write(f"\r{status.ljust(self._max_line_len)}")
            sys.stdout.flush()
        else:
            # In non-interactive logs, print periodically (every 500 steps or at end)
            if (
                self.current_timesteps - self._last_non_tty_print >= 500
                or self.current_timesteps >= self.total_timesteps
            ):
                self._last_non_tty_print = self.current_timesteps
                print(status)

    def log(self, message: str) -> None:
        """Print a normal scrolling log message cleanly above the pinned progress line."""
        self._max_line_len = max(self._max_line_len, len(message))
        if self.is_tty:
            # Clear the bottom progress line, print the message, and restore the progress line
            sys.stdout.write(f"\r{message.ljust(self._max_line_len)}\n")
            self.render()
        else:
            print(message)

    def close(self) -> None:
        """Finalize the progress bar on training completion."""
        if self.is_tty:
            self.render()
            sys.stdout.write("\n")
            sys.stdout.flush()
        ProgressDisplay._instance = None


def log_message(message: str) -> None:
    """Route a log message through the active ProgressDisplay if present, else print normally."""
    disp = ProgressDisplay.get_instance()
    if disp is not None:
        disp.log(message)
    else:
        try:
            print(message)
        except UnicodeEncodeError:
            clean_msg = message.encode("ascii", errors="replace").decode("ascii")
            print(clean_msg)


class TrainingProgressCallback(BaseCallback):
    """Stable-Baselines3 Callback to continuously update ProgressDisplay."""

    def __init__(self, display: ProgressDisplay, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.display = display

    def _on_step(self) -> bool:
        self.display.update(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self.display.update(self.display.total_timesteps)
        self.display.close()

