#!/usr/bin/env bash
# Source this file before running bots on Linux: source ./env_linux.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DISPLAY=:99
export BALATROBOT_PLATFORM=native
export BALATROBOT_LOVE_PATH="$(which love 2>/dev/null || echo '/usr/bin/love')"
export BALATROBOT_LOVELY_PATH="/usr/local/lib/liblovely.so"
export BALATROBOT_BALATRO_PATH="$SCRIPT_DIR/balatro_game"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! pgrep -x "Xvfb" > /dev/null 2>&1; then
    Xvfb :99 -screen 0 1024x768x24 -nolisten tcp > /dev/null 2>&1 &
fi

