#!/usr/bin/env bash
# ==============================================================================
# BalatroBot - VM Linux GCP Automated Setup Script
# Configures Ubuntu (22.04 / 24.04 LTS) for Native LÖVE multi-instance training.
# ==============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}  BALATROBOT - LINUX GCP VM PROVISIONING${NC}"
echo -e "${BLUE}  Platform: Native LÖVE + Lovely + Xvfb${NC}"
echo -e "${BLUE}=======================================================${NC}\n"

# Verify Linux
if [ "$(uname -s)" != "Linux" ]; then
    echo -e "${RED}[Error] This script must be run on Linux.${NC}"
    exit 1
fi

echo -e "${YELLOW}>>> [1/7] Updating apt repositories and installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y \
    curl \
    git \
    unzip \
    tar \
    build-essential \
    xvfb \
    love \
    libglu1-mesa \
    libgl1-mesa-glx \
    tmux \
    htop \
    rsync

echo -e "${GREEN}✓ System dependencies installed.${NC}\n"

echo -e "${YELLOW}>>> [2/7] Installing Astral uv package manager...${NC}"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo -e "${GREEN}✓ uv is available: $(uv --version)${NC}\n"

echo -e "${YELLOW}>>> [3/7] Installing Lovely Injector (liblovely.so for Linux x86_64)...${NC}"
LOVELY_VERSION="v0.8.0"
LOVELY_URL="https://github.com/ethangreen-dev/lovely-injector/releases/download/${LOVELY_VERSION}/lovely-x86_64-unknown-linux-gnu.tar.gz"

curl -sSL -o /tmp/lovely.tar.gz "$LOVELY_URL"
sudo tar -xzf /tmp/lovely.tar.gz -C /usr/local/lib/
sudo chmod 755 /usr/local/lib/liblovely.so
rm -f /tmp/lovely.tar.gz
echo -e "${GREEN}✓ liblovely.so installed at /usr/local/lib/liblovely.so${NC}\n"

echo -e "${YELLOW}>>> [4/7] Setting up Steamodded (SMODS) and BalatroBot Mod...${NC}"
LOVE_CONFIG_DIR="$HOME/.config/love"
MODS_DIR="$LOVE_CONFIG_DIR/Mods"
mkdir -p "$MODS_DIR"

if [ ! -d "$MODS_DIR/smods" ]; then
    echo "Cloning Steamodded (SMODS)..."
    git clone https://github.com/Steamodded/smods.git "$MODS_DIR/smods"
else
    echo "Steamodded already present, updating..."
    git -C "$MODS_DIR/smods" pull || true
fi

# Symlink current repository as balatrobot mod
echo "Symlinking BalatroBot mod into $MODS_DIR/balatrobot..."
ln -sfn "$SCRIPT_DIR" "$MODS_DIR/balatrobot"
echo -e "${GREEN}✓ Mods directory configured.${NC}\n"

echo -e "${YELLOW}>>> [5/7] Verifying Balatro Native Game Directory...${NC}"
GAME_DIR="$SCRIPT_DIR/balatro_game"
if [ ! -f "$GAME_DIR/main.lua" ]; then
    echo -e "${YELLOW}[Warning] $GAME_DIR/main.lua not found!${NC}"
    echo "Make sure the balatro_game/ folder is present at the repository root."
else
    echo -e "${GREEN}✓ balatro_game/ verified with main.lua.${NC}\n"
fi

echo -e "${YELLOW}>>> [6/7] Configuring Xvfb virtual display and environment variables...${NC}"
chmod +x "$SCRIPT_DIR/start_xvfb.sh" 2>/dev/null || true

# Start Xvfb virtual screen on :99 if not running
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "Launching Xvfb on display :99..."
    Xvfb :99 -screen 0 1024x768x24 -nolisten tcp > /dev/null 2>&1 &
    sleep 1
fi

# Generate env_linux.sh for persistent exports
ENV_FILE="$SCRIPT_DIR/env_linux.sh"
cat << EOF > "$ENV_FILE"
#!/usr/bin/env bash
# Source this file before running commands: source ./env_linux.sh
export DISPLAY=:99
export BALATROBOT_PLATFORM=native
export BALATROBOT_LOVE_PATH="\$(which love)"
export BALATROBOT_LOVELY_PATH="/usr/local/lib/liblovely.so"
export BALATROBOT_BALATRO_PATH="$SCRIPT_DIR/balatro_game"
export PATH="\$HOME/.cargo/bin:\$PATH"

# Auto-start Xvfb if not running
if ! pgrep -x "Xvfb" > /dev/null; then
    Xvfb :99 -screen 0 1024x768x24 -nolisten tcp > /dev/null 2>&1 &
fi
EOF
chmod +x "$ENV_FILE"

# Append to ~/.bashrc if not already present
if ! grep -q "source $ENV_FILE" "$HOME/.bashrc" 2>/dev/null; then
    echo -e "\n# BalatroBot environment\nsource \"$ENV_FILE\"" >> "$HOME/.bashrc"
fi
echo -e "${GREEN}✓ Environment configured in $ENV_FILE and added to ~/.bashrc.${NC}\n"

echo -e "${YELLOW}>>> [7/7] Synchronizing Python virtual environment with uv...${NC}"
cd "$SCRIPT_DIR"
uv sync --group dev --group test
echo -e "${GREEN}✓ Python dependencies synced.${NC}\n"

echo -e "${BLUE}=======================================================${NC}"
echo -e "${GREEN}  PROVISIONING COMPLETE!${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo -e "You can now run:"
echo -e "  1. Smoke test:     ${YELLOW}uv run python -m bots.rl.train --smoke-test${NC}"
echo -e "  2. RL Training:    ${YELLOW}uv run python -m bots.rl.train --num-envs 12 --timesteps 500000 --resume${NC}"
echo -e "  3. In tmux:        ${YELLOW}tmux new -s train 'source ./env_linux.sh && uv run python -m bots.rl.train --num-envs 12'${NC}"
echo -e "${BLUE}=======================================================${NC}\n"

