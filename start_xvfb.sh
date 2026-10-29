#!/usr/bin/env bash
# ==============================================================================
# Helper to check and start Xvfb (Virtual Framebuffer) on display :99
# ==============================================================================

if pgrep -x "Xvfb" > /dev/null; then
    echo "Xvfb is already running (PID: $(pgrep -x "Xvfb"))."
else
    echo "Starting Xvfb on display :99..."
    Xvfb :99 -screen 0 1024x768x24 -nolisten tcp > /dev/null 2>&1 &
    sleep 1
    if pgrep -x "Xvfb" > /dev/null; then
        echo "Xvfb successfully started on display :99."
    else
        echo "Error: Failed to start Xvfb."
        exit 1
    fi
fi

