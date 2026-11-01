#!/bin/bash

# ==========================================
# Discord Bot Boot Launcher
# ==========================================

cd /home/aqua/projects/discord-transbot

source ./.venv/bin/activate

echo "Starting Discord Bot at $(date)" >> bot_boot.log
nohup python3 app.py >> bot_boot.log 2>&1 &
