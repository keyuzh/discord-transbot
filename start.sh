#!/bin/bash

# ==========================================
# Discord Bot Boot Launcher
# ==========================================

cd /home/aqua/projects/discord-transbot

source ./.venv/bin/activate

echo "Starting Gemini Discord Bot at $(date)" >> bot_boot.log
nohup python3 app_gemini.py >> bot_boot.log 2>&1 &
