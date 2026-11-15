#!/bin/bash
# Run autopush every N minutes (default 15) until stopped:  nohup tools/autopush_loop.sh 15 &
# Survives closing the terminal, not a reboot (see docs/TRAINING.md for an @reboot cron line).
cd "$(dirname "$0")/.." || exit 1
n=${1:-15}
while true; do tools/autopush.sh >> progress/autopush.log 2>&1; sleep $((n * 60)); done
