#!/bin/sh
set -e

rm -f "/tmp/.X99-lock"

Xvfb "$DISPLAY" -screen 0 1280x800x24 &
sleep 2

x11vnc -display "$DISPLAY" -forever -shared -nopw -rfbport 5900 -quiet &

websockify --web /usr/share/novnc 6080 localhost:5900 &
sleep 1

exec python app.py
