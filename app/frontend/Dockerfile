FROM python:3.11-slim

WORKDIR /app/frontend

# Xvfb + x11vnc + websockify/novnc give this Tkinter GUI a virtual display
# that's streamed to a browser at :6080, since the container has no real screen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    novnc \
    websockify \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ .

ENV DISPLAY=:99
EXPOSE 6080

RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
