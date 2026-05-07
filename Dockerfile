FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download Silero VAD weights so first request is fast
RUN python agent.py download-files || true

EXPOSE 8080

# server.py runs BOTH the LiveKit worker (outbound to LiveKit Cloud) and
# an HTTP server on $PORT that serves the frontend + /token endpoint.
CMD ["python", "server.py"]
