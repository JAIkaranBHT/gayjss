"""
Single Railway-facing process.

Runs TWO children concurrently:
  1. The LiveKit agent worker (`python agent.py start`) — outbound WS to LiveKit Cloud
  2. A FastAPI HTTP server on $PORT that serves the frontend and /token endpoint

If either child exits, the whole process exits so Railway restarts cleanly.
"""
import asyncio
import logging
import os
import signal
import sys
from datetime import timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from livekit import api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("solace.server")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Solace")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/config")
async def config():
    ws_url = os.getenv("LIVEKIT_URL") or os.getenv("LIVEKIT_WEBSOCKET_URL")
    if not ws_url:
        raise HTTPException(500, "LIVEKIT_URL is not configured")
    return {"ws_url": ws_url}


@app.get("/token")
async def mint_token(room: str = "solace-test", identity: str = "user"):
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    ws_url = os.getenv("LIVEKIT_URL") or os.getenv("LIVEKIT_WEBSOCKET_URL")

    if not api_key or not api_secret:
        raise HTTPException(500, "LIVEKIT_API_KEY / LIVEKIT_API_SECRET are not set")
    if not ws_url:
        raise HTTPException(500, "LIVEKIT_URL is not set")

    at = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room=room,
                room_join=True,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_ttl(timedelta(hours=1))
    )
    return JSONResponse(
        {"token": at.to_jwt(), "ws_url": ws_url, "room": room, "identity": identity}
    )


# Frontend mounted LAST so API routes take precedence
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


async def run_http() -> None:
    port = int(os.getenv("PORT", "8080"))
    cfg = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", lifespan="on")
    server = uvicorn.Server(cfg)
    log.info("HTTP server starting on 0.0.0.0:%s", port)
    await server.serve()


async def run_worker() -> None:
    """Launch `python agent.py start` as a child process and stream its logs."""
    # Ensure LIVEKIT_URL is set for the child
    if not os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_WEBSOCKET_URL"):
        os.environ["LIVEKIT_URL"] = os.environ["LIVEKIT_WEBSOCKET_URL"]

    cmd = [sys.executable, "-u", "agent.py", "start"]
    log.info("Starting LiveKit worker subprocess: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(Path(__file__).parent),
    )

    async def pump():
        assert proc.stdout is not None
        async for line in proc.stdout:
            sys.stdout.write(f"[worker] {line.decode(errors='replace')}")
            sys.stdout.flush()

    pump_task = asyncio.create_task(pump())
    rc = await proc.wait()
    pump_task.cancel()
    log.error("Worker exited with code %s", rc)
    raise RuntimeError(f"worker exited with code {rc}")


async def main() -> None:
    tasks = [
        asyncio.create_task(run_http(), name="http"),
        asyncio.create_task(run_worker(), name="worker"),
    ]

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: [t.cancel() for t in tasks])
        except NotImplementedError:
            pass

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    for t in done:
        if t.exception():
            log.error("Task %s crashed: %s", t.get_name(), t.exception())
            raise t.exception()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
