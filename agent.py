import asyncio
import os
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import openai, deepgram, cartesia, silero

load_dotenv()

if not os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_WEBSOCKET_URL"):
    os.environ["LIVEKIT_URL"] = os.getenv("LIVEKIT_WEBSOCKET_URL")

INSTRUCTIONS = """
You are Solace, a warm, smart, helpful AI companion.
You can see the user's live screen share. When the user asks about what is on their screen, look at the image provided and answer based on what you see.
If the screen is not visible yet, tell them to share it.
Speak naturally like a helpful friend. Keep replies short and clear.
"""

latest_image = None

class SolaceAgent(Agent):
    def __init__(self):
        super().__init__(instructions=INSTRUCTIONS)

    async def llm_node(self, chat_ctx, tools, model_settings):
        global latest_image
        if latest_image:
            for msg in reversed(chat_ctx.messages):
                if msg.role == "user":
                    if isinstance(msg.content, str):
                        msg.content = [msg.content]
                    msg.content.append(llm.ChatImage(image=latest_image))
                    print("[agent] Injected screen frame into LLM context.")
                    break
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk


async def entrypoint(ctx: JobContext):
    global latest_image

    await ctx.connect()
    print(f"[agent] connected to room: {ctx.room.name}")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print("[agent] Screen share detected! Capturing frames.")
            asyncio.create_task(capture_frames(track))

    async def capture_frames(track: rtc.VideoTrack):
        global latest_image
        stream = rtc.VideoStream(track)
        async for event in stream:
            latest_image = event.frame

    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=SolaceAgent(),
        room=ctx.room,
    )

    await session.say("Hey, I'm here. Share your screen and ask me what you see!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
