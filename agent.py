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
You CAN see the user's live screen share — it is attached as an image to every message.
When the user asks what is on their screen, describe exactly what you see in the image.
Speak naturally like a helpful friend. Keep replies short and clear unless asked for more detail.
"""

# Global reference to the latest screen frame
latest_frame: rtc.VideoFrame | None = None


class SolaceAgent(Agent):
    def __init__(self):
        super().__init__(instructions=INSTRUCTIONS)

    async def llm_node(self, chat_ctx, tools, model_settings):
        global latest_frame

        # Inject latest screen frame into the last user message before sending to LLM
        if latest_frame is not None:
            for msg in reversed(chat_ctx.messages):
                if msg.role == "user":
                    # Make sure content is a list so we can append the image
                    if isinstance(msg.content, str):
                        msg.content = [msg.content]
                    elif not isinstance(msg.content, list):
                        msg.content = []
                    msg.content.append(llm.ChatImage(image=latest_frame))
                    print("[agent] ✅ Screen frame injected into LLM context!")
                    break
        else:
            print("[agent] ⚠️ No screen frame available yet.")

        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk


async def entrypoint(ctx: JobContext):
    global latest_frame

    await ctx.connect()
    print(f"[agent] ✅ Connected to room: {ctx.room.name}")

    async def capture_video_frames(track: rtc.VideoTrack):
        global latest_frame
        print(f"[agent] 🎥 Started capturing frames from track: {track.sid}")
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
        async for event in stream:
            latest_frame = event.frame

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print(f"[agent] 📺 Video track detected from {participant.identity}! Starting capture.")
            asyncio.create_task(capture_video_frames(track))

    # Check if video track is already published before we subscribed
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if (
                publication.track is not None
                and publication.track.kind == rtc.TrackKind.KIND_VIDEO
            ):
                print("[agent] 📺 Found existing video track, capturing...")
                asyncio.create_task(capture_video_frames(publication.track))

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

    await session.say("Hey! I'm here and I can see your screen. Ask me anything about what's on it!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
