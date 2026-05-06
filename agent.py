import asyncio
import os
import io
import base64
from dotenv import load_dotenv

from PIL import Image as PILImage
from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, llm
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

latest_frame = None


def frame_to_data_uri(frame) -> str:
    """Convert LiveKit VideoFrame → base64 JPEG data URI that GPT-4o can read."""
    try:
        img = PILImage.frombytes("RGBA", (frame.width, frame.height), bytes(frame.data))
        img_rgb = img.convert("RGB")
        # Resize if too large to reduce payload
        if img_rgb.width > 1280:
            ratio = 1280 / img_rgb.width
            img_rgb = img_rgb.resize((1280, int(img_rgb.height * ratio)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return "data:image/jpeg;base64," + b64
    except Exception as e:
        print(f"[agent] ❌ Frame conversion failed: {e}")
        return None


class SolaceAgent(Agent):
    def __init__(self):
        super().__init__(instructions=INSTRUCTIONS)

    async def llm_node(self, chat_ctx, tools, model_settings):
        global latest_frame

        if latest_frame is not None:
            data_uri = frame_to_data_uri(latest_frame)
            if data_uri:
                for msg in reversed(chat_ctx.messages):
                    if msg.role == "user":
                        if isinstance(msg.content, str):
                            msg.content = [msg.content]
                        elif not isinstance(msg.content, list):
                            msg.content = []
                        msg.content.append(llm.ChatImage(image=data_uri))
                        print("[agent] ✅ JPEG frame injected into LLM!")
                        break
        else:
            print("[agent] ⚠️ No screen frame yet — share your screen!")

        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk


async def entrypoint(ctx: JobContext):
    global latest_frame

    await ctx.connect()
    print(f"[agent] ✅ Connected to room: {ctx.room.name}")

    async def capture_video_frames(track: rtc.VideoTrack):
        global latest_frame
        print(f"[agent] 🎥 Capturing frames from: {track.sid}")
        stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
        async for event in stream:
            latest_frame = event.frame

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print(f"[agent] 📺 Video track from {participant.identity}!")
            asyncio.create_task(capture_video_frames(track))

    # Handle tracks already published before agent joined
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                print("[agent] 📺 Found existing video track!")
                asyncio.create_task(capture_video_frames(publication.track))

    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(agent=SolaceAgent(), room=ctx.room)
    await session.say("Hey! I'm here. Share your screen and ask me what you see!")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
