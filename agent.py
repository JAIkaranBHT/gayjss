import asyncio
import os
import io
import base64
from dotenv import load_dotenv

from PIL import Image as PILImage
from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, get_job_context
from livekit.agents.llm import ImageContent
from livekit.agents import ChatContext
from livekit.plugins import openai, deepgram, cartesia, silero

load_dotenv()

if not os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_WEBSOCKET_URL"):
    os.environ["LIVEKIT_URL"] = os.getenv("LIVEKIT_WEBSOCKET_URL")

INSTRUCTIONS = """
You are Solace, a warm, smart, helpful AI companion.
The user's screen is attached as an image to your context.
When asked what is on the screen, describe EXACTLY what you see in detail.
Never say you cannot see the screen. You always can when screen sharing is active.
Keep replies short and natural unless asked for detail.
"""

latest_pil_image = None


def frame_to_pil(frame) -> PILImage.Image | None:
    try:
        img = PILImage.frombytes("RGBA", (frame.width, frame.height), bytes(frame.data))
        img = img.convert("RGB")
        if img.width > 1280:
            ratio = 1280 / img.width
            img = img.resize((1280, int(img.height * ratio)), PILImage.LANCZOS)
        print(f"[agent] ✅ Frame captured: {img.width}x{img.height}")
        return img
    except Exception as e:
        print(f"[agent] ❌ Frame error: {e}")
        return None


def pil_to_data_uri(img: PILImage.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/jpeg;base64," + b64


class SolaceAgent(Agent):
    def __init__(self):
        self._tasks = []
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self):
        room = get_job_context().room

        async def capture_video_frames(track: rtc.VideoTrack):
            global latest_pil_image
            print(f"[agent] 🎥 Capturing frames from track {track.sid}")
            stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
            async for event in stream:
                latest_pil_image = frame_to_pil(event.frame)

        @room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                print(f"[agent] 📺 Screen share from {participant.identity}!")
                task = asyncio.create_task(capture_video_frames(track))
                self._tasks.append(task)
                task.add_done_callback(lambda t: self._tasks.remove(t))

        # Pick up already-shared tracks
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track and pub.track.kind == rtc.TrackKind.KIND_VIDEO:
                    print("[agent] 📺 Found existing video track!")
                    task = asyncio.create_task(capture_video_frames(pub.track))
                    self._tasks.append(task)
                    task.add_done_callback(lambda t: self._tasks.remove(t))

    async def on_user_turn_completed(self, turn_ctx, new_message):
        global latest_pil_image
        if latest_pil_image is not None:
            try:
                data_uri = pil_to_data_uri(latest_pil_image)
                chat_ctx = self.chat_ctx.copy()
                chat_ctx.add_message(
                    role="user",
                    content=[ImageContent(image=data_uri)]
                )
                await self.update_chat_ctx(chat_ctx)
                print("[agent] ✅ Screen injected via update_chat_ctx!")
            except Exception as e:
                print(f"[agent] ❌ Inject error: {e}")
        else:
            print("[agent] ⚠️ No screen frame yet")


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"[agent] ✅ Connected to room: {ctx.room.name}")

    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(agent=SolaceAgent(), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the user warmly and tell them to share their screen so you can see it."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
