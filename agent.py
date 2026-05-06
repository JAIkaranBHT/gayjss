import asyncio
import os
from dotenv import load_dotenv

from PIL import Image as PILImage
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    get_job_context,
)
from livekit.agents.llm import ImageContent
from livekit.plugins import openai, deepgram, cartesia, silero

load_dotenv()

# LiveKit SDK expects LIVEKIT_URL; accept LIVEKIT_WEBSOCKET_URL as a fallback
if not os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_WEBSOCKET_URL"):
    os.environ["LIVEKIT_URL"] = os.getenv("LIVEKIT_WEBSOCKET_URL")

INSTRUCTIONS = """
You are Solace, a warm, smart, helpful AI companion.
The user's screen is attached as an image to your context whenever they are screen sharing.
When asked what is on the screen, describe EXACTLY what you see in detail.
Never say you cannot see the screen. You always can when screen sharing is active.
Keep replies short and natural unless asked for detail.
"""

# Holds the most recent screen-share frame as a PIL Image
latest_pil_image: PILImage.Image | None = None


def frame_to_pil(frame: rtc.VideoFrame) -> PILImage.Image | None:
    """Convert an RGBA VideoFrame into a downscaled RGB PIL image."""
    try:
        img = PILImage.frombytes(
            "RGBA", (frame.width, frame.height), bytes(frame.data)
        )
        img = img.convert("RGB")
        if img.width > 1280:
            ratio = 1280 / img.width
            img = img.resize((1280, int(img.height * ratio)), PILImage.LANCZOS)
        return img
    except Exception as e:
        print(f"[agent] ❌ Frame error: {e}")
        return None


class SolaceAgent(Agent):
    def __init__(self) -> None:
        self._tasks: list[asyncio.Task] = []
        super().__init__(instructions=INSTRUCTIONS)

    async def on_enter(self) -> None:
        room = get_job_context().room

        async def capture_video_frames(track: rtc.RemoteVideoTrack) -> None:
            """Continuously pull frames from the screen-share track."""
            global latest_pil_image
            print(f"[agent] 🎥 Capturing frames from track {track.sid}")

            # Throttle to ~1 fps — we only need a recent frame per user turn
            stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGBA)
            frame_count = 0
            try:
                async for event in stream:
                    frame_count += 1
                    # Sample every ~15th frame (roughly 1-2 fps at typical 15-30fps)
                    if frame_count % 15 != 0:
                        continue
                    img = frame_to_pil(event.frame)
                    if img is not None:
                        latest_pil_image = img
                        if frame_count % 150 == 0:
                            print(
                                f"[agent] ✅ Latest frame: {img.width}x{img.height} "
                                f"(total frames seen: {frame_count})"
                            )
            except Exception as e:
                print(f"[agent] ❌ Stream ended: {e}")
            finally:
                await stream.aclose()

        def is_screenshare(pub_or_track) -> bool:
            """Best-effort check for screen share source."""
            source = getattr(pub_or_track, "source", None)
            if source is None:
                return True  # Fallback: accept any video if source is unknown
            return source == rtc.TrackSource.SOURCE_SCREENSHARE

        def start_capture(track: rtc.RemoteVideoTrack) -> None:
            task = asyncio.create_task(capture_video_frames(track))
            self._tasks.append(task)
            task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)

        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind != rtc.TrackKind.KIND_VIDEO:
                return
            if not is_screenshare(publication):
                print(f"[agent] ℹ️ Ignoring non-screenshare video from {participant.identity}")
                return
            print(f"[agent] 📺 Screen share subscribed from {participant.identity}")
            start_capture(track)

        @room.on("track_published")
        def on_track_published(
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            # Make sure we actively subscribe to any new screen-share track
            if (
                publication.kind == rtc.TrackKind.KIND_VIDEO
                and is_screenshare(publication)
                and not publication.subscribed
            ):
                print(f"[agent] 🔔 Subscribing to new screen share from {participant.identity}")
                publication.set_subscribed(True)

        # Handle tracks that were already published before the agent joined
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.kind != rtc.TrackKind.KIND_VIDEO:
                    continue
                if not is_screenshare(pub):
                    continue
                if not pub.subscribed:
                    print(f"[agent] 🔔 Subscribing to existing screen share from {participant.identity}")
                    pub.set_subscribed(True)
                if pub.track is not None:
                    print(f"[agent] 📺 Found existing screen-share track from {participant.identity}")
                    start_capture(pub.track)

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Attach the latest screen frame to the incoming user message."""
        global latest_pil_image
        if latest_pil_image is None:
            print("[agent] ⚠️ No screen frame yet — user is likely not sharing")
            return

        try:
            # Attach the image directly to the user's current message so the LLM
            # ties the visual context to the question being asked.
            image_content = ImageContent(image=latest_pil_image)

            if isinstance(new_message.content, list):
                new_message.content.append(image_content)
            elif isinstance(new_message.content, str):
                new_message.content = [new_message.content, image_content]
            else:
                new_message.content = [image_content]

            print("[agent] ✅ Screen frame attached to user turn")
        except Exception as e:
            print(f"[agent] ❌ Inject error: {e}")


async def entrypoint(ctx: JobContext) -> None:
    # Auto-subscribe so video tracks actually start flowing to the agent
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
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
