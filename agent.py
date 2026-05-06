import os
import json
import asyncio
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AgentServer,
    JobContext,
    AutoSubscribe,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent

# ONLY using LiveKit plugins - No OpenAI or Google plugins!
from livekit.plugins import livekit, silero

load_dotenv()

# Support the URL format used in your web app
if not os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_WEBSOCKET_URL"):
    os.environ["LIVEKIT_URL"] = os.getenv("LIVEKIT_WEBSOCKET_URL")

SOLACE_INSTRUCTIONS = """
You are Solace, a warm, smart, helpful AI companion.
You can see the user's live screen share. When the user asks about what is on their screen, look at the image provided and answer based on what you see.
If the screen is not visible yet, tell them to share it.

Style:
Speak naturally like a helpful friend. Keep normal voice replies short and clear.
If the user asks for more detail, a full quote, or asks you to continue, give the full answer completely. Never refuse.
"""

server = AgentServer()

@server.rtc_session(agent_name="solace-agent")
async def solace_session(ctx: JobContext):
    # AutoSubscribe ensures we automatically receive the screen share video track
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    print(f"[agent] connected to room: {ctx.room.name}")

    latest_image = None

    # Listen for the screen share starting
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print("[agent] Video track subscribed! Starting frame capture.")
            asyncio.create_task(video_stream_task(track))

    # Continuously save the latest frame from the screen share
    async def video_stream_task(track: rtc.VideoTrack):
        nonlocal latest_image
        video_stream = rtc.VideoStream(track)
        async for event in video_stream:
            latest_image = event.frame

    # THIS IS THE MAGIC: Right before the AI answers, we attach the screen frame!
    def before_llm_cb(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        nonlocal latest_image
        if latest_image:
            # Grab the last message the user just spoke
            last_msg = chat_ctx.messages[-1]
            if last_msg.role == "user":
                # Convert text content to list so we can append the image
                if isinstance(last_msg.content, str):
                    last_msg.content = [last_msg.content]
                
                # Inject the captured screen frame so LiveKit Inference can see it
                last_msg.content.append(llm.ChatImage(image=latest_image))

    # PURE LIVEKIT INFERENCE - NO EXTERNAL API KEYS
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=livekit.STT(), # LiveKit Sandbox STT
        llm=livekit.LLM(model="gpt-4o"), # Proxied to vision model via LiveKit Cloud
        tts=livekit.TTS(), # LiveKit Sandbox TTS
        chat_ctx=llm.ChatContext().append(
            role="system",
            text=SOLACE_INSTRUCTIONS,
        ),
        before_llm_cb=before_llm_cb, # Triggers our frame injection
    )

    await agent.start(ctx.room)
    await agent.say("Hey, I'm here. Share your screen and ask me what I see!")

if __name__ == "__main__":
    cli.run_app(server)
