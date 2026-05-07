import {
  Room,
  RoomEvent,
  Track,
  createLocalAudioTrack,
  createLocalScreenTracks,
  ConnectionState,
} from "livekit-client";

const els = {
  roomName: document.getElementById("roomName"),
  joinBtn: document.getElementById("joinBtn"),
  leaveBtn: document.getElementById("leaveBtn"),
  shareBtn: document.getElementById("shareBtn"),
  stopShareBtn: document.getElementById("stopShareBtn"),
  muteBtn: document.getElementById("muteBtn"),
  statusPill: document.getElementById("statusPill"),
  setup: document.getElementById("setup"),
  stage: document.getElementById("stage"),
  localScreen: document.getElementById("localScreen"),
  log: document.getElementById("log"),
  vuBar: document.querySelector("#vu .bar"),
  agentMeta: document.getElementById("agentMeta"),
};

const saved = localStorage.getItem("solace:roomName");
if (saved) els.roomName.value = saved;
els.roomName.addEventListener("change", () =>
  localStorage.setItem("solace:roomName", els.roomName.value)
);

let room = null;
let localAudioTrack = null;
let localScreenTracks = [];
let audioCtx = null;
let vuRaf = null;

function log(msg, cls = "") {
  const line = document.createElement("div");
  line.className = cls;
  const ts = new Date().toLocaleTimeString();
  line.textContent = `[${ts}] ${msg}`;
  els.log.appendChild(line);
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(text, cls = "") {
  els.statusPill.textContent = text;
  els.statusPill.className = "pill " + cls;
}

async function fetchToken(roomName) {
  const identity = "user-" + Math.random().toString(36).slice(2, 8);
  const res = await fetch(
    `/token?room=${encodeURIComponent(roomName)}&identity=${encodeURIComponent(identity)}`
  );
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Token endpoint ${res.status}: ${txt}`);
  }
  return res.json(); // { token, ws_url, room, identity }
}

async function join() {
  const roomName = els.roomName.value.trim() || "solace-test";
  els.joinBtn.disabled = true;
  setStatus("Fetching token…");

  let ws_url, token;
  try {
    ({ ws_url, token } = await fetchToken(roomName));
    log(`Token received for room "${roomName}"`, "ok");
  } catch (e) {
    log(`Token fetch failed: ${e.message}`, "err");
    setStatus("Error", "error");
    els.joinBtn.disabled = false;
    return;
  }

  setStatus("Connecting…");
  room = new Room({ adaptiveStream: true, dynacast: true });

  room
    .on(RoomEvent.ConnectionStateChanged, (state) => {
      log(`Connection state: ${state}`, "info");
      if (state === ConnectionState.Connected) setStatus("Connected", "connected");
      if (state === ConnectionState.Disconnected) setStatus("Disconnected");
      if (state === ConnectionState.Reconnecting) setStatus("Reconnecting…");
    })
    .on(RoomEvent.ParticipantConnected, (p) => {
      log(`Participant joined: ${p.identity}`, "ok");
      const id = p.identity.toLowerCase();
      if (id.includes("agent") || id.includes("solace")) {
        els.agentMeta.textContent = `Solace is here (${p.identity})`;
      }
    })
    .on(RoomEvent.ParticipantDisconnected, (p) => log(`Participant left: ${p.identity}`))
    .on(RoomEvent.TrackSubscribed, (track, pub, participant) => {
      log(`Subscribed to ${track.kind} from ${participant.identity}`, "ok");
      if (track.kind === Track.Kind.Audio) {
        const el = track.attach();
        el.autoplay = true;
        el.playsInline = true;
        document.body.appendChild(el);
        attachVuMeter(track.mediaStreamTrack);
      }
    })
    .on(RoomEvent.TrackUnsubscribed, (track) => track.detach().forEach((el) => el.remove()))
    .on(RoomEvent.Disconnected, (reason) => {
      log(`Disconnected (${reason ?? "unknown"})`);
      cleanup();
    });

  try {
    await room.connect(ws_url, token);
    log(`Connected to ${room.name}`, "ok");

    localAudioTrack = await createLocalAudioTrack({
      echoCancellation: true,
      noiseSuppression: true,
    });
    await room.localParticipant.publishTrack(localAudioTrack);
    log("Microphone published", "ok");

    els.setup.hidden = true;
    els.stage.hidden = false;
    els.leaveBtn.disabled = false;
  } catch (e) {
    log(`Join failed: ${e.message || e}`, "err");
    setStatus("Error", "error");
    els.joinBtn.disabled = false;
  }
}

async function startScreenShare() {
  if (!room) return;
  els.shareBtn.disabled = true;
  try {
    localScreenTracks = await createLocalScreenTracks({ audio: true });
    for (const track of localScreenTracks) {
      await room.localParticipant.publishTrack(track, {
        source:
          track.kind === Track.Kind.Video
            ? Track.Source.ScreenShare
            : Track.Source.ScreenShareAudio,
      });
      if (track.kind === Track.Kind.Video) {
        track.attach(els.localScreen);
        track.mediaStreamTrack.addEventListener("ended", () => {
          log("Browser stopped the screen share");
          stopScreenShare();
        });
      }
    }
    log("Screen share published as SCREEN_SHARE", "ok");
    els.stopShareBtn.disabled = false;
  } catch (e) {
    log(`Screen share failed: ${e.message || e}`, "err");
    els.shareBtn.disabled = false;
  }
}

async function stopScreenShare() {
  if (!room) return;
  for (const track of localScreenTracks) {
    try {
      await room.localParticipant.unpublishTrack(track, true);
    } catch (_) {}
  }
  localScreenTracks = [];
  els.localScreen.srcObject = null;
  els.shareBtn.disabled = false;
  els.stopShareBtn.disabled = true;
  log("Screen share stopped");
}

function toggleMute() {
  if (!localAudioTrack) return;
  if (localAudioTrack.isMuted) {
    localAudioTrack.unmute();
    els.muteBtn.textContent = "Mute mic";
    log("Mic unmuted");
  } else {
    localAudioTrack.mute();
    els.muteBtn.textContent = "Unmute mic";
    log("Mic muted");
  }
}

function attachVuMeter(mediaStreamTrack) {
  try {
    if (vuRaf) cancelAnimationFrame(vuRaf);
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(new MediaStream([mediaStreamTrack]));
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const loop = () => {
      analyser.getByteFrequencyData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i];
      const pct = Math.min(100, (sum / data.length / 128) * 120);
      els.vuBar.style.width = `${pct}%`;
      vuRaf = requestAnimationFrame(loop);
    };
    loop();
  } catch (e) {
    log(`VU meter error: ${e.message}`, "err");
  }
}

async function leave() {
  if (room) await room.disconnect();
  cleanup();
}

function cleanup() {
  if (vuRaf) cancelAnimationFrame(vuRaf);
  vuRaf = null;
  els.vuBar.style.width = "0%";
  els.agentMeta.textContent = "Waiting for Solace to join…";
  els.setup.hidden = false;
  els.stage.hidden = true;
  els.joinBtn.disabled = false;
  els.leaveBtn.disabled = true;
  els.shareBtn.disabled = false;
  els.stopShareBtn.disabled = true;
  els.localScreen.srcObject = null;
  localScreenTracks = [];
  localAudioTrack = null;
  room = null;
  setStatus("Disconnected");
}

els.joinBtn.addEventListener("click", join);
els.leaveBtn.addEventListener("click", leave);
els.shareBtn.addEventListener("click", startScreenShare);
els.stopShareBtn.addEventListener("click", stopScreenShare);
els.muteBtn.addEventListener("click", toggleMute);

log("Ready. Click Join room.", "info");
