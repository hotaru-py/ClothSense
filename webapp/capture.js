// Guided full-body capture: live pose checks, an on-screen guide, and an auto-capture countdown.
// Frames from getUserMedia carry no EXIF, so the server estimates the camera FOV (auto-FOV).
import { getLandmarker, LM, BONES } from "./pose.js";

const HOLD_MS = 700;            // all checks must pass this long before the countdown starts
const COUNTDOWN_MS = 3000;
const STILL_WINDOW_MS = 500;    // "hold still": max landmark travel over this window
const STILL_MAX = 0.02;         // ... as a fraction of the frame height
const ARM_DEG = [12, 45];       // shoulder->wrist angle away from straight down
const SPAN = [0.5, 0.88];       // nose-to-feet height as a fraction of the frame
const HEAD_LINE = 0.05, FEET_LINE = 0.95;

const vis = (q) => q && q.visibility > 0.5;

// p: normalised landmarks; aspect = width / height. Returns { ok, msg }.
export function checkPose(p, aspect) {
  if (!p) return { ok: false, msg: "Step into the frame, whole body visible" };
  const X = (q) => q.x * aspect;                                 // x in frame-height units
  const nose = p[LM.nose];
  const feet = [p[LM.lAnk], p[LM.rAnk], p[LM.lHeel], p[LM.rHeel]];
  const footY = Math.max(...feet.map((q) => q.y));
  if (!vis(nose) || nose.y < 0.02) return { ok: false, msg: "Step back: your head is cut off" };
  if (!(vis(p[LM.lAnk]) || vis(p[LM.rAnk])) || footY > 0.985) return { ok: false, msg: "Step back: show your feet" };
  const span = footY - nose.y;
  if (span > SPAN[1]) return { ok: false, msg: "Step back a little" };
  if (span < SPAN[0]) return { ok: false, msg: "Come a bit closer" };
  const hipX = (p[LM.lHip].x + p[LM.rHip].x) / 2;
  if (hipX < 0.3 || hipX > 0.7) return { ok: false, msg: "Move to the middle of the frame" };
  const shW = Math.hypot(X(p[LM.lSh]) - X(p[LM.rSh]), p[LM.lSh].y - p[LM.rSh].y);
  const torso = Math.abs((p[LM.lHip].y + p[LM.rHip].y) / 2 - (p[LM.lSh].y + p[LM.rSh].y) / 2);
  if (torso <= 0 || shW / torso < 0.45) return { ok: false, msg: "Turn to face the camera" };
  if (!vis(p[LM.lWr]) || !vis(p[LM.rWr])) return { ok: false, msg: "Keep both hands in view" };
  const armDeg = (sh, wr) => (Math.atan2(Math.abs(X(wr) - X(sh)), wr.y - sh.y) * 180) / Math.PI;
  const a = [armDeg(p[LM.lSh], p[LM.lWr]), armDeg(p[LM.rSh], p[LM.rWr])];
  if (Math.min(...a) < ARM_DEG[0]) return { ok: false, msg: "Move your arms slightly away from your body" };
  if (Math.max(...a) > ARM_DEG[1]) return { ok: false, msg: "Lower your arms a little" };
  return { ok: true, msg: "Perfect, hold still" };
}

export class GuidedCamera {
  constructor({ canvas, video, onState, onCapture }) {
    this.canvas = canvas; this.ctx = canvas.getContext("2d"); this.video = video;
    this.onState = onState || (() => {}); this.onCapture = onCapture || (() => {});
    this.facing = "user"; this.running = false; this.auto = true;
  }

  async start(facing = this.facing) {
    this.stop();
    this.facing = facing;
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("The camera needs a secure page (https:// or localhost).");
    this.onState({ msg: "Starting the camera…" });
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.canvas.width = this.video.videoWidth; this.canvas.height = this.video.videoHeight;
    this.mirror = facing === "user";
    this.onState({ msg: "Loading pose model…" });
    this.landmarker = (await getLandmarker()).landmarker;
    this.running = true; this.okSince = null; this.countdownEnd = null; this.history = []; this.lastTime = -1;
    requestAnimationFrame(() => this.loop());
  }

  stop() {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  loop() {
    if (!this.running) return;
    const v = this.video;
    if (v.readyState >= 2 && v.currentTime !== this.lastTime) {
      this.lastTime = v.currentTime;
      const now = performance.now();
      const p = this.landmarker.detectForVideo(v, now).landmarks?.[0] || null;
      let st = checkPose(p, v.videoWidth / v.videoHeight);
      if (st.ok && !this.still(p, now)) st = { ok: false, msg: "Hold still" };
      if (!st.ok) { this.okSince = null; this.countdownEnd = null; }
      else if (this.okSince === null) this.okSince = now;
      if (this.auto && st.ok && this.countdownEnd === null && now - this.okSince >= HOLD_MS) this.countdownEnd = now + COUNTDOWN_MS;
      const count = this.countdownEnd ? Math.max(0, Math.ceil((this.countdownEnd - now) / 1000)) : null;
      this.draw(p, st.ok);
      this.onState({ ...st, countdown: count });
      if (this.countdownEnd && now >= this.countdownEnd) { this.shoot(true); return; }
    }
    requestAnimationFrame(() => this.loop());
  }

  still(p, now) {
    const keys = [LM.nose, LM.lSh, LM.rSh, LM.lHip, LM.rHip, LM.lAnk, LM.rAnk].map((i) => [p[i].x, p[i].y]);
    this.history.push([now, keys]);
    while (this.history.length && now - this.history[0][0] > STILL_WINDOW_MS) this.history.shift();
    const [, first] = this.history[0];
    return keys.every(([x, y], i) => Math.hypot(x - first[i][0], y - first[i][1]) < STILL_MAX);
  }

  draw(p, ok) {
    const { ctx, canvas: c } = this, W = c.width, H = c.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (this.mirror) ctx.setTransform(-1, 0, 0, 1, W, 0);
    ctx.drawImage(this.video, 0, 0, W, H);
    // guide: head and feet lines, centre column
    ctx.save();
    ctx.lineWidth = Math.max(2, H / 300); ctx.setLineDash([H / 60, H / 80]);
    ctx.strokeStyle = "rgba(255,255,255,.7)";
    for (const y of [HEAD_LINE, FEET_LINE]) { ctx.beginPath(); ctx.moveTo(W * 0.25, y * H); ctx.lineTo(W * 0.75, y * H); ctx.stroke(); }
    ctx.setLineDash([]); ctx.fillStyle = "rgba(255,255,255,.06)"; ctx.fillRect(W * 0.3, HEAD_LINE * H, W * 0.4, (FEET_LINE - HEAD_LINE) * H);
    ctx.restore();
    if (!p) return;
    ctx.lineWidth = Math.max(3, H / 180); ctx.lineCap = "round";
    ctx.strokeStyle = ok ? "rgba(80,220,130,.95)" : "rgba(255,190,60,.95)";
    for (const [a, b] of BONES) {
      if (!vis(p[a]) || !vis(p[b])) continue;
      ctx.beginPath(); ctx.moveTo(p[a].x * W, p[a].y * H); ctx.lineTo(p[b].x * W, p[b].y * H); ctx.stroke();
    }
  }

  // Grab the current frame, un-mirrored (the pipeline needs the real image), as a JPEG blob.
  async shoot(auto = false) {
    const v = this.video;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.95));
    const meta = { source: "guided", facing: this.facing, width: c.width, height: c.height, auto };
    this.stop();
    navigator.vibrate?.(40);
    this.onCapture(blob, meta);
  }
}
