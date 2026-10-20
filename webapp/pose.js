// One MediaPipe Pose Landmarker per page, used by the guided camera.
import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
// Google's hosted models (byte-identical to the copies that used to sit in webapp/models/)
const MODEL = (m) => `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${m}/float16/1/pose_landmarker_${m}.task`;
const cache = {};

export const LM = { nose: 0, lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16, lHip: 23, rHip: 24,
  lKnee: 25, rKnee: 26, lAnk: 27, rAnk: 28, lHeel: 29, rHeel: 30 };
export const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [24, 26], [26, 28], [28, 30]];

// -> { landmarker, delegate, model }, VIDEO running mode
export function getLandmarker(model = "lite") {
  if (!cache[model]) cache[model] = create(model);
  return cache[model];
}

async function create(model) {
  const vision = await FilesetResolver.forVisionTasks(WASM);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL(model), delegate },
    runningMode: "VIDEO", numPoses: 1,
  });
  try {
    return { landmarker: await PoseLandmarker.createFromOptions(vision, opts("GPU")), delegate: "GPU", model };
  } catch (e) {
    console.warn("GPU delegate unavailable, using CPU", e);
    return { landmarker: await PoseLandmarker.createFromOptions(vision, opts("CPU")), delegate: "CPU", model };
  }
}
