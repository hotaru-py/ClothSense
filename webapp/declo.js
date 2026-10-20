// DeClotH demo page: photo -> job on the GPU (webapp/serve.py) -> streamed cloth/body meshes -> clothing options.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GuidedCamera } from "./capture.js";
import { submitJob, getJob, jobFile } from "./api.js";

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ viewer
class DecloViewer {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.05, 50);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x777780, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 0.8); key.position.set(1, 2, 2); this.scene.add(key);
    this.loader = new GLTFLoader();
    this.meshes = {};         // name -> THREE.Object3D on screen ("cloth", "body", "tryon")
    this.mode = "both";
    this.framed = false;
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    this.renderer.setAnimationLoop(() => { this.controls.update(); this.renderer.render(this.scene, this.camera); });
  }

  resize() {
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1;
    if (w === this.w && h === this.h) return;
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  async load(url) {
    const gltf = await this.loader.loadAsync(url);
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
      // unlit: the vertex colours already hold the photo's shading (textured) or the surface normals (snapshots);
      // lighting them again exaggerated the DMTet surface noise
      o.material = new THREE.MeshBasicMaterial({ vertexColors: !!o.geometry.attributes.color, side: THREE.DoubleSide });
    });
    return gltf.scene;
  }

  // Replace the meshes named in `urls` ({cloth: url, body: url} or {tryon: url}) once all have loaded.
  async set(urls) {
    const loaded = {};
    for (const [k, u] of Object.entries(urls)) loaded[k] = await this.load(u);
    for (const [k, obj] of Object.entries(loaded)) {
      if (this.meshes[k]) this.scene.remove(this.meshes[k]);
      this.meshes[k] = obj; this.scene.add(obj);
    }
    if (!this.framed) this.frame();
    this.apply();
  }

  clear(name) { if (this.meshes[name]) { this.scene.remove(this.meshes[name]); delete this.meshes[name]; } }

  frame() {
    const box = new THREE.Box3();
    Object.values(this.meshes).forEach((m) => box.expandByObject(m));
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3()), h = box.max.y - box.min.y;
    const d = 1.2 * h / Math.tan((this.camera.fov * Math.PI) / 360) / 2;
    this.controls.target.copy(c);
    this.camera.position.set(c.x + 0.25 * d, c.y + 0.05 * h, c.z + d);   // the person faces +z
    this.controls.update();
    this.framed = true;
  }

  show(mode) { this.mode = mode; this.apply(); }
  apply() {
    const tryon = !!this.meshes.tryon;
    if (this.meshes.cloth) this.meshes.cloth.visible = !tryon && this.mode !== "body";
    if (this.meshes.body) this.meshes.body.visible = !tryon && this.mode !== "cloth";
    if (tryon) {             // try-on GLB = [garment, body]
      const parts = []; this.meshes.tryon.traverse((o) => o.isMesh && parts.push(o));
      if (parts[0]) parts[0].visible = this.mode !== "body";
      if (parts[1]) parts[1].visible = this.mode !== "cloth";
    }
  }
}

// ------------------------------------------------------------------ flow
let viewer = null, cam = null, jobId = null, shownVersion = null, polling = null;

function showError(msg) { $("err").textContent = msg; }

async function start(photo) {
  showError("");
  try {
    jobId = await submitJob(photo);
  } catch (e) {
    showError(`Could not start: ${e.message}`);
    return;
  }
  history.replaceState(null, "", `?job=${jobId}`);   // a reload re-attaches to the running job
  attach();
}

function attach() {
  $("home").classList.add("hidden");
  $("job").classList.remove("hidden");
  $("opts").innerHTML = ""; $("optHint").classList.remove("hidden");
  if (!viewer) {
    try { viewer = new DecloViewer($("viewer")); } catch { $("vmsg").textContent = "No WebGL in this browser."; }
  }
  shownVersion = null;
  poll();
}

async function poll() {
  clearTimeout(polling);
  let st;
  try { st = await getJob(jobId); } catch (e) { $("stage").textContent = e.message; return; }
  $("stage").textContent = st.status === "error" ? `Failed: ${st.error}` : st.stage;
  $("barfill").style.width = `${Math.round(100 * (st.progress || 0))}%`;
  const bits = [];
  if (st.captions) bits.push(`Seen: ${st.captions.cloth}`);
  if (st.total) bits.push(`step ${st.step}/${st.total}`);
  if (st.total_s) bits.push(`${Math.round(st.total_s / 60)} min`);
  $("meta").textContent = bits.join(" · ");
  if (st.prep_s && !$("photo").src) $("photo").src = jobFile(jobId, "photo.png");
  const v = st.status === "done" ? "final" : st.version;
  if (viewer && v !== null && v !== undefined && v !== shownVersion) {
    const pre = v === "final" ? "final_" : `v${v}_`;
    try {
      await viewer.set({ cloth: jobFile(jobId, `${pre}cloth.glb`), body: jobFile(jobId, `${pre}body.glb`) });
      shownVersion = v;
      $("vmsg").textContent = v === "final" ? "Textured result" : v === 0 ? "Templates (start)" : `After ${v} steps`;
    } catch { /* not written yet: next poll */ }
  }
  if (st.status === "done") return loadOptions();
  if (st.status !== "error") polling = setTimeout(poll, 3000);
}

async function loadOptions() {
  const lib = (await (await fetch("api/library")).json()).library || [];
  const others = lib.filter((x) => x.id !== jobId);
  $("optHint").classList.add("hidden");
  const box = $("opts");
  box.innerHTML = "";
  const select = (btn) => box.querySelectorAll(".opt").forEach((x) => x.setAttribute("aria-pressed", String(x === btn)));
  const card = (thumb, title, sub) => {
    const b = document.createElement("button");
    b.className = "opt";
    const img = document.createElement("img");
    img.alt = ""; img.loading = "lazy"; img.src = thumb;
    img.onerror = () => { const d = document.createElement("div"); d.className = "noimg"; d.textContent = "no preview"; img.replaceWith(d); };
    const t = document.createElement("span"); t.textContent = title;
    const s = document.createElement("small"); s.textContent = sub;
    b.append(img, t, s); b.title = title;
    return b;
  };
  const own = card(`api/library/${jobId}.png`, "Your own clothes", "as reconstructed");
  own.setAttribute("aria-pressed", "true");
  own.onclick = () => { select(own); viewer.clear("tryon"); viewer.apply(); $("vmsg").textContent = "Your clothes"; };
  box.append(own);
  if (!others.length) {
    const p = document.createElement("small"); p.textContent = "Reconstruct another person to swap garments."; box.append(p);
  }
  for (const it of others) {
    const b = card(`api/library/${it.id}.png`, it.caption, it.shipped ? "example garment" : `from reconstruction ${it.id.slice(0, 6)}`);
    b.onclick = async () => {
      select(b);
      $("vmsg").textContent = "Fitting the garment…";
      try {
        await viewer.set({ tryon: jobFile(jobId, `tryon_${it.id}.glb`) });
        $("vmsg").textContent = it.caption;
      } catch (e) { $("vmsg").textContent = `Try-on failed: ${e.message}`; }
    };
    box.append(b);
  }
}

// ------------------------------------------------------------------ inputs
$("file").onchange = (e) => { const f = e.target.files[0]; if (f) start(f); };

$("btnCam").onclick = async () => {
  $("cam").classList.remove("hidden");
  cam = cam || new GuidedCamera({
    canvas: $("canvas"), video: $("video"),
    onState: (s) => { if (s.msg) $("cammsg").textContent = s.msg; },
    onCapture: (blob) => { cam.stop(); $("cam").classList.add("hidden"); start(blob); },
  });
  try { await cam.start(); } catch (e) { $("cammsg").textContent = e.message; }
};
$("btnShot").onclick = () => cam?.running && cam.shoot(false);
$("btnFlip").onclick = () => cam && cam.start(cam.facing === "user" ? "environment" : "user");
$("btnCamCancel").onclick = () => { cam && cam.stop(); $("cam").classList.add("hidden"); };
$("btnNew").onclick = () => {
  clearTimeout(polling);
  history.replaceState(null, "", location.pathname);
  $("job").classList.add("hidden"); $("home").classList.remove("hidden");
  if (viewer) { ["cloth", "body", "tryon"].forEach((k) => viewer.clear(k)); viewer.framed = false; }
  $("photo").removeAttribute("src");
};
document.querySelectorAll("#showBtns button").forEach((b) => b.onclick = () => {
  document.querySelectorAll("#showBtns button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  viewer && viewer.show(b.dataset.show);
});

// re-attach to a job given in the URL (?job=<id>)
const resume = new URLSearchParams(location.search).get("job");
if (resume) { jobId = resume; attach(); }
