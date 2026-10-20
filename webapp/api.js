// Client for the reconstruction jobs served by `webapp/serve.py` (same origin).

// photo: the ORIGINAL photo (uploads keep EXIF -> real focal length). hfov: number or "auto".
export async function submitJob(photo, { hfov = "auto" } = {}) {
  const q = new URLSearchParams({ hfov: String(hfov) });
  const r = await fetch(`api/jobs?${q}`, { method: "POST", body: photo,
    headers: { "Content-Type": photo.type || "image/jpeg" } });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `server error ${r.status}`);
  return j.id;
}

export async function getJob(id) {
  const r = await fetch(`api/jobs/${id}`, { cache: "no-store" });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || `server error ${r.status}`);
  return j;
}

export const jobFile = (id, name) => `api/jobs/${id}/${name}`;
