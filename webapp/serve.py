"""Serve the DeClotH demo: photo -> separate 3D cloth + body (~15 min on an RTX 3070 Ti Laptop) -> clothing options.

    python webapp/serve.py                  # http://localhost:8000  (loads the job service; GPU needed)
    python webapp/serve.py --https          # https://<LAN-IP>:8443  (phones: the camera needs https)
    python webapp/serve.py --static         # page only, no models (UI work; jobs are refused)

--https makes a self-signed certificate with openssl (ships with Git for Windows) the first time.
The phone will warn about the certificate once; accept it to continue.

API (same origin as the page, so no CORS):
    GET  /api/health                         -> {"declo": bool}
    POST /api/jobs?hfov=auto                 body = the photo (JPEG/PNG bytes, EXIF kept) -> {"id": ...}
    GET  /api/jobs/<id>                      -> status, stage, progress, step/total, latest snapshot version
    GET  /api/jobs/<id>/<file>               -> photo.png, v<k>_cloth.glb, v<k>_body.glb (snapshots), final_*.glb
    GET  /api/jobs/<id>/tryon_<other>.glb    -> another reconstruction's garment on this person (computed on request)
    GET  /api/library                        -> clothing options: shipped garments/ + finished reconstructions
    GET  /api/library/<id>.png               -> garment thumbnail
"""
import argparse, functools, http.server, json, shutil, socket, ssl, subprocess, sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CERT, KEY = ROOT / "work/webapp-cert.pem", ROOT / "work/webapp-key.pem"
MAX_UPLOAD = 25 * 2**20
CONTENT_TYPES = {".glb": "model/gltf-binary", ".png": "image/png", ".jpg": "image/jpeg"}


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1)); return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def make_cert(ip):
    openssl = shutil.which("openssl") or r"C:\Program Files\Git\usr\bin\openssl.exe"
    CERT.parent.mkdir(exist_ok=True)
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
                    "-keyout", str(KEY), "-out", str(CERT), "-subj", "/CN=declo-local",
                    "-addext", f"subjectAltName=IP:{ip},DNS:localhost"], check=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    service = None

    def end_headers(self):
        # dev server: make browsers revalidate every file, so an edited page is never served stale
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, default=float).encode(), "application/json")

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if 0 < n <= MAX_UPLOAD else None

    def do_GET(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        if parts[:2] == ["api", "health"]:
            return self._json(200, {"declo": self.service is not None})
        if parts[:2] == ["api", "library"]:
            if len(parts) == 3 and parts[2].endswith(".png"):           # /api/library/<id>.png = garment thumbnail
                if self.service is None:
                    return self._json(503, {"error": "models not loaded"})
                try:
                    return self._send(200, self.service.thumbnail(parts[2][:-4]), "image/png")
                except FileNotFoundError:
                    return self._json(404, {"error": "unknown library item"})
            return self._json(200, {"library": self.service.library() if self.service else []})
        if parts[:2] == ["api", "jobs"] and len(parts) >= 3:
            job = self.service.jobs.get(parts[2]) if self.service else None
            if job is None:
                return self._json(404, {"error": "unknown job"})
            if len(parts) == 3:
                return self._json(200, job.state)
            data = job.files.get(parts[3])
            if data is None and parts[3].startswith("tryon_"):
                if job.state.get("status") != "done":
                    return self._json(409, {"error": "reconstruction not finished"})
                try:
                    data = job.files[parts[3]] = self.service.tryon(job.id, parts[3][6:].removesuffix(".glb"))
                except FileNotFoundError:
                    return self._json(404, {"error": "unknown library item"})
            if data is None:
                return self._json(404, {"error": "not ready"})
            return self._send(200, data, CONTENT_TYPES.get(Path(parts[3]).suffix, "application/octet-stream"))
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/jobs":
            return self._json(404, {"error": "not found"})
        if self.service is None:
            return self._json(503, {"error": "models not loaded (server started with --static)"})
        data = self._body()
        if data is None:
            return self._json(413, {"error": "photo missing or larger than 25 MB"})
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        hfov = q.get("hfov") or "auto"
        try:
            hfov = hfov if hfov == "auto" else float(hfov)
        except ValueError:
            return self._json(400, {"error": "bad hfov"})
        return self._json(200, {"id": self.service.submit(data, hfov).id})

    def log_message(self, fmt, *args):            # keep the console readable: no per-poll lines
        parts = urlparse(self.path).path.strip("/").split("/")
        if not (parts[:2] == ["api", "jobs"] and len(parts) == 3 and self.command == "GET"):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--https", action="store_true")
    ap.add_argument("--static", action="store_true", help="serve the page only (no models, no jobs)")
    ap.add_argument("--declo", action="store_true", help=argparse.SUPPRESS)   # old flag: the demo is the default now
    ap.add_argument("--port", type=int)
    ap.add_argument("--geo-steps", type=int, default=150)
    ap.add_argument("--tex-steps", type=int, default=200)
    a = ap.parse_args()
    # Windows' registry often maps .js to text/plain, which browsers refuse for ES modules
    Handler.extensions_map.update({".js": "text/javascript", ".mjs": "text/javascript",
                                   ".json": "application/json", ".wasm": "application/wasm"})
    if not a.static:
        sys.path.insert(0, str(HERE))
        from declo_service import DeclothService
        Handler.service = DeclothService(geo_steps=a.geo_steps, tex_steps=a.tex_steps)
    handler = functools.partial(Handler, directory=str(HERE))
    ip = lan_ip()
    port = a.port or (8443 if a.https else 8000)
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    if a.https:
        if not CERT.exists():
            make_cert(ip)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(CERT, KEY)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print(f"serving https://{ip}:{port}  (phone on the same Wi-Fi)  and https://localhost:{port}", flush=True)
    else:
        print(f"serving http://localhost:{port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
