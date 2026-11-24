"""Local live viewer for the teacher run: http://127.0.0.1:8766  (newest pages first, auto refresh)

python data/teacher/watch.py
"""

import glob
import html
import json
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BW = os.environ.get("WATCH_BW", os.path.join(ROOT, "data/enhanced/mangadex_bw"))
PAGES = os.path.join(BW, "pages")
TEACH = os.environ.get("WATCH_TEACH", os.path.join(ROOT, "data/enhanced/teacher_klein4b"))
MANIFEST = os.path.join(BW, "manifest.jsonl")
TRAIN = os.environ.get("WATCH_TRAIN", "")  # training output dir (train.log, samples/)
PAIRS = os.environ.get("WATCH_PAIRS", "")
PORT = int(os.environ.get("WATCH_PORT", "8766"))
N = 24


def titles():
    out = {}
    try:
        for line in open(MANIFEST, encoding="utf-8"):
            d = json.loads(line)
            out[os.path.splitext(d["file"])[0]] = (d["title"], d["chapter_no"], d["ref"])
    except OSError:
        pass
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/img":
            kind, name = q["k"][0], os.path.basename(q["n"][0])
            if kind == "bw":
                hits = glob.glob(os.path.join(PAGES, glob.escape(name) + ".*"))
                path = hits[0] if hits else ""
            elif kind == "sample":
                files = sorted(glob.glob(os.path.join(TRAIN, "samples", "*.jpg")))
                path = files[-1] if files else ""
            elif kind == "ref":
                path = os.path.join(os.path.dirname(PAGES), "refs", name)
            else:
                path = os.path.join(TEACH, name + ".png")
            if not os.path.isfile(path):
                return self.send(b"", "text/plain", 404)
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            return self.send(open(path, "rb").read(), f"image/{'jpeg' if ext == 'jpg' else ext}")
        done = sorted(glob.glob(os.path.join(TEACH, "*.png")), key=os.path.getmtime, reverse=True)
        n_pages = len(os.listdir(PAGES)) if os.path.isdir(PAGES) else 0
        rate = ""
        if len(done) > 10:
            dt = os.path.getmtime(done[0]) - os.path.getmtime(done[10])
            if dt > 0:
                spp = dt / 10
                rate = f" · {spp:.1f} s/page · ~{(n_pages - len(done)) * spp / 3600:.1f} h left for the fetched pages"
        train = ""
        if TRAIN and os.path.exists(os.path.join(TRAIN, "train.log")):
            lines = [
                l
                for l in open(os.path.join(TRAIN, "train.log"), encoding="utf-8", errors="replace")
                if l.startswith(("step", "  held-out", "refreshed", "done", "TRAIN_EXIT", "Traceback"))
            ]
            sample = glob.glob(os.path.join(TRAIN, "samples", "*.jpg"))
            train = (
                "<h2>Training (stage 3)</h2><pre>"
                + html.escape("".join(lines[-12:]))
                + "</pre>"
                + (
                    f'<div class="sub">latest sample: {html.escape(os.path.basename(sorted(sample)[-1]))} '
                    "(row 1: no reference, row 2: with cover, row 3: target)</div>"
                    f'<img class="sample" src="/img?k=sample&n=x&t={len(sample)}">'
                    if sample
                    else ""
                )
            )
        n_pairs = (
            len(os.listdir(os.path.join(PAIRS, "color")))
            if PAIRS and os.path.isdir(os.path.join(PAIRS, "color"))
            else None
        )
        t = titles()
        cards = []
        for f in done[:N]:
            n = os.path.splitext(os.path.basename(f))[0]
            title, ch, ref = t.get(n, ("?", "?", ""))
            e = urllib.parse.quote(n)
            cards.append(f"""<div class="card"><div class="cap">{html.escape(title)} · ch. {html.escape(str(ch))}
              · {time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(f)))}</div>
              <div class="pair"><img src="/img?k=bw&n={e}" loading="lazy"><img src="/img?k=teach&n={e}" loading="lazy">
              <img class="ref" src="/img?k=ref&n={urllib.parse.quote(ref)}" loading="lazy" title="color reference"></div></div>""")
        page = f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Teacher run</title><style>
:root{{--bg:#111;--fg:#eee;--mut:#999;--card:#1c1c1c}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}}
h1{{font-size:18px;margin:0 0 4px}} .sub{{color:var(--mut);margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:16px}}
.card{{background:var(--card);border-radius:8px;padding:8px}} .cap{{color:var(--mut);margin-bottom:6px}}
.pair{{display:flex;gap:6px;align-items:flex-start}} .pair img{{width:42%;height:auto;border-radius:4px}}
.pair img.ref{{width:14%}} h2{{font-size:16px;margin:16px 0 6px}} pre{{background:var(--card);padding:8px;border-radius:6px;white-space:pre-wrap;font-size:12px}}
img.sample{{width:100%;height:auto;border-radius:6px;margin-bottom:12px}} @media(max-width:600px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><h1>Teacher run: klein 4B</h1>
{train}<h2>Teacher</h2><div class="sub">{len(done)} colored / {n_pages} black-and-white pages fetched{rate}{f" · {n_pairs} training pairs" if n_pairs is not None else ""} · left: original, middle: teacher, right: color reference · refreshes every 10 s</div>
<div class="grid">{"".join(cards)}</div></body></html>"""
        self.send(page.encode(), "text/html; charset=utf-8")


if __name__ == "__main__":
    print(f"http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
