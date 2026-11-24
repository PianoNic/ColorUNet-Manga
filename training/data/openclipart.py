"""Openclipart (CC0 1.0) -> exact line-art / flat-color training pairs.

  python data/openclipart.py fetch --ids 342531 343797 ...        # SVG + title/tags into data/raw_color/openclipart
  python data/openclipart.py fetch --ids-file ids.txt
  python data/openclipart.py pairs --out data/pairs               # filter, render, write pairs

fetch: uses the public detail page (license, title, tags) and /download/<id>/<file>.svg, politely.
  Bulk ID lists need the Openclipart API (account + app token: https://openclipart.org/api/tutorial).
  Every file's id, page, SVG url, license, title and tags go into manifest.jsonl.
pairs: drops anything whose title/tags name a known copyrighted character, franchise, brand or logo
  (BLOCKLIST, plus anything not CC0), then renders each SVG twice:
  - color/<name>.png: the clipart as drawn, on white
  - gray/<name>__lineart.png: the same SVG rewritten so dark fills/strokes are black ink and every
    other fill is white paper -> exact line art, pixel-aligned with the color target
  plus gray/<name>__clean.png (luma of the color render) and __tone.png (synthetic screentone).
"""

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.make_pairs import screentone, to_gray  # noqa: E402

SITE = "https://openclipart.org"
UA = {"User-Agent": "manga-colors-dataset/0.1 (CC0 clipart for colorization research)"}
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
for _p, _u in [
    ("", "http://www.w3.org/2000/svg"),
    ("xlink", "http://www.w3.org/1999/xlink"),
    ("inkscape", "http://www.inkscape.org/namespaces/inkscape"),
    ("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"),
]:
    ET.register_namespace(_p, _u)  # keep plain <svg> tags when rewriting

# Known copyrighted characters, franchises, brands and logos (title/tag match, case-insensitive).
BLOCKLIST = re.compile(
    r"\b("
    + "|".join(
        [
            "logo",
            "logos",
            "brand",
            "trademark",
            "emblem",
            "mascot",
            "disney",
            "mickey",
            "minnie",
            "donald duck",
            "goofy",
            "pixar",
            "frozen",
            "elsa",
            "simba",
            "marvel",
            "avengers",
            "spider-?man",
            "iron ?man",
            "hulk",
            "thor",
            "x-men",
            "deadpool",
            "wolverine",
            "dc comics",
            "batman",
            "superman",
            "wonder woman",
            "joker",
            "harley quinn",
            "flash",
            "pokemon",
            "pok[eé]mon",
            "pikachu",
            "nintendo",
            "mario",
            "luigi",
            "zelda",
            "link",
            "kirby",
            "yoshi",
            "sonic",
            "sega",
            "pac-?man",
            "tetris",
            "minecraft",
            "fortnite",
            "roblox",
            "among us",
            "naruto",
            "sasuke",
            "goku",
            "dragon ?ball",
            "one piece",
            "luffy",
            "sailor moon",
            "totoro",
            "ghibli",
            "doraemon",
            "hello kitty",
            "sanrio",
            "evangelion",
            "gundam",
            "attack on titan",
            "demon slayer",
            "my little pony",
            "simpsons",
            "homer simpson",
            "bart",
            "futurama",
            "spongebob",
            "garfield",
            "snoopy",
            "peanuts",
            "looney tunes",
            "bugs bunny",
            "tom and jerry",
            "scooby",
            "smurf",
            "tintin",
            "asterix",
            "star wars",
            "darth vader",
            "yoda",
            "stormtrooper",
            "star trek",
            "harry potter",
            "hogwarts",
            "lord of the rings",
            "gandalf",
            "shrek",
            "minion",
            "teletubbies",
            "barbie",
            "lego",
            "coca-?cola",
            "pepsi",
            "mcdonald",
            "nike",
            "adidas",
            "apple inc",
            "iphone",
            "google",
            "facebook",
            "twitter",
            "youtube",
            "windows",
            "microsoft",
            "android",
            "tux",
            "firefox",
            "chrome",
            "ubuntu",
            "fan ?art",
            "fanart",
            "parody",
            "cosplay",
            "celebrity",
            "president",
            "trump",
            "obama",
            # AI-generated uploads: unknown provenance, same problem as scraped data
            "ai",
            "ai[_ -]?art",
            "ai[_ -]?generated",
            "ai[_ -]?image",
            "generative",
            "midjourney",
            "stable ?diffusion",
            "dall-?e",
            "chatgpt",
            "gpt",
            "firefly",
            "craiyon",
            "novelai",
        ]
    )
    + r")\b",
    re.I,
)


# ------------------------------------------------------------------ fetch


def get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def fetch(ids, out):
    os.makedirs(os.path.join(out, "svg"), exist_ok=True)
    manifest = os.path.join(out, "manifest.jsonl")
    done = {json.loads(l)["id"] for l in open(manifest, encoding="utf-8")} if os.path.exists(manifest) else set()
    with open(manifest, "a", encoding="utf-8") as mf:
        for cid in ids:
            if cid in done:
                continue
            page = get(f"{SITE}/detail/{cid}").decode("utf-8", "replace")
            svg_url = re.search(r'"url": ?"(https:\\/\\/openclipart\.org\\/download\\/[^"]+\.svg)"', page)
            lic = re.search(r'"license": ?"([^"]+)"', page)
            title = re.search(r"<title>(.*?)(?: - Openclipart)?</title>", page, re.S)
            if not svg_url:
                print(f"{cid}: no SVG link, skipped")
                continue
            svg_url = svg_url.group(1).replace("\\/", "/")
            rec = {
                "id": cid,
                "page": f"{SITE}/detail/{cid}",
                "svg_url": svg_url,
                "license": lic.group(1).replace("\\/", "/") if lic else "",
                "title": html.unescape(title.group(1).strip()) if title else "",
                "tags": [html.unescape(t) for t in re.findall(r'href="/tag/[^"]*"[^>]*>([^<]+)<', page)],
                "file": f"svg/{cid}.svg",
            }
            with open(os.path.join(out, rec["file"]), "wb") as f:
                f.write(get(svg_url))
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{cid}: {rec['title']}  tags={rec['tags'][:6]}")
            time.sleep(1.0)


# ------------------------------------------------------------------ pairs


def _lum(color):
    """CSS color -> luminance 0-1, or None for none/url()/unknown."""
    if not color:
        return None
    c = color.strip().lower()
    if c in ("none", "transparent") or c.startswith("url("):
        return None
    named = {"black": (0, 0, 0), "white": (255, 255, 255)}
    if c in named:
        r, g, b = named[c]
    elif re.fullmatch(r"#[0-9a-f]{6}", c):
        r, g, b = (int(c[i : i + 2], 16) for i in (1, 3, 5))
    elif re.fullmatch(r"#[0-9a-f]{3}", c):
        r, g, b = (int(ch * 2, 16) for ch in c[1:])
    elif c.startswith("rgb("):
        r, g, b = (int(float(v.strip().rstrip("%"))) for v in c[4:-1].split(",")[:3])
    else:
        return 0.5  # other named colors: treat as a mid tone (fill -> white)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def lineart_svg(svg_text, ink=0.3):
    """Rewrite an SVG: dark fills/strokes -> black, everything else -> white (gradients included)."""

    def recolor(attr, value):
        lum = _lum(value)
        if value and value.strip().lower().startswith("url("):
            return "#ffffff" if attr == "fill" else "none"
        if lum is None:
            return value
        if attr == "fill":
            return "#000000" if lum < ink else "#ffffff"
        return "#000000" if lum < 0.6 else "#ffffff"

    root = ET.fromstring(svg_text)
    for el in root.iter():
        for attr in ("fill", "stroke"):
            if attr in el.attrib:
                el.attrib[attr] = recolor(attr, el.attrib[attr])
        if "style" in el.attrib:
            parts = []
            for decl in el.attrib["style"].split(";"):
                if ":" in decl:
                    k, v = (s.strip() for s in decl.split(":", 1))
                    if k in ("fill", "stroke"):
                        v = recolor(k, v)
                    elif k in ("filter", "opacity", "fill-opacity") and k != "fill-opacity":
                        v = "none" if k == "filter" else v
                    parts.append(f"{k}:{v}")
            el.attrib["style"] = ";".join(parts)
        if el.tag.endswith("stop"):  # gradient stops -> white
            el.attrib["style"] = "stop-color:#ffffff"
            el.attrib.pop("stop-color", None)
    return ET.tostring(root, encoding="unicode")


def fix_size(svg_text):
    """Give the root explicit pixel width/height from its viewBox when missing or relative (e.g. 100%)."""
    root = ET.fromstring(svg_text)
    vb = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    w, h = root.attrib.get("width", ""), root.attrib.get("height", "")
    num = lambda v: re.fullmatch(r"[\d.]+(px|pt|mm|cm|in)?", v.strip()) is not None
    if vb and not (num(w) and num(h)):
        _, _, vw, vh = (float(x) for x in re.split(r"[\s,]+", vb.strip())[:4])
        root.attrib["width"], root.attrib["height"] = str(vw), str(vh)
    return ET.tostring(root, encoding="unicode")


def render(svg_text, long_side):
    import resvg_py

    svg_text = fix_size(svg_text)
    probe = cv2.imdecode(
        np.frombuffer(bytes(resvg_py.svg_to_bytes(svg_string=svg_text, background="#ffffff", dpi=96)), np.uint8),
        cv2.IMREAD_COLOR,
    )
    h, w = probe.shape[:2]
    kw = {"width": long_side} if w >= h else {"height": long_side}
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg_text, background="#ffffff", dpi=96, **kw))
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def lineart_ok(line, gray, min_ink=0.005, min_cover=0.35):
    """Keep the line-art variant only if it has ink and covers enough of the drawing's edges.

    Clipart whose outlines are colored or gradient strokes loses them in the rewrite; such line art
    would teach the model to invent whole objects from nothing.
    """
    ink = line < 128
    if ink.mean() < min_ink:
        return False
    edges = cv2.Canny(gray, 50, 150) > 0
    if edges.sum() == 0:
        return False
    near_ink = cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return (edges & near_ink).sum() / edges.sum() >= min_cover


def blocked(rec):
    if rec.get("license", "").rstrip("/") != CC0.rstrip("/"):
        return "not CC0"
    text = " ".join([rec.get("title", "")] + rec.get("tags", []))
    m = BLOCKLIST.search(text)
    return f"blocklist: {m.group(0)}" if m else None


def pairs(src, out, long_side, seed):
    rng = np.random.default_rng(seed)
    for d in ("color", "gray"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(src, "manifest.jsonl"), encoding="utf-8")]
    kept, report = [], {}
    for rec in rows:
        why = blocked(rec)
        if why:
            report[why] = report.get(why, 0) + 1
            print(f"skip {rec['id']} ({rec['title']}): {why}")
            continue
        name = f"openclipart_{rec['id']}"
        try:
            svg = open(os.path.join(src, rec["file"]), encoding="utf-8").read()
            color = render(svg, long_side)
            line = render(lineart_svg(svg), long_side)
        except Exception as e:
            report["render error"] = report.get("render error", 0) + 1
            print(f"skip {rec['id']}: render error {e}")
            continue
        if line.shape != color.shape:
            line = cv2.resize(line, (color.shape[1], color.shape[0]))
        sat = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[..., 1]
        if (sat > 40).mean() < 0.02:
            report["no color"] = report.get("no color", 0) + 1
            continue
        clean = to_gray(color)
        cv2.imencode(".png", color)[1].tofile(os.path.join(out, "color", name + ".png"))
        if lineart_ok(to_gray(line), clean):
            cv2.imencode(".png", to_gray(line))[1].tofile(os.path.join(out, "gray", name + "__lineart.png"))
        else:
            report["line art dropped (colored/gradient outlines)"] = (
                report.get("line art dropped (colored/gradient outlines)", 0) + 1
            )
        cv2.imencode(".png", clean)[1].tofile(os.path.join(out, "gray", name + "__clean.png"))
        cv2.imencode(".png", screentone(clean, rng))[1].tofile(os.path.join(out, "gray", name + "__tone.png"))
        kept.append({k: rec[k] for k in ("id", "page", "svg_url", "license", "title")} | {"pair": name})
    with open(os.path.join(out, "manifest-openclipart.jsonl"), "w", encoding="utf-8") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"{len(kept)} kept, skipped: {report}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--ids", nargs="*", type=int, default=[])
    f.add_argument("--ids-file")
    f.add_argument("--out", default="data/raw_color/openclipart")
    p = sub.add_parser("pairs")
    p.add_argument("--src", default="data/raw_color/openclipart")
    p.add_argument("--out", default="data/pairs")
    p.add_argument("--long-side", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "fetch":
        ids = list(args.ids) + ([int(l) for l in open(args.ids_file) if l.strip()] if args.ids_file else [])
        fetch(ids, args.out)
    else:
        pairs(args.src, args.out, args.long_side, args.seed)


if __name__ == "__main__":
    main()
