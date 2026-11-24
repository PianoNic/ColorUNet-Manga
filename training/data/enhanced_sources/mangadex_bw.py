"""enhanced USE ONLY: black-and-white manga pages + each series' colored cover from MangaDex, as input
for the local teacher colorizer (data/teacher/klein_colorize.py). Same rules as mangadex_color.py
(URG Art. 19 enhanced use; never for the platform model; see README "enhanced data").

  python data/enhanced_sources/mangadex_bw.py --max-pages 8000

Most-followed titles in page format (no Long Strip, no Full Color), safe/suggestive; --chapters
chapters per title, --pages-per-chapter pages each (data-saver images). Colored pages are skipped.
The series cover (colored) is saved to <out>/refs/<manga id>.jpg as the color reference.
More Than a Doll (our evaluation series) is excluded.
Rate limits: API ~4 req/s, at-home server lookups 40/min, <= 3 image downloads at a time; the next
chapter's server lookup runs while the current chapter's images download.
"""

import argparse
import json
import os
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mangadex_color import API, FULL_COLOR, LONG_STRIP, fetch_image, get_json  # noqa: E402

EVAL_SERIES = {"aa6c76f7-5f5f-46b6-a800-911145f81b9b"}  # More Than a Doll: evaluation only


def titles(limit_titles):
    offset = 0
    while offset < limit_titles:
        q = urllib.parse.urlencode(
            [
                ("excludedTags[]", FULL_COLOR),
                ("excludedTags[]", LONG_STRIP),
                ("contentRating[]", "safe"),
                ("contentRating[]", "suggestive"),
                ("hasAvailableChapters", "true"),
                ("order[followedCount]", "desc"),
                ("includes[]", "cover_art"),
                ("limit", 100),
                ("offset", offset),
            ]
        )
        d = get_json(f"{API}/manga?{q}")
        if not d["data"]:
            return
        yield from d["data"]
        offset += 100
        if offset >= min(d["total"], 10000):
            return


def chapters(manga_id, n):
    q = urllib.parse.urlencode([("limit", 100), ("order[chapter]", "asc"), ("includeExternalUrl", 0)])
    seen, out = set(), []
    for c in get_json(f"{API}/manga/{manga_id}/feed?{q}")["data"]:
        num = c["attributes"]["chapter"]
        if c["attributes"]["pages"] < 8 or num in seen:
            continue
        seen.add(num)
        out.append((c["id"], num, c["attributes"]["translatedLanguage"]))
    # spread the picks over the series (not only chapter 1-2)
    if len(out) > n:
        out = [out[round(i * (len(out) - 1) / (n - 1))] for i in range(n)] if n > 1 else out[:1]
    return out


def bw_quality(data, min_side=800):
    """None if the page is a good black-and-white manga page, else the reason it is not."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "unreadable"
    if max(img.shape[:2]) < min_side:
        return "too small"
    small = cv2.resize(img, (256, int(256 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    content = hsv[..., 2] < 235
    if content.mean() < 0.12:
        return "mostly empty"
    if ((hsv[..., 1] > 40) & content).sum() / max(1, content.sum()) > 0.05:
        return "colored"
    return None


def cover_url(m):
    for r in m["relationships"]:
        if r["type"] == "cover_art" and "attributes" in r:
            return f"https://uploads.mangadex.org/covers/{m['id']}/{r['attributes']['fileName']}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/enhanced/mangadex_bw")
    ap.add_argument("--max-pages", type=int, default=8000)
    ap.add_argument("--chapters", type=int, default=2)
    ap.add_argument("--pages-per-chapter", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=3, help="concurrent image downloads (MangaDex: <= 3)")
    args = ap.parse_args()
    pages_dir, refs_dir = os.path.join(args.out, "pages"), os.path.join(args.out, "refs")
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(refs_dir, exist_ok=True)
    manifest = os.path.join(args.out, "manifest.jsonl")
    seen = set()
    if os.path.exists(manifest):
        seen = {json.loads(l)["chapter"] for l in open(manifest, encoding="utf-8")}
    total = len(os.listdir(pages_dir))
    rejected = {}
    pool = ThreadPoolExecutor(max_workers=args.parallel)
    lookup = ThreadPoolExecutor(max_workers=1)  # at-home lookups, paced to 40/min inside get_json

    def at_home(ch_id):
        return get_json(f"{API}/at-home/server/{ch_id}", key="athome", interval=1.6)

    def jobs():
        for m in titles(100000):
            if m["id"] in EVAL_SERIES:
                continue
            cu = cover_url(m)
            if not cu:
                continue
            for ch_id, num, lang in chapters(m["id"], args.chapters):
                if ch_id not in seen:
                    yield m, cu, ch_id, num, lang

    with open(manifest, "a", encoding="utf-8") as mf:
        it = jobs()
        nxt = next(it, None)
        fut = lookup.submit(at_home, nxt[2]) if nxt else None
        while nxt and total < args.max_pages:
            m, cu, ch_id, num, lang = nxt
            home = fut.result()
            nxt = next(it, None)
            fut = lookup.submit(at_home, nxt[2]) if nxt else None  # prefetch the next lookup
            title = next(iter(m["attributes"]["title"].values()))
            ref = os.path.join(refs_dir, m["id"] + ".jpg")
            if not os.path.exists(ref):
                data = fetch_image(cu + ".512.jpg", False) or fetch_image(cu, False)
                if not data:
                    continue
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if img is not None else None
                if hsv is None or (hsv[..., 1] > 50).mean() < 0.15:  # gray cover: no color reference
                    rejected["no colored cover"] = rejected.get("no colored cover", 0) + 1
                    continue
                with open(ref, "wb") as f:
                    f.write(data)
            base, h = home["baseUrl"], home["chapter"]["hash"]
            files = home["chapter"]["dataSaver"]
            picks = list(enumerate(files[1 : args.pages_per_chapter + 1]))  # skip the cover/credits page
            report_it = "mangadex.org" not in base
            datas = pool.map(lambda p: fetch_image(f"{base}/data-saver/{h}/{p[1]}", report_it), picks)
            kept = 0
            for (i, fn), data in zip(picks, datas):
                why = bw_quality(data) if data else "download failed"
                if why:
                    rejected[why] = rejected.get(why, 0) + 1
                    continue
                dst = os.path.join(pages_dir, f"{m['id'][:8]}_{ch_id[:8]}_{i:03d}{os.path.splitext(fn)[1]}")
                with open(dst, "wb") as f:
                    f.write(data)
                kept += 1
                total += 1
                mf.write(
                    json.dumps(
                        {
                            "file": os.path.basename(dst),
                            "manga": m["id"],
                            "title": title,
                            "chapter": ch_id,
                            "chapter_no": num,
                            "lang": lang,
                            "ref": os.path.basename(ref),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            mf.flush()
            print(f"{total} pages | +{kept}/{len(picks)} {title} ch.{num} ({lang}) | rejected {rejected}", flush=True)
    print(f"done: {total} pages in {pages_dir}; rejected {rejected}")


if __name__ == "__main__":
    main()
