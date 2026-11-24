"""enhanced USE ONLY: colored manga pages from MangaDex for enhanced training (owner in Switzerland,
URG Art. 19 enhanced use). These pages are copyrighted works (mostly scanlations); never use them
for the platform model, never publish models trained on them. See README "enhanced data".

  python data/enhanced_sources/mangadex_color.py --max-pages 20000
  python data/make_pairs.py data/enhanced/mangadex_color/pages --out data/pairs_enhanced --variants clean,tone

Titles tagged "Full Color", page format (Long Strip excluded), safe/suggestive only, most-followed
first; up to --chapters chapters per title and --pages-per-chapter pages each, data-saver images
(enough for training at ~768 px). Respects MangaDex rate limits (at-home server: 40 req/min) and
reports image loads to the MangaDex@Home network as its API docs ask.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

API = "https://api.mangadex.org"
# MangaDex asks API clients for a contact: set MANGADEX_CONTACT to an email or URL.
UA = {"User-Agent": f"colorunet-manga/0.1 ({os.environ.get('MANGADEX_CONTACT', 'no contact set')})"}
FULL_COLOR = "f5ba408b-0e7a-484d-8d49-4e9125ac96de"
LONG_STRIP = "3e2b8dae-350e-4ab8-a8ce-016e844b9f0d"
_last = {}


def paced(key, interval):
    wait = _last.get(key, 0) + interval - time.time()
    if wait > 0:
        time.sleep(wait)
    _last[key] = time.time()


def get_json(url, key="api", interval=0.25):
    for attempt in range(6):
        paced(key, interval)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            retry = float(e.headers.get("Retry-After") or 5 * 2**attempt)
            print(f"  HTTP {e.code}, waiting {retry:.0f} s", flush=True)
            time.sleep(min(retry, 300))
        except Exception as e:
            print(f"  retry after {e}", flush=True)
            time.sleep(5 * 2**attempt)
    raise RuntimeError(url)


_reporter = ThreadPoolExecutor(max_workers=2)


def _send_report(body):
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                "https://api.mangadex.network/report", data=body, headers={**UA, "Content-Type": "application/json"}
            ),
            timeout=3,
        )
    except Exception:
        pass


def report(url, ok, nbytes, ms):
    """Fire-and-forget: the report endpoint is often slow and must never hold up downloads."""
    body = json.dumps({"url": url, "success": ok, "bytes": nbytes, "duration": ms, "cached": False}).encode()
    if _reporter._work_queue.qsize() < 50:
        _reporter.submit(_send_report, body)


def titles(limit_titles):
    offset = 0
    while offset < limit_titles:
        q = urllib.parse.urlencode(
            [
                ("includedTags[]", FULL_COLOR),
                ("excludedTags[]", LONG_STRIP),
                ("contentRating[]", "safe"),
                ("contentRating[]", "suggestive"),
                ("hasAvailableChapters", "true"),
                ("order[followedCount]", "desc"),
                ("limit", 100),
                ("offset", offset),
            ]
        )
        d = get_json(f"{API}/manga?{q}")
        if not d["data"]:
            return
        yield from d["data"]
        offset += 100
        if offset >= d["total"]:
            return


def chapters(manga_id, n):
    q = urllib.parse.urlencode(
        [("limit", 100), ("order[chapter]", "asc"), ("includeExternalUrl", 0), ("includes[]", "scanlation_group")]
    )
    seen, out = set(), []
    for c in get_json(f"{API}/manga/{manga_id}/feed?{q}")["data"]:
        num = c["attributes"]["chapter"]
        if c["attributes"]["pages"] == 0 or num in seen:
            continue
        seen.add(num)
        groups = [
            r["attributes"]["name"] for r in c["relationships"] if r["type"] == "scanlation_group" and "attributes" in r
        ]
        out.append((c["id"], num, c["attributes"]["translatedLanguage"], groups))
        if len(out) >= n:
            break
    return out


def page_quality(data, min_side=800):
    """None if the page is good colored training data, else the reason it is not."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "unreadable"
    if max(img.shape[:2]) < min_side:
        return "too small"
    small = cv2.resize(img, (256, int(256 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    content = hsv[..., 2] < 235  # not paper white
    if content.mean() < 0.15:
        return "mostly empty (credits/title page)"
    colorful = (hsv[..., 1] > 50) & content
    if colorful.sum() / content.sum() < 0.25:
        return "black-and-white or faintly tinted"
    return None


def fetch_image(url, report_it):
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            data = r.read()
        ok = True
    except Exception:
        data, ok = b"", False
    if report_it:
        report(url, ok, len(data), int((time.time() - t0) * 1000))
    return data if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/enhanced/mangadex_color")
    ap.add_argument("--max-pages", type=int, default=60000)
    ap.add_argument("--max-titles", type=int, default=100000)
    ap.add_argument("--chapters", type=int, default=3, help="good chapters per title (variety beats depth)")
    ap.add_argument(
        "--chapters-best",
        type=int,
        default=6,
        help="good chapters for titles whose first good chapter is fully colored",
    )
    ap.add_argument("--tries", type=int, default=10, help="chapters looked at per title to find good ones")
    ap.add_argument("--pages-per-chapter", type=int, default=12)
    ap.add_argument("--parallel", type=int, default=3, help="concurrent image downloads (MangaDex: <= 3)")
    args = ap.parse_args()
    pages_dir = os.path.join(args.out, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    manifest = os.path.join(args.out, "manifest.jsonl")
    seen_chapters = set()
    if os.path.exists(manifest):
        seen_chapters = {json.loads(l)["chapter"] for l in open(manifest, encoding="utf-8")}
    total = len(os.listdir(pages_dir))
    rejected = {}
    pool = ThreadPoolExecutor(max_workers=args.parallel)

    with open(manifest, "a", encoding="utf-8") as mf:
        for m in titles(args.max_titles):
            if total >= args.max_pages:
                break
            title = next(iter(m["attributes"]["title"].values()))
            good_chapters, want, bw_in_row = 0, args.chapters, 0
            for ch_id, num, lang, groups in chapters(m["id"], args.tries):
                if good_chapters >= want or total >= args.max_pages:
                    break
                if bw_in_row >= 3:  # mostly a black-and-white series: next title
                    break
                if ch_id in seen_chapters:
                    good_chapters += 1
                    continue
                home = get_json(f"{API}/at-home/server/{ch_id}", key="athome", interval=1.6)  # 40/min
                base, h = home["baseUrl"], home["chapter"]["hash"]
                files = home["chapter"]["dataSaver"]
                # skip the first page (often a credits/cover page) when there are enough pages
                picks = list(enumerate(files[1 : args.pages_per_chapter + 1] if len(files) > 3 else files))
                url_of = lambda fn: f"{base}/data-saver/{h}/{fn}"
                report_it = "mangadex.org" not in base

                # 1) probe: look at up to 2 pages first; a black-and-white chapter is rejected after
                #    one download instead of a whole chapter's worth
                results, verdict = [], None
                for i, fn in picks[:2]:
                    data = fetch_image(url_of(fn), report_it)
                    why = page_quality(data) if data else "download failed"
                    results.append((i, fn, data, why))
                    if why is None or why == "black-and-white or faintly tinted":
                        verdict = why
                        break
                if verdict == "black-and-white or faintly tinted":
                    rejected[verdict] = rejected.get(verdict, 0) + 1
                    print(f"{total} pages | skip {title} ch.{num}: black-and-white", flush=True)
                    bw_in_row += 1
                    continue
                bw_in_row = 0
                # 2) the rest of the chapter in parallel (<= 3 at a time)
                rest = picks[len(results) :]
                for (i, fn), data in zip(rest, pool.map(lambda p: fetch_image(url_of(p[1]), report_it), rest)):
                    results.append((i, fn, data, page_quality(data) if data else "download failed"))

                kept = 0
                for i, fn, data, why in sorted(results):
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
                                "groups": groups,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                mf.flush()
                if kept:
                    # 3) a fully colored first good chapter marks a colored series: take more of it
                    if good_chapters == 0 and kept >= 0.8 * len(results):
                        want = args.chapters_best
                    good_chapters += 1
                print(
                    f"{total} pages | +{kept}/{len(results)} {title} ch.{num} ({lang}) | rejected {rejected}",
                    flush=True,
                )
    print(f"done: {total} pages in {pages_dir}; rejected {rejected}")


if __name__ == "__main__":
    main()
