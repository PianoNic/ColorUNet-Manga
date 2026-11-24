"""Public-domain color comics from Wikimedia Commons (Category:Public domain comics, recursively).

  python data/fetch_commons.py --size-only                    # walk categories, write manifest, print size
  python data/fetch_commons.py --download --max-gb 1          # then download what the manifest lists

Keeps only files whose Commons license metadata clearly says public domain (LicenseShortName
"Public domain" or a PD-* license template), and only raster images (jpg/png/tif/webp); PDF/DjVu
scans are counted but skipped. Skips AI-generated subcategories. Every kept file goes into
data/raw_color/commons/manifest.jsonl with its source page, license, author and category.
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "manga-colors-dataset/0.1 (public-domain comics for colorization research; python urllib)"}
ROOT = "Category:Public domain comics"
SKIP_CATS = re.compile(r"AI[- ]generated|artificial intelligence|Zarya of the Dawn|Midjourney|Stable Diffusion", re.I)
IMAGE_MIME = {"image/jpeg", "image/png", "image/tiff", "image/webp"}


# Wikimedia rate limits (2026, https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits): clients
# without contact info in the User-Agent get 10 requests/min; with an email or URL in it, 200/min.
# Set COMMONS_CONTACT (an email or URL) to be identified; otherwise we pace at ~9/min.
CONTACT = os.environ.get("COMMONS_CONTACT")
if CONTACT:
    UA = {"User-Agent": f"manga-colors-dataset/0.1 ({CONTACT}) python-urllib"}
PACE = 0.35 if CONTACT else 6.5
_last = [0.0]


def api(**params):
    params.update(format="json", formatversion=2)
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        wait = _last[0] + PACE - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            retry = float(e.headers.get("Retry-After") or max(5, 5 * 2**attempt))
            print(f"  HTTP {e.code}, waiting {retry:.0f} s", flush=True)
            time.sleep(min(retry, 300))
        except Exception as e:
            print(f"  retry {attempt + 1} after {e}", flush=True)
            time.sleep(min(60, 5 * 2**attempt))
    raise RuntimeError(f"API failed: {url}")


def walk(max_depth):
    """Yield (category, file title) for all files under ROOT, breadth first, skipping AI categories."""
    seen, queue = {ROOT}, [(ROOT, 0)]
    while queue:
        cat, depth = queue.pop(0)
        cont = {}
        while True:
            d = api(action="query", list="categorymembers", cmtitle=cat, cmtype="file|subcat", cmlimit=500, **cont)
            n_files = 0
            for m in d["query"]["categorymembers"]:
                if m["ns"] == 14:
                    if depth < max_depth and m["title"] not in seen and not SKIP_CATS.search(m["title"]):
                        seen.add(m["title"])
                        queue.append((m["title"], depth + 1))
                elif m["ns"] == 6:
                    n_files += 1
                    yield cat, m["title"]
            print(f"  [{len(seen)} cats seen, {len(queue)} queued] depth {depth} {cat}: {n_files} files", flush=True)
            if "continue" not in d:
                break
            cont = {"cmcontinue": d["continue"]["cmcontinue"]}
    print(f"{len(seen)} categories walked")


def is_public_domain(meta):
    short = meta.get("LicenseShortName", {}).get("value", "")
    lic = meta.get("License", {}).get("value", "")
    return "public domain" in short.lower() or lic.lower().startswith("pd")


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


# Characters whose old stories are public domain but whose names/designs are still trademarks
# (same rule as the Openclipart blocklist): Disney, King Features, Paws, ERB, DC, Dille, ...
TRADEMARKED = re.compile(
    r"mickey|minnie|popeye|olive oyl|garfield|buck rogers|tarzan|captain marvel|shazam|"
    r"felix the cat|superman|batman|betty boop|krazy kat|flash gordon|phantom|mandrake|"
    r"dick tracy|little orphan annie|peanuts|snoopy|tintin|asterix",
    re.I,
)


def skip_reason(r, min_side=800):
    """Files we don't train on even though they are public domain."""
    if TRADEMARKED.search(r["title"] + " " + r.get("category", "")):
        return "trademarked character"
    if r.get("restrictions"):
        return f"restricted ({r['restrictions']})"  # trademarked, nazi symbols, currency
    if max(r["width"], r["height"]) < min_side:
        return "too small"
    if re.search(r"\blogo|photo|\(\d{9,}\)|advert", r["title"], re.I):
        return "not a comic page (logo/photo/advert)"
    return None


def file_info(titles):
    d = api(
        action="query",
        prop="imageinfo",
        titles="|".join(titles),
        iiprop="url|size|mime|extmetadata",
        iiextmetadatafilter="LicenseShortName|License|Artist|Credit|Restrictions",
    )
    for p in d["query"]["pages"]:
        if "imageinfo" in p:
            yield p["title"], p["imageinfo"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_color/commons")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--size-only", action="store_true")
    ap.add_argument("--download", action="store_true", help="download the files in the manifest")
    ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--max-side", type=int, default=1600, help="ask Commons for a resized version no bigger than this")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "manifest.jsonl")

    if not args.download:
        files = {}
        for cat, title in walk(args.max_depth):
            files.setdefault(title, cat)
        print(f"{len(files)} files found")
        kept, stats = [], {"not_pd": 0, "not_image": 0, "not_image_bytes": 0}
        titles = list(files)
        for i in range(0, len(titles), 50):
            for title, ii in file_info(titles[i : i + 50]):
                meta = ii.get("extmetadata", {})
                if not is_public_domain(meta):
                    stats["not_pd"] += 1
                    continue
                if ii["mime"] not in IMAGE_MIME:
                    stats["not_image"] += 1
                    stats["not_image_bytes"] += ii["size"]
                    continue
                kept.append(
                    {
                        "title": title,
                        "category": files[title],
                        "url": ii["url"],
                        "page": ii["descriptionurl"],
                        "mime": ii["mime"],
                        "bytes": ii["size"],
                        "width": ii["width"],
                        "height": ii["height"],
                        "license": meta.get("LicenseShortName", {}).get("value", ""),
                        "license_template": meta.get("License", {}).get("value", ""),
                        "artist": strip_html(meta.get("Artist", {}).get("value", "")),
                        "restrictions": meta.get("Restrictions", {}).get("value", ""),
                    }
                )
            time.sleep(0.2)
        with open(manifest, "w", encoding="utf-8") as f:
            for k in kept:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
        gb = sum(k["bytes"] for k in kept) / 1e9
        print(
            f"kept {len(kept)} public-domain images, {gb:.2f} GB at full size; skipped {stats['not_pd']} "
            f"not clearly PD, {stats['not_image']} non-image files ({stats['not_image_bytes'] / 1e9:.2f} GB)"
        )
        print(f"manifest: {manifest}")
        return

    rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    for r in rows:
        r["skip"] = skip_reason(r)
    print(f"{sum(not r['skip'] for r in rows)} of {len(rows)} files pass the filter")
    total = 0
    for i, r in enumerate(rows):
        if r["skip"]:
            continue
        ext = os.path.splitext(urllib.parse.urlparse(r["url"]).path)[1].lower()  # urls carry ?utm_ params
        name = re.sub(r"[^\w.-]+", "_", r["title"].split(":", 1)[1])
        dst = os.path.join(args.out, "pages", os.path.splitext(name)[0] + ext)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        r["file"] = os.path.relpath(dst, args.out)
        if os.path.exists(dst):
            total += os.path.getsize(dst)
            continue
        url = r["url"]
        if max(r["width"], r["height"]) > args.max_side:  # Commons thumbnail service, keeps PD status
            ii = api(
                action="query",
                prop="imageinfo",
                titles=r["title"],
                iiprop="url",
                iiurlwidth=int(args.max_side * r["width"] / max(r["width"], r["height"])),
            )
            url = ii["query"]["pages"][0]["imageinfo"][0].get("thumburl", url)
        data = None
        for attempt in range(6):
            wait = _last[0] + PACE - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as resp:
                    data = resp.read()
                break
            except urllib.error.HTTPError as e:
                retry = float(e.headers.get("Retry-After") or 5 * 2**attempt)
                print(f"  {r['title']}: HTTP {e.code}, waiting {retry:.0f} s", flush=True)
                time.sleep(min(retry, 300))
        if data is None:
            r["skip"] = "download failed"
            continue
        if (total + len(data)) / 1e9 > args.max_gb:
            raise SystemExit(f"stopping before {args.max_gb} GB ({total / 1e9:.2f} GB, {i} files)")
        with open(dst, "wb") as f:
            f.write(data)
        total += len(data)
        r["file"] = os.path.relpath(dst, args.out)
        if i % 100 == 0:
            print(f"{i}/{len(rows)} {total / 1e9:.2f} GB", flush=True)
    with open(manifest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} files, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
