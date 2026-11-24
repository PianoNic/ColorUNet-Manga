"""Download grayscale test pages from MangaDex, the way a manga reader app gets them.

Flow: /manga search -> /manga/{id}/feed -> /at-home/server/{chapter}
-> {baseUrl}/data/{hash}/{file}. Test use only: these pages belong to their creators.

  python data/fetch_mangadex.py --title "More Than a Doll" --chapters 3 --out eval/testset/more-than-a-doll
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

API = "https://api.mangadex.org"
UA = "manga-colors-eval/0.1 (local testing)"


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        f.write(r.read())


def find_manga(title):
    q = urllib.parse.urlencode({"title": title, "limit": 5, "order[relevance]": "desc"})
    res = get_json(f"{API}/manga?{q}")["data"]
    if not res:
        raise SystemExit(f"no MangaDex result for {title!r}")
    for m in res:
        name = next(iter(m["attributes"]["title"].values()))
        print(f"  candidate: {name}  ({m['id']})")
    return res[0]["id"], next(iter(res[0]["attributes"]["title"].values()))


def chapters(manga_id, lang, limit):
    q = urllib.parse.urlencode(
        [("translatedLanguage[]", l) for l in lang.split(",")]
        + [("order[chapter]", "asc"), ("limit", 100), ("includeExternalUrl", 0)]
    )
    out, seen = [], set()
    for c in get_json(f"{API}/manga/{manga_id}/feed?{q}")["data"]:
        num = c["attributes"]["chapter"]
        if c["attributes"]["pages"] == 0 or num in seen:
            continue
        seen.add(num)
        out.append((num, c["id"]))
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="More Than a Doll")
    ap.add_argument("--manga-id", help="skip search and use this MangaDex id")
    ap.add_argument("--lang", default="en", help="comma list, e.g. en,es,pt-br (text language barely matters here)")
    ap.add_argument("--chapters", type=int, default=3)
    ap.add_argument("--data-saver", action="store_true", help="compressed pages (smaller, lossy)")
    ap.add_argument("--out", default="eval/testset/more-than-a-doll")
    args = ap.parse_args()

    manga_id, name = (args.manga_id, args.manga_id) if args.manga_id else find_manga(args.title)
    print(f"using: {name} ({manga_id})")
    os.makedirs(args.out, exist_ok=True)
    folder, key = ("data-saver", "dataSaver") if args.data_saver else ("data", "data")

    for num, ch_id in chapters(manga_id, args.lang, args.chapters):
        home = get_json(f"{API}/at-home/server/{ch_id}")
        base, h = home["baseUrl"], home["chapter"]["hash"]
        files = home["chapter"][key]
        print(f"chapter {num}: {len(files)} pages")
        for i, fn in enumerate(files, 1):
            dst = os.path.join(args.out, f"ch{float(num or 0):06.1f}_p{i:03d}{os.path.splitext(fn)[1]}")
            if not os.path.exists(dst):
                download(f"{base}/{folder}/{h}/{fn}", dst)
                time.sleep(0.25)  # be polite to MangaDex@Home
        time.sleep(1)


if __name__ == "__main__":
    main()
