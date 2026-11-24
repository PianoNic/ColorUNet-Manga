"""Download Pepper&Carrot pages (David Revoy, CC BY 4.0).

NOT FOR TRAINING. On 2026-10-08 the author added "AI derivation prohibited" to all his art, applied
retroactively (https://www.davidrevoy.com/article1178/license-update-ai-derivation-prohibited-on-all-my-art-lore-stories-and-comics).
We stopped using it on 2026-10-12 and deleted the pages, pairs and the checkpoint trained on them.
The script is kept for reference only and refuses to run without --override-ai-prohibition.

  python data/fetch_peppercarrot.py --out data/raw_color/peppercarrot
  python data/make_pairs.py data/raw_color/peppercarrot --out data/pairs --variants clean,tone

English pages (with speech bubbles, which teaches the model that bubbles stay white), hi-res.
~285 story pages, ~630 MB. Writes ATTRIBUTION.md next to the pages with what CC BY 4.0 requires:
author, title, source link, license link, and the changes we make. Stops before passing --max-gb.
"""

import argparse
import json
import os
import time
import urllib.request

BASE = "https://www.peppercarrot.com/0_sources"
UA = {"User-Agent": "manga-colors-data/0.1 (training data, CC BY 4.0 attribution kept)"}

ATTRIBUTION = """# Pepper&Carrot — attribution

Pages in this folder are from **Pepper&Carrot** by **David Revoy**, https://www.peppercarrot.com,
licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**,
https://creativecommons.org/licenses/by/4.0/ .

- Art and scenario: David Revoy. English text: the Pepper&Carrot translation and proofreading
  contributors credited on each episode's credits page and at https://www.peppercarrot.com/en/license/index.html .
- Downloaded {date} from {base}/<episode>/hi-res/ (English versions). Per-page source URLs are listed below.
- **Changes made by us:** pages are resized, converted to grayscale, and some copies get synthetic
  screentone and JPEG compression, to train a manga colorization model. The original color pages
  are used as training targets. These changes are ours and not endorsed by the author.

When a model trained on this data is published or deployed, keep this credit with it (e.g. on the
platform's credits page): "Training data includes Pepper&Carrot by David Revoy, www.peppercarrot.com,
CC BY 4.0."

## Pages

| File | Source |
| --- | --- |
"""


def get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_color/peppercarrot")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--res", default="hi-res", choices=["hi-res", "low-res"])
    ap.add_argument("--max-gb", type=float, default=1.0, help="stop before downloading more than this")
    ap.add_argument("--override-ai-prohibition", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.override_ai_prohibition:
        raise SystemExit(
            'Pepper&Carrot is "AI derivation prohibited" since 2026-10-08; not used for training. See the docstring.'
        )

    os.makedirs(args.out, exist_ok=True)
    episodes = json.load(get(f"{BASE}/episodes-v1.json"))
    rows, total = [], 0
    for ep in episodes:
        if args.lang not in ep.get("translated_languages", []):
            print(f"skip {ep['name']}: no {args.lang} version")
            continue
        for key, fn in sorted(ep["pages"].items()):
            if not key.isdigit():  # skip cover, title and credits pages
                continue
            name = f"{args.lang}_{fn}"
            dst = os.path.join(args.out, name)
            url = f"{BASE}/{ep['name']}/{args.res}/{name}"
            if not os.path.exists(dst):
                with get(url) as r:
                    size = int(r.headers.get("Content-Length", 0))
                    if (total + size) / 1e9 > args.max_gb:
                        raise SystemExit(f"stopping: would pass {args.max_gb} GB ({total / 1e9:.2f} GB so far)")
                    data = r.read()
                with open(dst, "wb") as f:
                    f.write(data)
                time.sleep(0.2)
            total += os.path.getsize(dst)
            rows.append(f"| {name} | {url} |")
        print(f"{ep['name']}: done, {total / 1e6:.0f} MB so far")

    with open(os.path.join(args.out, "ATTRIBUTION.md"), "w", encoding="utf-8") as f:
        f.write(ATTRIBUTION.format(date=time.strftime("%Y-%m-%d"), base=BASE) + "\n".join(rows) + "\n")
    print(f"{len(rows)} pages, {total / 1e6:.0f} MB; attribution in {args.out}/ATTRIBUTION.md")


if __name__ == "__main__":
    main()
