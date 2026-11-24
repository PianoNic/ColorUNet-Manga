"""Side-by-side crops of several outputs of the same page, at full resolution, with labels.

  python eval/compare_crops.py ch0114.0_p013 --cols original:eval/testset/more-than-a-doll \
      stock:eval/out/baseline ours:eval/out/v2_768 --boxes 2180,180 1550,450 --size 360 \
      --out eval/compare/fix1_lightness.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import io as mio  # noqa: E402


def load(folder, page):
    for p in mio.list_images(folder):
        if os.path.splitext(os.path.basename(p))[0] == page:
            img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
            return img
    raise FileNotFoundError(f"{page} not in {folder}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page", help="file name without extension")
    ap.add_argument("--cols", nargs="+", required=True, help="label:folder, one per column")
    ap.add_argument("--boxes", nargs="+", required=True, help="x,y top-left of each crop row (full-res px)")
    ap.add_argument("--size", type=int, default=360)
    ap.add_argument("--zoom", type=int, default=1, help="nearest-neighbour zoom of each crop")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cols = [(c.split(":", 1)[0], load(c.split(":", 1)[1], args.page)) for c in args.cols]
    s, z = args.size, args.zoom
    header = np.full((34, s * z * len(cols), 3), 255, np.uint8)
    for i, (label, _) in enumerate(cols):
        cv2.putText(header, label, (i * s * z + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    rows = [header]
    for b in args.boxes:
        x, y = map(int, b.split(","))
        crops = []
        for _, img in cols:
            c = img[y : y + s, x : x + s]
            c = cv2.copyMakeBorder(c, 0, s - c.shape[0], 0, s - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            crops.append(cv2.resize(c, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST))
        rows.append(np.concatenate(crops, 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imencode(".jpg", np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
