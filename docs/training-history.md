# Training history

How the released checkpoint came about (October 2026, RTX 5080 locally and RunPod for stage 3).
Numbers are held-out Lab a/b L1 (lower is better) on each stage's own held-out pages, so they are
comparable within a stage, not across stages.

| Stage | Started from | Data | Steps | Held-out a/b L1 |
| --- | --- | --- | --- | --- |
| 0 | ImageNet ConvNeXt-Tiny | Pepper&Carrot | stopped at 1k | deleted with the data (licence change, see [DATASETS.md](../training/DATASETS.md)) |
| 1 | ImageNet ConvNeXt-Tiny | Commons public-domain comics + Openclipart, 2,459 color pages | 20k (best 18k) | 10.30 → 8.80 |
| 2 | stage 1 | 13,926 MangaDex color pages + 15% Commons, white-paper augmentation | 55k (best 42.5k) | 7.84 → 7.31 |
| 3 | stage 2 | 6,663 FLUX.2 [klein] 4B teacher colorings of MangaDex pages + 25% stage-2 pages; new series-cover reference encoder (28M) and color-class head | 8k | 3.86 → 4.09 (unstable) |
| 3b | stage 3 snapshot | same | 2k | 3.66 |
| 3c | stage 3b snapshot | same | 18k (released: 10k) | 3.80 → 3.63 (18k) |

Step 10,000 of stage 3c (3.67 without hints) was picked over the lowest-loss steps by eye: its colors
are fuller (colorfulness 0.97 of the teacher's, against 0.86 at step 6,000) without new artefacts.

What each stage fixed:

- **Stage 1** learned general comic coloring but left skin and light hair white: in the training
  pairs (converted color pages) skin is light gray, while in manga it is bare paper.
- **Stage 2** (real manga, white-paper augmentation) added blush and light tints and removed stage 1's
  smears; hints started to work (blonde / pink / blue: 14% / 2% / 3% → 49% / 26% / 22% of the way).
- **Stage 3** made colors consistent per series: the series cover goes in as a reference, so
  characters keep their hair, eye and clothing colors across pages. The teacher (FLUX.2 [klein] 4B, a
  4-billion-parameter image editor) is far too slow for a reader; the student is a single 61M pass.
