# Training data

Every source used for a released checkpoint, its licence, and whether it may be redistributed. The
training data itself is **not published**: the stage-2 and stage-3 sources are commercial manga pages
that may not be redistributed, and the stage-1 sources are easy to fetch again with the scripts here.
Licences checked 2026-10-08 to 2026-10-13.

## Used

| Source | Stages | Licence / terms | Redistributable | Fetch / build |
| --- | --- | --- | --- | --- |
| Wikimedia Commons, [Category:Public domain comics](https://commons.wikimedia.org/wiki/Category:Public_domain_comics) (Golden Age US comics): 3,357 files, 2,459 in color | 1, 2 (15%), 3 (via stage 2) | Public domain, per each file's Commons licence metadata (kept in `manifest.jsonl`); trademarked characters (Mickey, Popeye, Garfield, Tarzan, Captain Marvel, ...) excluded | yes | `data/fetch_commons.py`, `data/make_pairs.py --clean-scans` |
| [Openclipart](https://openclipart.org) | 1 | CC0 1.0, checked per item; titles/tags naming copyrighted characters, brands, logos or AI generation dropped | yes | `data/openclipart.py` |
| [timm `convnext_tiny.fb_in1k`](https://huggingface.co/timm/convnext_tiny.fb_in1k) (encoder initialisation) | all | Weights Apache-2.0 (timm model card), ConvNeXt code MIT; trained on ImageNet-1k, whose image terms are research-only | weights yes | downloaded by timm |
| MangaDex color pages (official color editions and fan colorings): 13,926 pages from 1,614 series | 2, 3 (25% mix) | Copyrighted by their publishers and authors; read through the public MangaDex API | **no** | `data/enhanced_sources/mangadex_color.py` |
| MangaDex black-and-white pages: 8,031 pages from 63 series, each with its series cover | 3 | Copyrighted by their publishers and authors | **no** | `data/enhanced_sources/mangadex_bw.py` |
| Teacher colorings of those pages by [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), with the series cover as color reference: 6,663 training pairs (7,074 after refresh), 212 held out | 3 | Model Apache-2.0; the outputs are colorings of the copyrighted pages above | **no** | `data/teacher/klein_colorize.py`, `data/teacher/make_teacher_pairs.py` |
| Creators of the platform | none yet | Signed permission (`data/creator_intake.md`) | per agreement | as delivered |

The released checkpoint ([ColorUNet-Manga-61M](https://huggingface.co/PianoNic/ColorUNet-Manga-61M),
stage 3c) was trained on the "no" rows above. The weights contain no page images, but whether training
on copyrighted pages is allowed depends on the jurisdiction; check before using the weights commercially.
A license-clean checkpoint (stage 1: Commons + Openclipart only) can be rebuilt with the commands in
[README.md](README.md).

## Rejected

| Source | Why |
| --- | --- |
| Pepper&Carrot (David Revoy, CC BY 4.0) | Our first source. On 2026-10-08 the author added "AI derivation prohibited" to all his art, applied retroactively ([post](https://www.davidrevoy.com/article1178/license-update-ai-derivation-prohibited-on-all-my-art-lore-stories-and-comics)). We deleted the pages, the 564 pairs and the checkpoint trained on them; `data/fetch_peppercarrot.py` refuses to run. |
| manga-colorization-v2 (code, weights, FFDNet denoiser) | Used as an early baseline; no licence that allows deriving a model from it. Removed 2026-10-13 with everything made with it; nothing here derives from it. |
