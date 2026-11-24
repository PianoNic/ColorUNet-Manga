# Creator data intake

What we ask creators for, in exchange for (for example) free colorization of their series on the
platform. Pages are used only to train our colorization model; they are never republished.

## What we need

Most valuable first:

1. **Matching pairs: the same page in black-and-white and in color.** The ideal training example.
   Any official or self-made colored version of a B/W page works (color chapters, colored webtoon
   versions, colored reprints, color covers that also exist as line art).
2. **Color pages, covers and illustrations on their own**, in the style of the series. We make the
   B/W input ourselves; still very useful.
3. **Flat-color "base" layers** (before shading or effects), if the creator works in layers. These
   teach clean, flat coloring.
4. **B/W pages without color versions:** not needed for training, but useful as test pages, with
   the creator's permission.

Volume: anything helps; a few dozen pages from one series already improves that series. Thousands of
pages across many creators is what makes the model good for everyone.

Variety we want: different art styles, digital and print (screentone) pages, action, quiet scenes,
text-heavy pages, backgrounds, night scenes, and series with many recurring characters.

## Formats

| | Best | Also fine | Avoid |
| --- | --- | --- | --- |
| File type | PNG, lossless TIFF, PSD/CLIP (layers) | High-quality JPEG (quality 90+), WebP lossless | Screenshots, re-compressed social media images |
| Resolution | Original working resolution (e.g. 2000-4000 px long side) | ≥ 1200 px on the long side | Under 1000 px |
| Pairs | Same size and framing in B/W and color, same file name with `_bw` / `_color` | Slightly different crop (we align them) | Redrawn or retraced color versions |
| Text | Either way; text-free versions are a bonus | Lettered pages | Watermarks over the art |
| Delivery | A zip or shared folder per series, one subfolder per chapter | | |

Please also tell us: series name, which pages are pairs, and whether anyone else (publisher,
co-artist, colorist) holds rights to the pages.

## Permission template

*Template only, not legal advice; have a lawyer review it before use.*

> **Permission to use artwork for training a colorization model**
>
> I, **[creator name / pen name]**, confirm that I own the rights to the artwork listed below (or am
> authorized by all rights holders to grant this permission): **[series name, chapters/pages, or
> "all files delivered in [folder/zip name]"]** (the "Artwork").
>
> I grant **[platform company name]** a non-exclusive, worldwide, royalty-free, perpetual and
> irrevocable permission to:
>
> 1. use the Artwork, and copies or modified versions of it (for example resized, converted to black
>    and white, or cropped), to **train, evaluate and improve machine learning models** that colorize
>    manga and comics;
> 2. use those trained models, and the colorized images they produce for any series,
>    **commercially**, including on the [platform name] platform, without further payment to me
>    beyond what is stated below.
>
> **[Platform company name] agrees:**
>
> - not to publish, sell or redistribute the Artwork itself, outside of the colorization service
>   for my own series;
> - not to use the Artwork to train models that generate new artwork in my style or imitate my
>   characters;
> - to store the Artwork securely and delete the original files on my written request (models
>   already trained may continue to be used);
> - in return: **[what the creator receives, e.g. free colorization of my series on the platform /
>   payment of X / credit on the platform's credits page]**.
>
> Credit: I **[want / do not want]** to be credited as a training data contributor, as
> **[credit name]**.
>
> Name: ____________________ Date: __________ Signature: ____________________
>
> Contact (email): ____________________

## Our side (internal checklist)

- Keep each signed permission with the delivered files and record it in `data/raw_color/<source>/manifest.jsonl`
  (creator, series, files, date, permission document).
- Never mix creator data with data whose license is unclear.
- Honor deletion requests for the source files; note them in the manifest.
