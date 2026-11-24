# Reference pages (evaluation only — never training)

Seedream 5.0 Flash colorizations of Sono Bisque Doll pages, provided by the owner as a quality target
and as a palette source for testing hints. Seedream returns 2048x2048 squares; the inputs are
landscape spreads, so outputs are resized to the input size before comparing.

| Input (gray) | Seedream output | Original file names (H:\Downloads\ta) |
| --- | --- | --- |
| spread1_gray.png (1500x1071) | spread1_seedream.jpeg | Download.png / Download (1).jpeg |
| spread2_gray.png (1218x867) | spread2_seedream.jpeg | Download (1).png / Download (2).jpeg |
| cover_gray.png (1649x1182) | cover_seedream.jpeg | dark_collabsed.png / Download.jpeg |

Do not train on these: they are a commercial manga's pages run through a hosted model.

## Probe-page target

`probe_pages_target.jpeg`: a hosted model's colorization of the four probe pages that training
samples every N steps (ch. 114 p1, ch. 114 p17, ch. 115 p13, ch. 115.5 p9; originals side by side in
the same layout). The owner's quality target for the final model. Evaluation only, never training.
Note: it redraws the line art and garbles lettering ("GOLOS TU Y YO"); our pipeline keeps the
original lines and text exactly, so compare colors, not lines.
