# Detection harness

Offline testbed for Alt1-style screen detection on macOS captures. Ports the
clue solver's exact matching and OCR (from skillbert/alt1) so captures can be
tested without the game, and adds scale-tolerant matching with confidence
scores for scaled/blurry interfaces.

Assets (needle sprites, fonts, clue database) belong to runeapps.org and are
not committed. Fetch them first:

    python fetch_assets.py

Run:

    python run_harness.py --sanity            # synthetic round-trip checks
    python run_harness.py --frame frame.png   # exact match + scale sweep
    python part2_experiments.py               # confidence experiments
                                              # (expects captures in ./frames,
                                              #  override with RUNEKIT_FRAMES)

Files: `alt1port.py` (faithful Alt1 ports), `confidence.py` (ZNCC detection,
template-based soft OCR, clue DB fuzzy match, Apple Vision wrapper),
`run_harness.py`, `part2_experiments.py`.

Celtic knot reader check on saved 1x crops (the app leaves one at
`<config>/RuneKit/debug_knot_reject.png` after a failed read; keep good ones
in `frames/knot_*.png`):

    python knot_check.py --rings 16,16,16,0 --crossings 8 frames/knot_3ring_1233.png
