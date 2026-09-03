"""Regression check for the celtic knot reader on saved 1x crops.

    python knot_check.py [--rings 16,16,16,0] [--crossings 8] crop.png ...

Each crop is the 504x326 normalized knot rect that a rejected read leaves at
<app config>/RuneKit/debug_knot_reject.png; copy the ones worth keeping into
frames/ (ignored by git, the artwork is Jagex's). Without crops it runs every
frames/knot_*.png. Needle sprites come from the app's own cache, so run the
app once first; override with --cache.
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runekit import detection  # noqa: E402
from runekit.cluehelper import knot  # noqa: E402
from runekit.cluehelper.assets import ClueAssets  # noqa: E402

DEFAULT_CACHE = os.path.expanduser("~/Library/Preferences/cupco.de/RuneKit")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crops", nargs="*")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="app config dir holding clue_assets/")
    ap.add_argument("--rings", help="expected slots per ring, blue,gold,black,silver")
    ap.add_argument("--crossings", type=int, help="expected crossing count")
    args = ap.parse_args()
    crops = args.crops or sorted(glob.glob(str(Path(__file__).parent / "frames" / "knot_*.png")))
    if not crops:
        print("no crops given and none in frames/knot_*.png")
        return 2
    assets = ClueAssets(Path(args.cache))
    want_rings = [int(v) for v in args.rings.split(",")] if args.rings else None
    failed = 0
    for crop in crops:
        norm = detection.to_array(np.asarray(Image.open(crop).convert("RGB")))
        started = time.perf_counter()
        state = knot.read_normalized(norm, assets)
        elapsed = (time.perf_counter() - started) * 1000
        if state is None:
            print(f"{crop}: REJECTED ({elapsed:.0f} ms)")
            failed += 1
            continue
        rings = [len(p) for p in state.paths]
        solutions = knot.solve_knot(state)
        best = knot.pick_solution(solutions)
        verdict = "ok"
        if want_rings is not None and rings != want_rings:
            verdict = f"FAIL rings {rings} != {want_rings}"
        elif args.crossings is not None and len(state.intersections) != args.crossings:
            verdict = f"FAIL crossings {len(state.intersections)} != {args.crossings}"
        elif not solutions:
            verdict = "FAIL no solution"
        failed += verdict != "ok"
        print(
            f"{crop}: rings {rings}, {len(state.intersections)} crossings, "
            f"{state.rune_count} runes, {len(solutions)} solutions"
            f"{' (sure)' if best and best.sure else ''}, {elapsed:.0f} ms: {verdict}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
