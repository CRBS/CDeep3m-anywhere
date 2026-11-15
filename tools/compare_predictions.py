#!/usr/bin/env python3
"""Pixel comparison of two CDeep3M predictions of the same images.

  compare_predictions.py --a <dir A> --b <dir B> --out <dir> [--labels <dir>] [--images <dir>]
                         [--name-a PyTorch --name-b Caffe] [--threshold 127]

<dir A>/<dir B> hold the probability maps (Segmented_*.png, 0..255) written by runprediction.sh
(the 'ensembled' folder). Files are paired in sorted order. Writes
  <out>/per_image.tsv   one row per image
  <out>/summary.txt     totals over all images
  <out>/side_by_side/   image | A | B | difference (white: both, red: only A, blue: only B)
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def load(pattern):
    files = sorted(glob.glob(pattern))
    return files, [np.array(Image.open(f).convert("L")) for f in files]


def dice(x, y):
    s = x.sum() + y.sum()
    return 2.0 * (x & y).sum() / s if s else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", help="folder of label PNGs (same number and order as the images)")
    ap.add_argument("--images", help="folder of the input PNGs, for the side-by-side pictures")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--threshold", type=int, default=127, help="probability (0..255) above which a pixel is foreground")
    a = ap.parse_args()

    fa, pa = load(os.path.join(a.a, "Segmented_*.png"))
    fb, pb = load(os.path.join(a.b, "Segmented_*.png"))
    if not pa or len(pa) != len(pb):
        raise SystemExit("need the same number of Segmented_*.png in both folders (found %d and %d)" % (len(pa), len(pb)))
    gt = im = None
    if a.labels:
        _, gt = load(os.path.join(a.labels, "*.png"))
        if len(gt) != len(pa):
            raise SystemExit("%d labels but %d predictions" % (len(gt), len(pa)))
        gt = [g > g.max() / 2.0 if g.max() else g > 0 for g in gt]
    if a.images:
        _, im = load(os.path.join(a.images, "*.png"))
        if len(im) != len(pa):
            raise SystemExit("%d images but %d predictions" % (len(im), len(pa)))

    os.makedirs(os.path.join(a.out, "side_by_side"), exist_ok=True)
    cols = ["image", "dice_%s_vs_%s" % (a.name_a, a.name_b), "pixels_disagree_pct", "prob_mean_abs_diff", "prob_max_abs_diff",
            "foreground_pct_%s" % a.name_a, "foreground_pct_%s" % a.name_b]
    if gt:
        cols += ["foreground_pct_label", "dice_%s_vs_label" % a.name_a, "dice_%s_vs_label" % a.name_b]
    rows = []
    tot = dict(ab=0, a=0, b=0, dis=0, n=0, absdiff=0.0, maxdiff=0, ag=0, bg=0, g=0)
    for i, (x, y) in enumerate(zip(pa, pb)):
        if x.shape != y.shape:
            raise SystemExit("size mismatch at %s" % fa[i])
        ma, mb = x > a.threshold, y > a.threshold
        d = np.abs(x.astype(np.int16) - y.astype(np.int16))
        row = [os.path.basename(fa[i]), "%.4f" % dice(ma, mb), "%.3f" % (100.0 * (ma != mb).mean()),
               "%.5f" % (d.mean() / 255.0), "%.4f" % (d.max() / 255.0), "%.2f" % (100.0 * ma.mean()), "%.2f" % (100.0 * mb.mean())]
        tot["ab"] += (ma & mb).sum(); tot["a"] += ma.sum(); tot["b"] += mb.sum(); tot["dis"] += (ma != mb).sum()
        tot["n"] += ma.size; tot["absdiff"] += d.sum() / 255.0; tot["maxdiff"] = max(tot["maxdiff"], int(d.max()))
        if gt:
            g = gt[i]
            row += ["%.2f" % (100.0 * g.mean()), "%.4f" % dice(ma, g), "%.4f" % dice(mb, g)]
            tot["ag"] += (ma & g).sum(); tot["bg"] += (mb & g).sum(); tot["g"] += g.sum()
        rows.append(row)

        diff = np.zeros(x.shape + (3,), np.uint8)
        diff[ma & mb] = (255, 255, 255)
        diff[ma & ~mb] = (255, 0, 0)
        diff[~ma & mb] = (0, 0, 255)
        panels = ([np.stack([im[i]] * 3, -1)] if im else []) + [np.stack([x] * 3, -1), np.stack([y] * 3, -1), diff]
        gap = np.full((x.shape[0], 8, 3), 128, np.uint8)
        strip = []
        for p in panels:
            strip += [p, gap]
        Image.fromarray(np.concatenate(strip[:-1], 1)).save(os.path.join(a.out, "side_by_side", "compare_%04d.png" % (i + 1)))

    with open(os.path.join(a.out, "per_image.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    lines = ["%d images, threshold %d/255" % (len(pa), a.threshold),
             "Dice %s vs %s:            %.4f" % (a.name_a, a.name_b, 2.0 * tot["ab"] / max(tot["a"] + tot["b"], 1)),
             "pixels that disagree:      %.3f %%" % (100.0 * tot["dis"] / tot["n"]),
             "probability difference:    mean %.5f, max %.4f (scale 0..1)" % (tot["absdiff"] / tot["n"], tot["maxdiff"] / 255.0),
             "foreground:                %s %.2f %%, %s %.2f %%" % (a.name_a, 100.0 * tot["a"] / tot["n"], a.name_b, 100.0 * tot["b"] / tot["n"])]
    if gt:
        lines += ["foreground in the labels:  %.2f %%" % (100.0 * tot["g"] / tot["n"]),
                  "Dice %s vs labels:        %.4f" % (a.name_a, 2.0 * tot["ag"] / max(tot["a"] + tot["g"], 1)),
                  "Dice %s vs labels:        %.4f" % (a.name_b, 2.0 * tot["bg"] / max(tot["b"] + tot["g"], 1))]
    lines.append("side-by-side panels: %sprediction %s | prediction %s | difference (white both, red only %s, blue only %s)"
                 % ("image | " if im else "", a.name_a, a.name_b, a.name_a, a.name_b))
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
