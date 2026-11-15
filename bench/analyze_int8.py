import csv, glob, os, sys
import numpy as np
from PIL import Image
# usage: VAL_LABELS=<dir with ground-truth PNGs> python3 analyze_int8.py <benchmark output dir>
B = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
LABELS = sorted(glob.glob(os.path.join(os.environ.get("VAL_LABELS", ""), "*.png")))
def load(name):
    fs = sorted(glob.glob(f"{B}/{name}/ensembled/Segmented_*.png"))
    return np.stack([np.array(Image.open(f).convert("L")) for f in fs])
def dice(a, b):
    a, b = a > 127, b > 127; d = a.sum() + b.sum()
    return 1.0 if d == 0 else 2.0 * (a & b).sum() / d
def acc(x):
    gt = np.stack([np.array(Image.open(f).convert("L")) > 127 for f in LABELS])[:len(x)]; p = x[:len(gt)] > 127
    tp = (gt & p).sum(); fp = (~gt & p).sum(); fn = (gt & ~p).sum()
    return dict(dice=2*tp/(2*tp+fp+fn), iou=tp/(tp+fp+fn), precision=tp/(tp+fp), recall=tp/(tp+fn))
T = {}
for r in csv.DictReader(open(f"{B}/timings.tsv"), delimiter="\t"): T[r["run"]] = float(r["wall_s"])
out = ["# ncnn fp32 vs ncnn int8 (CPU), 1fm model", "",
       "Calibration: 20 labelled training images (KL method), 20 real slices. Model size: fp32 96.2 MB, int8 31.8 MB.", "",
       "## Speed (wall time of the full pipeline)", "", "| Job | fp32 | int8 | Speedup |", "|---|---|---|---|"]
def row(label, f, i):
    f = np.mean([T[k] for k in f if k in T]); i = np.mean([T[k] for k in i if k in T]); out.append(f"| {label} | {f:.0f} s | {i:.0f} s | **{f/i:.2f}x** |")
row("10 validation images, 1 variant", ["val1_fp32"], ["val1_int8"])
row("10 validation images, 8 variants", ["val8_fp32"], ["val8_int8"])
row("11 test images, 8 variants (mean of 2 runs)", ["test8_fp32", "test8_fp32b"], ["test8_int8", "test8_int8b"])
out += ["", "## Accuracy against ground truth (10 labelled validation images)", "", "| Run | Dice | IoU | Precision | Recall |", "|---|---|---|---|---|"]
for k, l in (("val1_fp32", "fp32, 1 variant"), ("val1_int8", "int8, 1 variant"), ("val8_fp32", "fp32, 8 variants"), ("val8_int8", "int8, 8 variants")):
    a = acc(load(k)); out.append(f"| {l} | {a['dice']:.4f} | {a['iou']:.4f} | {a['precision']:.4f} | {a['recall']:.4f} |")
out += ["", "## Agreement of int8 with fp32 (final ensembled segmentation, 0-255 scale)", "", "| Job | mean abs diff | max abs diff | % pixels differing by >8 | Dice between them |", "|---|---|---|---|---|"]
for l, f, i in (("validation, 1 variant", "val1_fp32", "val1_int8"), ("validation, 8 variants", "val8_fp32", "val8_int8"), ("test set, 8 variants", "test8_fp32", "test8_int8")):
    a, b = load(f), load(i); d = np.abs(a.astype(int) - b.astype(int))
    out.append(f"| {l} | {d.mean():.3f} | {d.max()} | {(d > 8).mean()*100:.3f} | {np.mean([dice(x, y) for x, y in zip(a, b)]):.4f} |")
open(f"{B}/INT8_REPORT.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
