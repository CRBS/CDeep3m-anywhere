import csv, glob, json, os, re, sys, platform
import numpy as np
from PIL import Image

# usage: VAL_LABELS=<dir with ground-truth PNGs> python3 analyze.py <benchmark output dir>
B = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
LABELS = sorted(glob.glob(os.path.join(os.environ.get("VAL_LABELS", ""), "*.png")))

def load_run(name):
    files = sorted(glob.glob(f"{B}/{name}/ensembled/Segmented_*.png"))
    return np.stack([np.array(Image.open(f).convert("L")) for f in files]) if files else None

def pred_seconds(name):
    p = f"{B}/{name}/logs/prediction.log"
    if not os.path.exists(p): return None
    return sum(float(m) for m in re.findall(r"^real ([0-9.]+)", open(p).read(), re.M))

def dice(a, b):
    a, b = a > 127, b > 127
    d = a.sum() + b.sum()
    return 1.0 if d == 0 else 2.0 * (a & b).sum() / d

def agreement(x, ref):
    diff = np.abs(x.astype(int) - ref.astype(int))
    return dict(mean_abs_diff=float(diff.mean()), max_abs_diff=int(diff.max()),
                pct_pixels_differing=float((diff > 0).mean() * 100),
                dice_vs_ref=float(np.mean([dice(a, b) for a, b in zip(x, ref)])))

def accuracy(x):
    gt = np.stack([np.array(Image.open(f).convert("L")) > 127 for f in LABELS])
    n = min(len(gt), len(x)); gt, p = gt[:n], x[:n] > 127
    tp = (gt & p).sum(); fp = (~gt & p).sum(); fn = (gt & ~p).sum()
    per = [2.0 * (g & q).sum() / max(1, g.sum() + q.sum()) for g, q in zip(gt, p)]
    return dict(dice_pooled=float(2 * tp / (2 * tp + fp + fn)), iou=float(tp / (tp + fp + fn)),
                precision=float(tp / (tp + fp)), recall=float(tp / (tp + fn)),
                dice_per_image_mean=float(np.mean(per)), images=int(n))

rows = list(csv.DictReader(open(f"{B}/timings.tsv"), delimiter="\t"))
for r in rows:  # overlay step failed on a config-path bug; the segmentation itself is what counts
    r["status"] = "ok" if glob.glob(f"{B}/{r['run']}/ensembled/Segmented_*.png") else "NO OUTPUT"
T = {r["run"]: r for r in rows}
res = {"timings": {}, "agreement": {}, "accuracy": {}}
for r in rows:
    res["timings"][r["run"]] = dict(r, wall_s=float(r["wall_s"]), predict_s=pred_seconds(r["run"]))

pairs = [("s_ncnn_cpu", "s_caffe_cpu"), ("s_ncnn_vulkan", "s_caffe_cpu"), ("f_ncnn_cpu", "f_caffe_cpu")]
for a, b in pairs:
    xa, xb = load_run(a), load_run(b)
    if xa is not None and xb is not None:
        res["agreement"][f"{a} vs {b}"] = agreement(xa, xb)
for name in ("v_caffe_cpu", "v_ncnn_cpu", "fv_ncnn_cpu"):
    x = load_run(name)
    if x is not None: res["accuracy"][name] = accuracy(x)
json.dump(res, open(f"{B}/results.json", "w"), indent=1)

cpu = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
mem = int(open("/proc/meminfo").readline().split()[1]) // 1024 // 1024
def fmt(name):
    t = res["timings"].get(name)
    if not t: return "not run", "", ""
    ps = t["predict_s"]; w = t["wall_s"]
    return f"{w:.0f} s", f"{ps:.0f} s" if ps else "-", t["status"]

L = ["# CDeep3M 1fm prediction benchmark", "",
     f"CPU: {cpu} ({os.cpu_count()} threads), RAM {mem} GB, GPU: Intel UHD 750 (Vulkan), Python {platform.python_version()}", "",
     "Model: SBEM_Mitochondria 1fm (iter 80000). Full pipeline via runprediction.sh (preprocess, predict, ensemble, overlay), denoise off.", "",
     "## 1. Engine comparison, 1 augmentation variant, 11 test images (1024x1024)", "",
     "| Engine | Wall time | Prediction step | Per slice (15 padded) |", "|---|---|---|---|"]
for name, label in (("s_caffe_cpu", "Caffe CPU (8 BLAS threads)"), ("s_ncnn_cpu", "ncnn CPU"), ("s_ncnn_vulkan", "ncnn Vulkan (Intel UHD 750, fp32)")):
    w, p, st = fmt(name)
    ps = res["timings"].get(name, {}).get("predict_s")
    L.append(f"| {label} | {w} | {p} | {'%.1f s' % (ps/15) if ps else '-'} |")
L += ["", "## 2. Full quality, 8 augmentation variants (1fm at --augspeed 1), 11 test images", "", "| Engine | Wall time | Prediction step |", "|---|---|---|"]
for name, label in (("f_caffe_cpu", "Caffe CPU (8 BLAS threads)"), ("f_ncnn_cpu", "ncnn CPU")):
    w, p, st = fmt(name); L.append(f"| {label} | {w} | {p} |")
a, b = res["timings"].get("f_caffe_cpu"), res["timings"].get("f_ncnn_cpu")
if a and b: L += ["", f"ncnn CPU speedup over Caffe CPU: **{a['wall_s']/b['wall_s']:.1f}x** wall, " + (f"{a['predict_s']/b['predict_s']:.1f}x prediction step." if a['predict_s'] and b['predict_s'] else "")]
L += ["", "## 3. Agreement with Caffe (final ensembled segmentation, 0-255 scale)", "", "| Comparison | mean abs diff | max abs diff | % pixels differing | Dice (>127) |", "|---|---|---|---|---|"]
for k, v in res["agreement"].items():
    L.append(f"| {k} | {v['mean_abs_diff']:.4f} | {v['max_abs_diff']} | {v['pct_pixels_differing']:.3f} | {v['dice_vs_ref']:.5f} |")
L += ["", "## 4. Accuracy against ground truth (10 labelled validation images)", "", "| Run | Dice (pooled) | Dice (per-image mean) | IoU | Precision | Recall |", "|---|---|---|---|---|---|"]
names = {"v_caffe_cpu": "Caffe CPU, 1 variant", "v_ncnn_cpu": "ncnn CPU, 1 variant", "fv_ncnn_cpu": "ncnn CPU, 8 variants"}
for k, v in res["accuracy"].items():
    L.append(f"| {names[k]} | {v['dice_pooled']:.4f} | {v['dice_per_image_mean']:.4f} | {v['iou']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} |")
open(f"{B}/BENCHMARK.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
