#!/usr/bin/env python3
"""Make int8 calibration data for convert_model.sh.

  make_calibration.py --repo <patched CDeep3M2 checkout> --images <folder of PNG slices> --out <dir>

Runs the CDeep3M2 preprocessing on the images so the network sees exactly what it sees in
the pipeline, then writes the real (non-padding) slices as float32 .npy files of shape
(frames, H, W), the window of neighbouring slices the model takes plus <out>/list.txt for ncnn2table. Use training images, not the images you
evaluate on.
"""
import argparse, glob, os, subprocess, sys, tempfile
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="1fm", help="1fm, 3fm or 5fm (sets the number of frames)")
    a = ap.parse_args()
    images, out = os.path.abspath(a.images), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        aug = os.path.join(tmp, "aug")
        subprocess.run([sys.executable, "DefDataPackages.py", images, aug], cwd=a.repo, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "PreprocessPackage.py", images, aug, "1", "1", a.model, "10"], cwd=a.repo, check=True, stdout=subprocess.DEVNULL)
        h5 = glob.glob(os.path.join(aug, a.model, "Pkg001_Z01", "image_stacks_v1.h5"))[0]
        data = h5py.File(h5, "r")["data"][()]          # (H, W, Z) with 2+2 padding planes
    frames = int(a.model[0])                        # 1fm -> 1 channel, 3fm -> 3, 5fm -> 5
    names = []
    for i in range(2, data.shape[2] - 2):           # real slices only; their windows are fully inside the stack
        p = os.path.join(out, "slice%03d.npy" % i)
        window = data[:, :, i - frames // 2:i + frames // 2 + 1].astype(np.float32).transpose(2, 0, 1)
        np.save(p, np.ascontiguousarray(window))
        names.append(p)
    open(os.path.join(out, "list.txt"), "w").write("\n".join(names) + "\n")
    print("wrote %d calibration slices and %s" % (len(names), os.path.join(out, "list.txt")))


if __name__ == "__main__":
    main()
