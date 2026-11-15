#!/usr/bin/env python3
"""Make int8 calibration data for convert_model.sh.

  make_calibration.py --repo <patched CDeep3M2 checkout> --images <folder of PNG slices> --out <dir>

Runs the CDeep3M2 preprocessing on the images so the network sees exactly what it sees in
the pipeline, then writes the real (non-padding) slices as float32 .npy files of shape
(1, H, W) plus <out>/list.txt for ncnn2table. Use training images, not the images you
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
    ap.add_argument("--model", default="1fm")
    a = ap.parse_args()
    images, out = os.path.abspath(a.images), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        aug = os.path.join(tmp, "aug")
        subprocess.run([sys.executable, "DefDataPackages.py", images, aug], cwd=a.repo, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "PreprocessPackage.py", images, aug, "1", "1", a.model, "10"], cwd=a.repo, check=True, stdout=subprocess.DEVNULL)
        h5 = glob.glob(os.path.join(aug, a.model, "Pkg001_Z01", "image_stacks_v1.h5"))[0]
        data = h5py.File(h5, "r")["data"][()]          # (H, W, Z) with 2+2 padding planes
    names = []
    for i in range(2, data.shape[2] - 2):
        p = os.path.join(out, "slice%03d.npy" % i)
        np.save(p, np.ascontiguousarray(data[:, :, i].astype(np.float32))[None])
        names.append(p)
    open(os.path.join(out, "list.txt"), "w").write("\n".join(names) + "\n")
    print("wrote %d calibration slices and %s" % (len(names), os.path.join(out, "list.txt")))


if __name__ == "__main__":
    main()
