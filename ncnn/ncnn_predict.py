#!/usr/bin/env python3
"""ncnn replacement for caffe predict_seg_new.bin (1-frame models).

Reads image_stacks_vN.h5 (H, W, D uint8), runs the network on every z slice and
writes <predict>_shift_<i>.h5 with dataset 'data' of shape (1, 2, H, W), the same
files predict_seg_new.bin produces, so the existing merge steps are unchanged.
"""
import argparse, os, sys, time
import h5py
import numpy as np
import ncnn


def resolve_precision(requested, device, model_dir):
    """Return (precision, dir with model.param/model.bin).

    requested None means the default: int8 when an int8 model exists and the device is the CPU,
    otherwise fp32 (with a note). An explicit request that cannot be met is an error.
    """
    int8_dir = os.path.join(model_dir, "int8")
    have_int8 = os.path.isfile(os.path.join(int8_dir, "model.param"))
    if requested == "int8":
        if device != "cpu":
            sys.exit("int8 is only supported with --device cpu (not tested on Vulkan)")
        if not have_int8:
            sys.exit("no int8 model in %s; run convert_model.sh with a calibration list" % int8_dir)
        return "int8", int8_dir
    if requested is None:
        if device == "cpu" and have_int8:
            return "int8", int8_dir
        print("note: using fp32 (%s)" % ("int8 is CPU only" if device != "cpu" else "no int8 model in " + int8_dir),
              file=sys.stderr, flush=True)
    return "fp32", model_dir


def read_frames(model_dir):
    """Number of input channels (= neighbouring slices) from the Input layer in model.param."""
    with open(os.path.join(model_dir, "model.param")) as f:
        for line in f.read().splitlines()[2:]:
            tok = line.split()
            if tok and tok[0] == "Input":
                for t in tok:
                    if t.startswith("2="):
                        return int(t[2:])
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="dir with model.param and model.bin")
    ap.add_argument("--data", required=True, help="input image_stacks_vN.h5")
    ap.add_argument("--predict", required=True, help="output prefix, e.g. <dir>/test.h5")
    ap.add_argument("--device", default="cpu", choices=["cpu", "vulkan"])
    ap.add_argument("--precision", default=None, choices=["fp32", "int8"],
                    help="default: int8 if <model-dir>/int8 exists and --device cpu, otherwise fp32. "
                         "An explicit int8 fails if that is not possible. int8 is CPU only.")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all)")
    ap.add_argument("--skip-edge", type=int, default=0,
                    help="do not predict the first and last N slices (z padding the merge step discards, "
                         "2 for the 1-frame model). Tiny placeholder files keep the slice numbering unchanged.")
    a = ap.parse_args()

    precision, model_dir = resolve_precision(a.precision, a.device, a.model_dir)
    data = h5py.File(a.data, "r")["data"][()]
    net = ncnn.Net()
    if a.device == "vulkan":
        net.opt.use_vulkan_compute = True
        net.set_vulkan_device(a.gpu_index)
        # fp32 end to end: fp16 changed the output slightly on Intel UHD 750
        net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = False
    else:
        net.opt.use_vulkan_compute = False
        if a.threads > 0:
            net.opt.num_threads = a.threads
    net.load_param(os.path.join(model_dir, "model.param"))
    net.load_model(os.path.join(model_dir, "model.bin"))

    t_all = time.time()
    depth = data.shape[2]
    frames = read_frames(model_dir)
    if 2 * a.skip_edge >= depth:
        sys.exit("--skip-edge %d leaves no slices to predict in a stack of %d" % (a.skip_edge, depth))
    for i in range(depth):
        if i < a.skip_edge or i >= depth - a.skip_edge:
            with h5py.File("%s_shift_%d.h5" % (a.predict, i), "w") as f:   # placeholder, never read
                f.create_dataset("data", data=np.zeros((1, 2, 1, 1), dtype=np.float32))
            print("slice %d skipped (z padding)" % i, flush=True)
            continue
        t = time.time()
        # frames F: channels are the slices i-F//2 .. i+F//2 (same window as the Caffe shift);
        # indices outside the stack are clamped (only matters for the padding shifts the merge discards)
        planes = [min(max(i - frames // 2 + j, 0), depth - 1) for j in range(frames)]
        x = np.ascontiguousarray(data[:, :, planes].astype(np.float32).transpose(2, 0, 1))
        ex = net.create_extractor()
        ex.input("data", ncnn.Mat(x).clone())
        ret, out = ex.extract("prob")
        if ret != 0:
            sys.exit("ncnn extract failed on slice %d" % i)
        with h5py.File("%s_shift_%d.h5" % (a.predict, i), "w") as f:
            f.create_dataset("data", data=np.array(out)[None].astype(np.float32))
        print("slice %d done in %.2fs" % (i, time.time() - t), flush=True)
    print("ncnn %s %s total %.2fs for %d slices (%d predicted)" % (a.device, precision, time.time() - t_all, depth, depth - 2 * a.skip_edge))


if __name__ == "__main__":
    main()
