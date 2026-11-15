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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="dir with model.param and model.bin")
    ap.add_argument("--data", required=True, help="input image_stacks_vN.h5")
    ap.add_argument("--predict", required=True, help="output prefix, e.g. <dir>/test.h5")
    ap.add_argument("--device", default="cpu", choices=["cpu", "vulkan"])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "int8"],
                    help="int8 uses <model-dir>/int8 (see convert_model.sh); CPU only")
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all)")
    a = ap.parse_args()

    model_dir = a.model_dir
    if a.precision == "int8":
        if a.device != "cpu":
            sys.exit("int8 is only supported with --device cpu (not tested on Vulkan)")
        model_dir = os.path.join(a.model_dir, "int8")
        if not os.path.isfile(os.path.join(model_dir, "model.param")):
            sys.exit("no int8 model in %s; run convert_model.sh with a calibration list" % model_dir)
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
    for i in range(data.shape[2]):
        t = time.time()
        x = np.ascontiguousarray(data[:, :, i].astype(np.float32))[None]
        ex = net.create_extractor()
        ex.input("data", ncnn.Mat(x).clone())
        ret, out = ex.extract("prob")
        if ret != 0:
            sys.exit("ncnn extract failed on slice %d" % i)
        with h5py.File("%s_shift_%d.h5" % (a.predict, i), "w") as f:
            f.create_dataset("data", data=np.array(out)[None].astype(np.float32))
        print("slice %d done in %.2fs" % (i, time.time() - t), flush=True)
    print("ncnn %s %s total %.2fs for %d slices" % (a.device, a.precision, time.time() - t_all, data.shape[2]))


if __name__ == "__main__":
    main()
