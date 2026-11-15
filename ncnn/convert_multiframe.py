#!/usr/bin/env python3
"""Rewrite a CDeep3M 3fm/5fm deploy network as an equivalent plain 2D network for ncnn.

  convert_multiframe.py <deploy.prototxt> <model.caffemodel> <out_dir> [--proto caffe.proto]

The multi-frame models start with two 3D convolutions over a stack of F neighbouring slices
(3fm: F=3, 5fm: F=5) that reduce the depth to 1; everything after the Reshape is 2D.
Both 3D layers are rewritten exactly as 2D convolutions with the slices as input channels:

  conv1_1  kernel 3x3x3, stride 2, pad 1  ->  2D conv, F input channels, D1*32 output channels,
           where D1 = (F-1)//2 + 1 is the depth left after conv1_1 and
           W'[d*32+o, p] = W[o, kz = p - (2*d - 1)]  (zero outside the kernel).
           The BatchNorm/Scale after it act per channel, so their parameters are tiled D1 times.
  conv1_2  kernel 3x3xD1, pad z 0  ->  2D conv over D1*32 channels,
           W'[o, d*32+c] = W[o, c, kz = d].

Writes <out_dir>/deploy2d.prototxt and <out_dir>/model2d.caffemodel for caffe2ncnn.
Needs the python protobuf package and caffe.proto from the CDeep3M Caffe fork
(--proto, or set CAFFE_PROTO).
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np


def load_caffe_pb2(proto):
    tmp = tempfile.mkdtemp()
    subprocess.run(["protoc", "-I", os.path.dirname(proto), "--python_out=" + tmp, proto], check=True)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    sys.path.insert(0, tmp)
    import caffe_pb2
    return caffe_pb2


def layer_span(text, name):
    """(start, end) of the 'layer { ... name: "<name>" ... }' block, with brace matching."""
    m = re.search(r'name\s*:\s*"%s"' % re.escape(name), text)
    if not m:
        sys.exit("layer %s not found in the prototxt" % name)
    start = text.rfind("layer", 0, m.start())
    depth, i = 0, text.index("{", start)
    while True:
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deploy"); ap.add_argument("caffemodel"); ap.add_argument("out")
    ap.add_argument("--proto", default=os.environ.get("CAFFE_PROTO"))
    a = ap.parse_args()
    if not a.proto:
        sys.exit("give caffe.proto of the CDeep3M Caffe fork with --proto or CAFFE_PROTO")
    os.makedirs(a.out, exist_ok=True)

    text = "\n".join(l for l in open(a.deploy).read().splitlines() if not l.lstrip().startswith("#"))
    dims = [int(x) for x in re.search(r"input_shape\s*\{([^}]*)\}", text).group(1).replace("dim:", " ").split()]
    n, c, h, w, F = dims
    D1 = (F - 1) // 2 + 1
    print("frames F=%d, depth after conv1_1 D1=%d" % (F, D1))

    # ---- prototxt: plain 2D
    text = re.sub(r"^\s*phase\s*:\s*PREDICT\s*$", "", text, flags=re.M)
    s, e = layer_span(text, "reshape")                       # in-place 4D reshape, not needed any more
    text = text[:s] + text[e:]
    text = re.sub(r"input_shape\s*\{[^}]*\}", "input_shape {\n dim: %d\n dim: %d\n dim: %d\n dim: %d\n}" % (n, F, h, w), text, count=1)
    for name, fix in (("conv1_1", lambda b: re.sub(r"num_output\s*:\s*32", "num_output: %d" % (32 * D1), b)),
                      ("conv1_2", lambda b: b)):
        s, e = layer_span(text, name)
        b = fix(text[s:e])
        b = re.sub(r"(kernel_size\s*:\s*\d+)\s+(kernel_size\s*:\s*\d+)\s+kernel_size\s*:\s*\d+", r"\1 \2", b)
        b = re.sub(r"(pad\s*:\s*\d+)\s+(pad\s*:\s*\d+)\s+pad\s*:\s*\d+", r"\1 \2", b)
        text = text[:s] + b + text[e:]
    open(os.path.join(a.out, "deploy2d.prototxt"), "w").write(text)

    # ---- weights
    pb = load_caffe_pb2(a.proto)
    net = pb.NetParameter()
    net.ParseFromString(open(a.caffemodel, "rb").read())
    L = {l.name: l for l in net.layer}

    def get(layer, i):
        b = L[layer].blobs[i]
        return np.array(b.data, dtype=np.float32).reshape([int(x) for x in b.shape.dim])

    def put(layer, i, arr):
        b = L[layer].blobs[i]
        del b.shape.dim[:]; b.shape.dim.extend(arr.shape)
        del b.data[:]; b.data.extend(arr.astype(np.float32).ravel().tolist())

    W1 = get("conv1_1", 0)                                   # (32, 1, kh, kw, kz)
    assert W1.shape[1] == 1 and W1.shape[2:4] == (3, 3), W1.shape
    kz1 = W1.shape[4]
    W1n = np.zeros((D1 * 32, F, 3, 3), np.float32)
    for d in range(D1):
        for p in range(F):
            kz = p - (2 * d - 1)                              # stride 2, pad 1 in z
            if 0 <= kz < kz1:
                W1n[d * 32:(d + 1) * 32, p] = W1[:, 0, :, :, kz]
    put("conv1_1", 0, W1n)
    put("conv1_1", 1, np.tile(get("conv1_1", 1), D1))
    for layer in ("bn_conv1_1", "scale_conv1_1"):
        for i in range(len(L[layer].blobs)):
            arr = get(layer, i)
            if arr.size == 32:
                put(layer, i, np.tile(arr, D1))
    W2 = get("conv1_2", 0)                                    # (64, 32, 3, 3, kz)
    assert W2.shape[4] == D1, "conv1_2 depth %d != D1 %d" % (W2.shape[4], D1)
    put("conv1_2", 0, np.concatenate([W2[:, :, :, :, d] for d in range(D1)], axis=1))
    open(os.path.join(a.out, "model2d.caffemodel"), "wb").write(net.SerializeToString())
    print("wrote %s/deploy2d.prototxt and model2d.caffemodel" % a.out)


if __name__ == "__main__":
    main()
