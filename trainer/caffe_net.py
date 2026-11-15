"""Build a CDeep3M network (1fm, 3fm or 5fm) in PyTorch from its Caffe deploy.prototxt.

The network is the plain 2D inception-residual FCN after the first two layers. The 3fm/5fm models start
with two 3D convolutions over F neighbouring slices; they are implemented exactly (shared kernels, tiled
BatchNorm) so that the parameters stay those of the original Caffe model and can be written back to it.

Layer names are the Caffe layer names, so weights are loaded/exported by name.
"""
import math
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ protobuf text parser
_TOKEN = re.compile(r"""\s*(?:\#[^\n]*|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|([{}:])|([^\s{}:#"']+))""")


def _tokens(text):
    out, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == "":
                break
            raise ValueError("cannot parse prototxt near: %r" % text[pos:pos + 40])
        pos = m.end()
        if m.group(1) is not None:
            out.append(("str", m.group(1)[1:-1]))
        elif m.group(2) is not None:
            out.append(("sym", m.group(2)))
        elif m.group(3) is not None:
            out.append(("word", m.group(3)))
    return out


def _convert(word):
    for conv in (int, float):
        try:
            return conv(word)
        except ValueError:
            pass
    return word


def _block(toks, i):
    d = {}
    while i < len(toks) and toks[i] != ("sym", "}"):
        kind, key = toks[i]
        i += 1
        if toks[i] == ("sym", ":"):
            i += 1
        if toks[i] == ("sym", "{"):
            val, i = _block(toks, i + 1)
            i += 1                                   # closing brace
        else:
            k2, v2 = toks[i]
            val = v2 if k2 == "str" else _convert(v2)
            i += 1
        d.setdefault(key, []).append(val)
    return d, i


def parse_prototxt(path):
    text = open(path).read()
    d, _ = _block(_tokens(text), 0)
    return d


def _one(d, key, default=None):
    v = d.get(key)
    return v[0] if v else default


def _pair(p, base, hkey, wkey, default):
    """(h, w) from a repeated field (first two entries) or explicit _h/_w fields."""
    if hkey in p or wkey in p:
        return int(_one(p, hkey, default)), int(_one(p, wkey, default))
    v = p.get(base)
    if not v:
        return default, default
    return int(v[0]), int(v[1] if len(v) > 1 else v[0])


def geometry(p):
    k = _pair(p, "kernel_size", "kernel_h", "kernel_w", 1)
    s = _pair(p, "stride", "stride_h", "stride_w", 1)
    pd = _pair(p, "pad", "pad_h", "pad_w", 0)
    dil = int(_one(p, "dilation", 1))
    return k, s, pd, dil


# ------------------------------------------------------------------ special layers
class MultiFrameConv1(nn.Module):
    """conv1_1 of 3fm/5fm: 3x3xkz conv over F slices, stride s, pad_z, as an exact 2D convolution.

    Parameters keep the Caffe layout (out, 1, kh, kw, kz). The effective 2D weight has D1*out output
    channels (depth position major) and F input channels; D1 = (F + 2*pad_z - kz)//stride + 1.
    """

    def __init__(self, frames, out=32, k=3, kz=3, stride=2, pad=1, pad_z=1):
        super().__init__()
        self.out, self.k, self.stride, self.pad = out, k, stride, pad
        self.frames = frames
        self.depth = (frames + 2 * pad_z - kz) // stride + 1
        self.weight = nn.Parameter(torch.zeros(out, 1, k, k, kz))
        self.bias = nn.Parameter(torch.zeros(out))
        idx = torch.full((self.depth, frames), kz, dtype=torch.long)       # kz -> the zero column
        for d in range(self.depth):
            for p in range(frames):
                kzi = p - (stride * d - pad_z)
                if 0 <= kzi < kz:
                    idx[d, p] = kzi
        self.register_buffer("idx", idx, persistent=False)

    def effective_weight(self):
        w = F.pad(self.weight[:, 0], (0, 1))                    # (out, k, k, kz+1)
        sel = w[:, :, :, self.idx]                               # (out, k, k, D1, F)
        return sel.permute(3, 0, 4, 1, 2).reshape(self.depth * self.out, self.frames, self.k, self.k)

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), self.bias.repeat(self.depth),
                        stride=self.stride, padding=self.pad)


class TiledBN(nn.Module):
    """BatchNorm(+Scale) with `channels` parameters applied to a (N, depth*channels, H, W) tensor
    (depth major), i.e. statistics shared across the depth positions as in the 3D Caffe model."""

    def __init__(self, channels, depth):
        super().__init__()
        self.channels, self.depth = channels, depth
        self.bn = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.001)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.bn(x.reshape(n * self.depth, self.channels, h, w))
        return y.reshape(n, self.depth * self.channels, h, w)


# ------------------------------------------------------------------ the network
class CaffeNet(nn.Module):
    def __init__(self, deploy_prototxt):
        super().__init__()
        net = parse_prototxt(deploy_prototxt)
        dims = [int(x) for x in net["input_shape"][0]["dim"]]
        self.frames = dims[-1]
        self.size = (dims[2], dims[3])
        self.input_name = net["input"][0]
        self.layers = []                     # execution plan
        self.mods = nn.ModuleDict()
        self.lr_mult = {}                    # parameter name -> (lr_mult, decay_mult)
        chans = {self.input_name: self.frames}
        raw = [l for l in net["layer"]]
        i = 0
        while i < len(raw):
            L = raw[i]
            i += 1
            name, typ = _one(L, "name"), _one(L, "type")
            bottoms, tops = L.get("bottom", []), L.get("top", [])
            mults = [(float(_one(p, "lr_mult", 1)), float(_one(p, "decay_mult", 1))) for p in L.get("param", [])]
            if typ == "Reshape":
                continue                                       # depth is already folded into channels
            if typ == "Convolution" or typ == "Deconvolution":
                cp = L["convolution_param"][0]
                (kh, kw), (sh, sw), (ph, pw), dil = geometry(cp)
                out = int(_one(cp, "num_output"))
                cin = chans[bottoms[0]]
                special = typ == "Convolution" and self.frames > 1 and name == "conv1_1"
                if special:
                    kz = [int(v) for v in cp["kernel_size"]][-1]
                    pad_z = [int(v) for v in cp["pad"]][-1]
                    m = MultiFrameConv1(self.frames, out=out, k=kh, kz=kz, stride=sh, pad=ph, pad_z=pad_z)
                    cout = out * m.depth
                    self.depth1 = m.depth
                elif typ == "Convolution":
                    if self.frames > 1 and name == "conv1_2":
                        cin = chans[bottoms[0]]                   # D1*32 channels
                    m = nn.Conv2d(cin, out, (kh, kw), (sh, sw), (ph, pw), dilation=dil)
                    cout = out
                else:
                    m = nn.ConvTranspose2d(cin, out, (kh, kw), (sh, sw), (ph, pw), dilation=dil)
                    cout = out
                self._init(m, cp)
                self.mods[name] = m
                self.lr_mult[name] = mults
                chans[tops[0]] = cout
                self.layers.append((name, typ, bottoms, tops, None))
            elif typ == "BatchNorm":
                nxt = raw[i] if i < len(raw) else None
                assert nxt is not None and _one(nxt, "type") == "Scale", "BatchNorm without Scale: " + name
                scale_name = _one(nxt, "name")
                i += 1
                c = chans[bottoms[0]]
                if self.frames > 1 and bottoms[0] == "conv1_1":
                    m = TiledBN(c // self.depth1, self.depth1)
                else:
                    m = nn.BatchNorm2d(c, eps=1e-5, momentum=0.001)
                self.mods[name] = m
                self.bn_scale = getattr(self, "bn_scale", {})
                self.bn_scale[name] = scale_name
                smults = [(float(_one(p, "lr_mult", 1)), float(_one(p, "decay_mult", 1))) for p in nxt.get("param", [])]
                self.lr_mult[name] = smults or [(1.0, 1.0), (1.0, 1.0)]
                chans[tops[0]] = c
                self.layers.append((name, "BN", bottoms, tops, None))
            elif typ == "Scale":
                raise AssertionError("unexpected standalone Scale: " + name)
            elif typ == "ReLU":
                chans[tops[0]] = chans[bottoms[0]]
                self.layers.append((name, "ReLU", bottoms, tops, None))
            elif typ == "Pooling":
                pp = L["pooling_param"][0]
                assert _one(pp, "pool") == "MAX", "only MAX pooling is used"
                (kh, kw), (sh, sw), (ph, pw), _ = geometry(pp)
                chans[tops[0]] = chans[bottoms[0]]
                self.layers.append((name, "Pool", bottoms, tops, ((kh, kw), (sh, sw), (ph, pw))))
            elif typ == "Eltwise":
                ep = L.get("eltwise_param", [{}])[0]
                assert _one(ep, "operation", "SUM") == "SUM"
                coeff = [float(c) for c in ep.get("coeff", [])] or [1.0] * len(bottoms)
                chans[tops[0]] = chans[bottoms[0]]
                self.layers.append((name, "Sum", bottoms, tops, coeff))
            elif typ == "Concat":
                chans[tops[0]] = sum(chans[b] for b in bottoms)
                self.layers.append((name, "Concat", bottoms, tops, None))
            elif typ == "Softmax":
                self.logits_blob, self.prob_blob = bottoms[0], tops[0]
            else:
                raise NotImplementedError("layer type %s (%s)" % (typ, name))

    # Caffe fillers: gaussian(std), xavier (uniform, fan_in), constant(value), bilinear
    def _init(self, m, cp):
        wf = cp.get("weight_filler", [{}])[0]
        bf = cp.get("bias_filler", [{}])[0]
        w = m.weight
        if isinstance(m, MultiFrameConv1):
            fan_in = w.shape[1] * w.shape[2] * w.shape[3] * w.shape[4]
        elif isinstance(m, nn.ConvTranspose2d):
            fan_in = w.shape[1] * w.shape[2] * w.shape[3]
        else:
            fan_in = w.shape[1] * w.shape[2] * w.shape[3]
        t = _one(wf, "type", "constant")
        with torch.no_grad():
            if t == "gaussian":
                w.normal_(_one(wf, "mean", 0.0), float(_one(wf, "std", 1.0)))
            elif t == "xavier":
                s = math.sqrt(3.0 / fan_in)
                w.uniform_(-s, s)
            elif t == "bilinear":
                kh, kw = w.shape[2], w.shape[3]
                f = math.ceil(kw / 2.0)
                c = (2 * f - 1 - f % 2) / (2.0 * f)
                ys, xs = np.meshgrid(np.arange(kh), np.arange(kw), indexing="ij")
                ker = (1 - np.abs(xs / f - c)) * (1 - np.abs(ys / f - c))
                w.copy_(torch.from_numpy(ker.astype(np.float32)).expand_as(w))
            else:
                w.fill_(float(_one(wf, "value", 0.0)))
            if m.bias is not None:
                m.bias.fill_(float(_one(bf, "value", 0.0)))

    def forward(self, x, return_prob=False):
        blobs = {self.input_name: x}
        for name, typ, bottoms, tops, extra in self.layers:
            if typ in ("Convolution", "Deconvolution"):
                y = self.mods[name](blobs[bottoms[0]])
            elif typ == "BN":
                y = self.mods[name](blobs[bottoms[0]])
            elif typ == "ReLU":
                y = F.relu(blobs[bottoms[0]])
            elif typ == "Pool":
                k, s, p = extra
                y = F.max_pool2d(blobs[bottoms[0]], k, s, p, ceil_mode=True)
            elif typ == "Sum":
                y = sum(c * blobs[b] for c, b in zip(extra, bottoms))
            elif typ == "Concat":
                y = torch.cat([blobs[b] for b in bottoms], 1)
            blobs[tops[0]] = y
        logits = blobs[self.logits_blob]
        return F.softmax(logits, 1) if return_prob else logits

    def recalibrate_bn(self, sampler, batches=40):
        """Re-estimate every BatchNorm's running mean/variance for the CURRENT weights.

        The running statistics are a slow moving average (time constant ~1000 forward passes), so right
        after fine-tuning they still describe the old weights and eval-mode predictions can be badly off
        (seen: 64% instead of 5% mitochondria pixels after 30 updates). This replaces them with the
        cumulative average over `batches` forward passes on class-balanced training patches, in train mode
        without gradients (the same kind of batches the statistics are averaged over in training).
        """
        bns = [m for m in self.modules() if isinstance(m, nn.BatchNorm2d)]
        for b in bns:
            b.reset_running_stats()
            b.momentum = None                      # cumulative average
        was_training = self.training
        self.train()
        with torch.no_grad():
            for _ in range(batches):
                self(torch.from_numpy(sampler.batch_arrays()[0]))
        for b in bns:
            b.momentum = 0.001
        self.train(was_training)

    # ---------------------------------------------------------------- weights <-> caffemodel
    def _blobs(self, net, layer):
        for l in net.layer:
            if l.name == layer:
                return l
        raise KeyError(layer)

    @staticmethod
    def _arr(b):
        return np.array(b.data, dtype=np.float32).reshape([int(x) for x in b.shape.dim])

    def load_caffemodel(self, path, pb):
        net = pb.NetParameter()
        net.ParseFromString(open(path, "rb").read())
        L = {l.name: l for l in net.layer}
        with torch.no_grad():
            for name, mod in self.mods.items():
                if isinstance(mod, (nn.BatchNorm2d, TiledBN)):
                    bn = mod if isinstance(mod, nn.BatchNorm2d) else mod.bn
                    b, s = L[name].blobs, L[self.bn_scale[name]].blobs
                    factor = float(self._arr(b[2]).ravel()[0])
                    factor = 1.0 / factor if factor != 0 else 0.0
                    bn.running_mean.copy_(torch.from_numpy(self._arr(b[0]) * factor))
                    bn.running_var.copy_(torch.from_numpy(self._arr(b[1]) * factor))
                    bn.weight.copy_(torch.from_numpy(self._arr(s[0])))
                    bn.bias.copy_(torch.from_numpy(self._arr(s[1])))
                    continue
                W, bias = self._arr(L[name].blobs[0]), self._arr(L[name].blobs[1])
                if isinstance(mod, MultiFrameConv1):
                    mod.weight.copy_(torch.from_numpy(W))
                elif self.frames > 1 and name == "conv1_2":
                    W = np.concatenate([W[..., d] for d in range(W.shape[4])], axis=1)
                    mod.weight.copy_(torch.from_numpy(W))
                else:
                    if W.ndim == 5:
                        assert W.shape[4] == 1
                        W = W[..., 0]
                    mod.weight.copy_(torch.from_numpy(W))
                mod.bias.copy_(torch.from_numpy(bias))

    def export_caffemodel(self, template_path, out_path, pb):
        """Write the current weights into a copy of the original .caffemodel (same layers, same layout)."""
        net = pb.NetParameter()
        net.ParseFromString(open(template_path, "rb").read())
        L = {l.name: l for l in net.layer}

        def put(blob, arr):
            del blob.shape.dim[:]
            blob.shape.dim.extend(arr.shape)
            del blob.data[:]
            blob.data.extend(np.ascontiguousarray(arr, dtype=np.float32).ravel().tolist())

        with torch.no_grad():
            for name, mod in self.mods.items():
                if isinstance(mod, (nn.BatchNorm2d, TiledBN)):
                    bn = mod if isinstance(mod, nn.BatchNorm2d) else mod.bn
                    b, s = L[name].blobs, L[self.bn_scale[name]].blobs
                    put(b[0], bn.running_mean.numpy())
                    put(b[1], bn.running_var.numpy())
                    put(b[2], np.array([1.0], dtype=np.float32))     # scale factor 1: stored stats are the stats
                    put(s[0], bn.weight.numpy())
                    put(s[1], bn.bias.numpy())
                    continue
                blobs = L[name].blobs
                W = mod.weight.numpy()
                if isinstance(mod, MultiFrameConv1):
                    pass                                              # already (out, 1, k, k, kz)
                elif self.frames > 1 and name == "conv1_2":
                    d1 = self.depth1
                    W = np.stack(np.split(W, d1, axis=1), axis=-1)    # (64, 32, k, k, D1)
                elif len(blobs[0].shape.dim) == 5:
                    W = W[..., None]
                put(blobs[0], W)
                put(blobs[1], mod.bias.numpy())
        open(out_path, "wb").write(net.SerializeToString())


def load_caffe_pb2(proto_path):
    """Compile the Caffe fork's caffe.proto (needs protoc) and import it."""
    tmp = tempfile.mkdtemp()
    subprocess.run(["protoc", "-I", os.path.dirname(os.path.abspath(proto_path)), "--python_out=" + tmp,
                    os.path.abspath(proto_path)], check=True)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    sys.path.insert(0, tmp)
    import caffe_pb2
    return caffe_pb2
