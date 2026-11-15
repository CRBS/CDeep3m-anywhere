"""Patch sampler that reproduces the CDeep3M Caffe fork's PatchData layer.

* Volumes: HDF5 files with datasets 'data' and 'label' of shape (H, W, Z) written by PreprocessTrainingData.py
  (16 geometric/z-order variants of the training stack).
* A patch is drawn as in patch_sampler.cpp: pick a volume uniformly, draw a uniformly random centre
  (in-plane so that the patch fits, z anywhere in the stack), read the label at the centre and accept the
  centre with probability 1/skip_rate[label] (rejection sampling, class balancing), repeat until accepted.
* The data patch has `frames` slices around the centre slice, zero filled outside the stack (pad_method
  ZERO); the label patch is the centre slice. The fork's on-the-fly mirror option is commented out in its
  transformer, so there is no flipping here; augmentation lives in the preprocessed volumes.
"""
import glob
import math
import os

import h5py
import numpy as np


def skip_rates(probs, num_top=2):
    """Replicates SampleSelector::ComputeLabelSkipRate for label_class_selection.prototxt
    (ignore_rest_of_label true). probs: {label: prob}."""
    ordered = sorted(probs.items(), key=lambda kv: -kv[1])[:num_top]
    bottom = np.float32(ordered[-1][1])
    scale = np.float32(1.0 / (float(bottom) if bottom != 0 else 0.00001))
    return {label: int(math.ceil(float(np.float32(prob) * scale))) for label, prob in ordered}


class PatchSampler:
    def __init__(self, volume_files, frames=1, patch=320, batch=5, probs=None, seed=0):
        self.frames, self.patch, self.batch = frames, patch, batch
        self.rates = skip_rates(probs or {0: 1.0, 1: 0.3})
        self.rng = np.random.default_rng(seed)
        self.data, self.label = [], []
        for f in volume_files:
            with h5py.File(f, "r") as h:
                d, l = h["data"][()], h["label"][()]
            if d.dtype != np.uint8:
                di = d.astype(np.uint8)
                d = di if np.array_equal(di, d) else d.astype(np.float32)   # keep exact values
            self.data.append(d)
            self.label.append(l.astype(np.uint8))
        if not self.data:
            raise FileNotFoundError("no training volumes given")
        self.shape = self.data[0].shape

    @classmethod
    def from_dir(cls, directory, **kw):
        files = sorted(glob.glob(os.path.join(directory, "training_full_stacks_*.h5")))
        return cls(files, **kw)

    # ---- state for resuming
    def state(self):
        return self.rng.bit_generator.state

    def set_state(self, st):
        self.rng.bit_generator.state = st

    def _accept(self, label):
        r = self.rates.get(int(label), 0)
        return r > 0 and int(self.rng.integers(0, 2 ** 31)) % r == 0

    def one_patch(self):
        h = self.patch
        half = h // 2
        v = int(self.rng.integers(len(self.data)))
        d, l = self.data[v], self.label[v]
        H, W, Z = d.shape
        while True:
            cy = int(self.rng.integers(half, H - half))          # [half, H-half-1]
            cx = int(self.rng.integers(half, W - half))
            cz = int(self.rng.integers(0, Z))
            if self._accept(l[cy, cx, cz]):
                break
        ys, xs = slice(cy - half, cy + half), slice(cx - half, cx + half)
        k = self.frames // 2
        patch = np.zeros((self.frames, h, h), dtype=np.float32)
        for j in range(self.frames):
            z = cz - k + j
            if 0 <= z < Z:
                patch[j] = d[ys, xs, z]
        return patch, l[ys, xs, cz].astype(np.int64)

    def batch_arrays(self):
        ps = [self.one_patch() for _ in range(self.batch)]
        return np.stack([p[0] for p in ps]), np.stack([p[1] for p in ps])
