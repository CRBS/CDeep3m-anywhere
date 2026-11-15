"""Caffe-style loss and SGD solver for the CDeep3M training recipe."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .caffe_net import MultiFrameConv1, TiledBN


def balanced_softmax_loss(logits, labels, rates, rng, ignore_label=2):
    """SoftmaxWithLoss as in the fork: the reported loss is over all valid pixels; the gradient only flows
    through pixels accepted by the sample selector (label l accepted with probability 1/rates[l]) and is
    normalised by the number of ALL valid pixels. Returns (loss_for_backward, loss_to_report)."""
    valid = labels != ignore_label
    ce = F.cross_entropy(logits, labels.clamp(max=logits.shape[1] - 1), reduction="none")
    n_valid = valid.sum().clamp_min(1).to(ce.dtype)
    accept = torch.zeros_like(valid)
    for label, rate in rates.items():
        m = labels == label
        if rate <= 1:
            accept |= m
        else:
            draws = torch.from_numpy(rng.integers(0, rate, size=m.shape)).to(m.device)
            accept |= m & (draws == 0)
    back = (ce * (accept & valid)).sum() / n_valid
    report = (ce.detach() * valid).sum() / n_valid
    return back, report


def poly_lr(base_lr, it, max_iter, power):
    return base_lr * (1.0 - float(it) / max_iter) ** power


class CaffeSGD:
    """v = momentum*v + lr*lr_mult*(grad + wd*decay_mult*w);  w -= v   (Caffe's SGDSolver)."""

    def __init__(self, model, momentum=0.9, weight_decay=0.0005):
        self.momentum, self.wd = momentum, weight_decay
        self.items = []                                  # (param, lr_mult, decay_mult)
        for name, mod in model.mods.items():
            mults = model.lr_mult[name]
            if isinstance(mod, (nn.BatchNorm2d, TiledBN)):
                bn = mod if isinstance(mod, nn.BatchNorm2d) else mod.bn
                params = [bn.weight, bn.bias]
            else:
                params = [mod.weight, mod.bias]
            for i, p in enumerate(params):
                lr_m, dec_m = mults[i] if i < len(mults) else (1.0, 1.0)
                self.items.append((p, lr_m, dec_m))
        self.history = [torch.zeros_like(p) for p, _, _ in self.items]

    def zero_grad(self):
        for p, _, _ in self.items:
            p.grad = None

    @torch.no_grad()
    def step(self, lr):
        for (p, lr_m, dec_m), h in zip(self.items, self.history):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd and dec_m:
                g = g + (self.wd * dec_m) * p
            h.mul_(self.momentum).add_(g, alpha=lr * lr_m)
            p.sub_(h)

    def state_dict(self):
        return {"history": [h.clone() for h in self.history]}

    def load_state_dict(self, sd):
        for h, s in zip(self.history, sd["history"]):
            h.copy_(s)
