#!/usr/bin/env python3
"""Train (or fine-tune) a CDeep3M 1fm/3fm/5fm model in PyTorch, with the Caffe recipe, resumable.

  python -m trainer.train --model 1fm --deploy <1fm/deploy.prototxt> --data <train_data dir> \
      --out <run dir> [--weights <pretrained .caffemodel> --caffe-proto <caffe.proto>] \
      [--max-iter 30000] [--base-lr 0.01] [--iter-size 8]

Defaults are those of CDeep3M2's caffetrain.sh (30000 updates, batch 5 x iter_size 8, 320x320 patches,
SGD momentum 0.9, weight decay 0.0005, poly lr power 0.8).

Progress is saved atomically to <out>/latest.pt every --checkpoint-minutes (default 10) and every
--checkpoint-every updates, on SIGTERM/SIGINT/SIGHUP (e.g. when the PC shuts down; the unfinished update is
dropped) and at the end. Running the same command again resumes from <out>/latest.pt. Small progress files
(progress/<run>.csv/.json inside the repo, if --progress-dir is given) are meant to be pushed by tools/autopush.sh.
"""
import argparse
import csv
import json
import os
import signal
import time

import numpy as np
import torch

from .caffe_net import CaffeNet, load_caffe_pb2
from .data import PatchSampler
from .solver import CaffeSGD, balanced_softmax_loss, poly_lr


class Stop(Exception):
    pass


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["1fm", "3fm", "5fm"])
    ap.add_argument("--deploy", required=True, help="the model's deploy.prototxt (defines the network)")
    ap.add_argument("--data", required=True, help="folder with training_full_stacks_*.h5")
    ap.add_argument("--out", required=True, help="run folder for checkpoints")
    ap.add_argument("--weights", help="pretrained .caffemodel to start from (needs --caffe-proto)")
    ap.add_argument("--caffe-proto", default=os.environ.get("CAFFE_PROTO"))
    ap.add_argument("--max-iter", type=int, default=30000)
    ap.add_argument("--base-lr", type=float, default=1e-2)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--iter-size", type=int, default=8)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--patch", type=int, default=320)
    ap.add_argument("--average-loss", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-minutes", type=float, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=100, help="also every N updates")
    ap.add_argument("--keep", type=int, default=3, help="numbered checkpoints to keep (plus latest.pt)")
    ap.add_argument("--progress-dir", help="write small progress .csv/.json here (tracked by git)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto = the GPU if this PyTorch build sees one (CUDA or ROCm build), otherwise the CPU")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index for --device cuda/auto")
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    if a.device != "cpu" and torch.cuda.is_available():
        dev = torch.device("cuda:%d" % a.gpu)
        dev_name = "%s (%s)" % (dev, torch.cuda.get_device_name(a.gpu))
    elif a.device == "cuda":
        raise SystemExit("--device cuda: this PyTorch build (%s) sees no GPU; use --device cpu or another PyTorch build" % torch.__version__)
    else:
        dev, dev_name = torch.device("cpu"), "cpu"
    os.makedirs(a.out, exist_ok=True)
    run = "%s" % a.model
    net = CaffeNet(a.deploy).to(dev)
    expect = {"1fm": 1, "3fm": 3, "5fm": 5}[a.model]
    assert net.frames == expect, "deploy.prototxt has %d frames, model %s needs %d" % (net.frames, a.model, expect)
    torch.manual_seed(a.seed)
    sampler = PatchSampler.from_dir(a.data, frames=net.frames, patch=a.patch, batch=a.batch, seed=a.seed)
    opt = CaffeSGD(net, a.momentum, a.weight_decay)
    loss_rng = np.random.default_rng(a.seed + 1)

    it, elapsed_before, history = 0, 0.0, []
    latest = os.path.join(a.out, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sampler.set_state(ck["sampler"])
        loss_rng.bit_generator.state = ck["loss_rng"]
        it, elapsed_before, history = ck["iter"], ck["elapsed"], ck["history"]
        print("resumed from %s at update %d" % (latest, it), flush=True)
    elif a.weights:
        if not a.caffe_proto:
            raise SystemExit("--weights needs --caffe-proto (caffe.proto of the CDeep3M Caffe fork)")
        net.load_caffemodel(a.weights, load_caffe_pb2(a.caffe_proto))
        print("initialised from %s" % a.weights, flush=True)
    else:
        print("training from scratch (Caffe fillers)", flush=True)

    stop = {"flag": False, "sig": None}

    def on_signal(signum, frame):
        stop["flag"], stop["sig"] = True, signum

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)

    t_start = time.time()
    last_ck = time.time()
    csv_path = None
    if a.progress_dir:
        os.makedirs(a.progress_dir, exist_ok=True)
        csv_path = os.path.join(a.progress_dir, "%s.csv" % run)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["update", "lr", "loss_reported", "loss_backward", "seconds_per_update", "elapsed_hours"])

    def save(final=False):
        nonlocal last_ck
        ck = {"model": net.state_dict(), "opt": opt.state_dict(), "sampler": sampler.state(),
              "loss_rng": loss_rng.bit_generator.state, "iter": it,
              "elapsed": elapsed_before + time.time() - t_start, "history": history, "args": vars(a)}
        atomic_save(ck, latest)
        numbered = os.path.join(a.out, "ckpt_%07d.pt" % it)
        if it % a.checkpoint_every == 0 or final:
            atomic_save(ck, numbered)
            olds = sorted(f for f in os.listdir(a.out) if f.startswith("ckpt_") and f.endswith(".pt"))
            for f in olds[:-a.keep]:
                os.remove(os.path.join(a.out, f))
        last_ck = time.time()
        if a.progress_dir:
            st = {"model": a.model, "update": it, "max_iter": a.max_iter,
                  "last_loss": history[-1][1] if history else None,
                  "avg_loss_last_%d" % a.average_loss: float(np.mean([h[1] for h in history[-a.average_loss:]])) if history else None,
                  "elapsed_hours": round((elapsed_before + time.time() - t_start) / 3600, 3),
                  "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": bool(final and it >= a.max_iter)}
            json.dump(st, open(os.path.join(a.progress_dir, "%s.json" % run), "w"), indent=1)

    net.train()
    print("starting at update %d of %d (iter_size %d, batch %d, %d threads, device %s, torch %s)" % (
        it, a.max_iter, a.iter_size, a.batch, a.threads, dev_name, torch.__version__), flush=True)
    try:
        while it < a.max_iter:
            lr = poly_lr(a.base_lr, it, a.max_iter, a.power)
            t0 = time.time()
            opt.zero_grad()
            back_sum = rep_sum = 0.0
            for _ in range(a.iter_size):
                if stop["flag"]:
                    raise Stop()
                x, y = sampler.batch_arrays()
                logits = net(torch.from_numpy(x).to(dev))
                back, rep = balanced_softmax_loss(logits, torch.from_numpy(y).to(dev), sampler.rates, loss_rng)
                (back / a.iter_size).backward()                     # Caffe normalises accumulated grads by iter_size
                back_sum += float(back) / a.iter_size
                rep_sum += float(rep) / a.iter_size
            opt.step(lr)
            it += 1
            history.append((it, rep_sum))
            dt = time.time() - t0
            print("update %d/%d  lr %.5f  loss %.5f (avg%d %.5f)  %.1fs" % (
                it, a.max_iter, lr, rep_sum, a.average_loss, float(np.mean([h[1] for h in history[-a.average_loss:]])), dt), flush=True)
            if csv_path:
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([it, "%.6f" % lr, "%.6f" % rep_sum, "%.6f" % back_sum, "%.1f" % dt,
                                            "%.3f" % ((elapsed_before + time.time() - t_start) / 3600)])
            if it % a.checkpoint_every == 0 or (time.time() - last_ck) > a.checkpoint_minutes * 60:
                save()
        save(final=True)
        print("finished %d updates" % it, flush=True)
    except Stop:
        save()
        print("stopped by signal %s at update %d; checkpoint saved, run the same command to resume" % (stop["sig"], it), flush=True)
        raise SystemExit(0)


if __name__ == "__main__":
    main()
