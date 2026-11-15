#!/usr/bin/env python3
"""Write a trained checkpoint back into a CDeep3M .caffemodel (same layers and layout as the original).

  python -m trainer.export_caffemodel --model 1fm --deploy deploy.prototxt --checkpoint run/latest.pt \
      --template original.caffemodel --out trained.caffemodel --caffe-proto caffe.proto

The result can be used with the original CDeep3M tools or converted with ncnn/convert_model.sh.
"""
import argparse
import os

import torch

from .caffe_net import CaffeNet, load_caffe_pb2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--template", required=True, help="an original .caffemodel of the same model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--caffe-proto", default=os.environ.get("CAFFE_PROTO"))
    ap.add_argument("--data", help="training data folder, used to re-estimate the BatchNorm statistics")
    ap.add_argument("--recalibrate", type=int, default=40,
                    help="forward passes used to re-estimate BatchNorm statistics before exporting (0 = keep "
                         "the running averages; needed after short fine-tuning runs, see CaffeNet.recalibrate_bn)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    pb = load_caffe_pb2(a.caffe_proto)
    net = CaffeNet(a.deploy)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    if a.recalibrate > 0:
        if not a.data:
            raise SystemExit("--recalibrate needs --data (or pass --recalibrate 0)")
        from .data import PatchSampler
        sampler = PatchSampler.from_dir(a.data, frames=net.frames, batch=5, seed=12345)
        net.recalibrate_bn(sampler, a.recalibrate)
        print("re-estimated BatchNorm statistics over %d batches" % a.recalibrate)
    net.export_caffemodel(a.template, a.out, pb)
    print("wrote %s (update %d)" % (a.out, ck["iter"]))


if __name__ == "__main__":
    main()
