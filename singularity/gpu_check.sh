#!/bin/bash
# Run inside the container (singularity exec --nv ... gpu_check.sh): shows whether the GPU is usable for
# training (PyTorch CUDA) and for prediction (ncnn Vulkan). Exit code 0 only when both work.
ok=0
echo "== nvidia-smi"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>&1 || { echo "no NVIDIA GPU visible (forgot --nv?)"; ok=1; }
echo "== PyTorch CUDA (training)"
python3 - <<'PY' || ok=1
import sys, torch
if not torch.cuda.is_available():
    print("CUDA not available"); sys.exit(1)
cap = torch.cuda.get_device_capability(0)
print("torch %s, CUDA %s, %s, compute %d.%d, supported %s"
      % (torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), cap[0], cap[1],
         "sm_%d%d" % cap in torch.cuda.get_arch_list()))
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
PY
echo "== Vulkan (ncnn prediction)"
vulkaninfo --summary 2>&1 | grep -E "deviceName|driverInfo|ERROR" | sed 's/^ *//'
dev=$(python3 /opt/cdeep3m2/ncnn_predict.py --pick-device 2>/dev/null | tail -1)
echo "ncnn --device auto picks: ${dev:-nothing}"
[[ "$dev" == vulkan* ]] || { echo "Vulkan GPU not usable: prediction would run on the CPU"; ok=1; }
[ $ok = 0 ] && echo "== ALL OK" || echo "== PROBLEMS (see above)"
exit $ok
