"""
Smoke test: verify PointNet++ now survives the PTQ prepare flow.
Builds the model, runs swap/fuse/disable-geometry/prepare_qat, then a forward
in calibration mode + a forward in INT8-sim mode. No dataloader, no training.

Run on a GPU machine:
  python quant/smoke_test_pn2_quant.py --task seg --cfg cfgs/s3dis/pointnet++.yaml
  python quant/smoke_test_pn2_quant.py --task cls --cfg cfgs/modelnet40ply2048/pointnet++.yaml
  python quant/smoke_test_pn2_quant.py --task partseg --cfg cfgs/shapenetpart/pointnet++.yaml
"""
import os, sys, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
import torch.quantization as quant
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig
from openpoints.models.layers.quant_utils import (
    swap_custom_convs_to_standard, fuse_convbn_modules,
    disable_quantization_for_geometry,
)

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["seg", "cls", "partseg"])
p.add_argument("--cfg", required=True)
p.add_argument("--backend", default="fbgemm")
args = p.parse_args()

cfg = EasyConfig(); cfg.load(args.cfg, recursive=True)
model = build_model_from_cfg(cfg.model).cuda()
print("=== MODEL BUILT OK ===  params %.4f M" % (sum(x.numel() for x in model.parameters())/1e6))

B, N = 2, cfg.get("num_points", 2048) if args.task != "seg" else 24000
in_ch = cfg.model.encoder_args.in_channels
data = {
    "pos": torch.randn(B, N, 3).cuda(),
    "x":   torch.randn(B, in_ch, N).cuda(),
    "y":   torch.zeros(B, N, dtype=torch.long).cuda() if args.task != "cls"
           else torch.zeros(B, dtype=torch.long).cuda(),
}
if args.task == "partseg":
    data["cls"] = torch.zeros(B, 1, dtype=torch.long).cuda()

# fp32 forward
with torch.no_grad():
    out = model(data)
print("=== FP32 FORWARD OK ===  out", tuple(out.shape))

# quant prep
model.cpu().eval()
swap_custom_convs_to_standard(model)
fuse_convbn_modules(model)
disable_quantization_for_geometry(model)
torch.backends.quantized.engine = args.backend
model.qconfig = quant.get_default_qat_qconfig(args.backend)
model.train()
quant.prepare_qat(model, inplace=True)
print("=== prepare_qat OK ===")

# calibration-mode forward
model.apply(quant.disable_fake_quant); model.apply(quant.enable_observer)
model.eval().cuda()
data = {k: v.cuda() for k, v in data.items()}
with torch.no_grad():
    out = model(data)
print("=== CALIBRATION FORWARD OK ===  out", tuple(out.shape))

# INT8-sim forward
model.apply(quant.enable_fake_quant); model.apply(quant.disable_observer)
with torch.no_grad():
    out = model(data)
print("=== INT8-SIM FORWARD OK ===  out", tuple(out.shape))
print("\nALL PASSED — PointNet++ PTQ infra works for task=%s" % args.task)
