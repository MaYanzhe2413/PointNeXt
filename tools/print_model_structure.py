import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from textwrap import shorten
from typing import Any, Iterable, List

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpoints.utils.config import EasyConfig
from openpoints.models import build_model_from_cfg


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_node(name: str, node: Any, indent: int) -> Iterable[str]:
    prefix = " " * indent
    if isinstance(node, dict):
        yield f"{prefix}{name}:"
        for key in sorted(node.keys()):
            yield from _print_node(key, node[key], indent + 2)
    elif isinstance(node, (list, tuple)):
        if all(not isinstance(item, (dict, list, tuple)) for item in node):
            formatted = ", ".join(_format_value(item) for item in node)
            yield f"{prefix}{name}: [{formatted}]"
        else:
            yield f"{prefix}{name}:"
            for idx, item in enumerate(node):
                yield from _print_node(f"[{idx}]", item, indent + 2)
    else:
        yield f"{prefix}{name}: {_format_value(node)}"


def _extract_shape(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [_extract_shape(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _extract_shape(val) for key, val in obj.items()}
    if obj is None:
        return None
    return str(type(obj).__name__)


def _kernel_size_str(module: nn.Module) -> str:
    if hasattr(module, "kernel_size"):
        kernel = module.kernel_size
        if isinstance(kernel, (list, tuple)):
            return "x".join(str(k) for k in kernel)
        return str(kernel)
    if isinstance(module, nn.Linear):
        return f"{module.out_features}x{module.in_features}"
    if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
        return "x".join(str(dim) for dim in module.weight.shape)
    return "-"


def _yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


def _module_requires_report(name: str, module: nn.Module) -> bool:
    has_children = any(True for _ in module.children())
    if not has_children:
        return True
    lowered = module.__class__.__name__.lower()
    if "res" in lowered or "incep" in lowered:
        return True
    return False


def _collect_model_details(model: nn.Module, sample_input: Any, max_width: int = 60) -> List[dict]:
    records: List[dict] = []
    hooks = []
    call_counter = defaultdict(int)

    def register_hook(name: str, module: nn.Module) -> None:
        def hook(_module: nn.Module, inputs: tuple, output: Any) -> None:
            call_counter[name] += 1
            suffix = "" if call_counter[name] == 1 else f"#{call_counter[name]}"
            record_name = f"{name}{suffix}" if name else _module.__class__.__name__
            input_shape = _extract_shape(inputs if len(inputs) != 1 else inputs[0])
            output_shape = _extract_shape(output)
            params = sum(p.numel() for p in _module.parameters(recurse=False))
            lowered = _module.__class__.__name__.lower()
            records.append({
                "name": record_name,
                "type": _module.__class__.__name__,
                "input": shorten(pformat(input_shape, compact=True), width=max_width, placeholder="…"),
                "output": shorten(pformat(output_shape, compact=True), width=max_width, placeholder="…"),
                "kernel": _kernel_size_str(_module),
                "params": params,
                "residual": _yes_no("res" in lowered),
                "inception": _yes_no("incep" in lowered)
            })

        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if name == "":
            continue
        if _module_requires_report(name, module):
            register_hook(name, module)

    with torch.no_grad():
        model(sample_input)

    for hook in hooks:
        hook.remove()

    return records


def _build_sample_inputs(cfg: EasyConfig, batch_size: int, num_points: int, feature_dim: int, device: torch.device):
    feature_last = cfg.model.get("feature_last_dim", False)
    encoder_args = cfg.model.get("encoder_args", {})
    inferred_in_channels = encoder_args.get("in_channels", cfg.model.get("in_channels", feature_dim))
    in_channels = inferred_in_channels if inferred_in_channels is not None else feature_dim

    points = torch.randn(batch_size, num_points, 3, device=device)
    if feature_last:
        features = torch.randn(batch_size, num_points, in_channels, device=device)
    else:
        features = torch.randn(batch_size, in_channels, num_points, device=device)

    return {"pos": points, "x": features}


def main() -> None:
    parser = argparse.ArgumentParser(description="Print detailed model structure from a YAML cfg file.")
    parser.add_argument("--cfg", required=True, help="Path to the cfg YAML file.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for shape inference.")
    parser.add_argument("--num-points", type=int, default=1024, help="Dummy number of points per sample.")
    parser.add_argument("--feature-dim", type=int, default=3, help="Dummy feature dimension if not specified in cfg.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for the dummy forward pass.")
    parser.add_argument("--max-width", type=int, default=60, help="Max character width for shape fields.")
    args = parser.parse_args()

    cfg_path = os.path.abspath(args.cfg)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)

    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)

    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise KeyError("'model' section not found in the provided cfg.")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    print("Model Configuration Snippet")
    print("============================")
    for line in _print_node("model", model_cfg, indent=0):
        print(line)

    cls_args = cfg.get("cls_args")
    if cls_args is not None:
        print()
        print("Classification Head")
        print("===================")
        for line in _print_node("cls_args", cls_args, indent=0):
            print(line)

    model = build_model_from_cfg(model_cfg).to(device)
    model.eval()

    sample_input = _build_sample_inputs(cfg, args.batch_size, args.num_points, args.feature_dim, device)

    print()
    print("Layer-wise Summary")
    print("===================")
    records = _collect_model_details(model, sample_input, max_width=args.max_width)

    header = f"{'Layer':<45} {'Type':<25} {'Input Shape':<{args.max_width}} {'Output Shape':<{args.max_width}} {'Kernel':<15} {'Params':<10} {'Residual':<8} {'Inception':<9}"
    print(header)
    print("-" * len(header))
    for record in records:
        print(
            f"{record['name']:<45} {record['type']:<25} "
            f"{record['input']:<{args.max_width}} {record['output']:<{args.max_width}} "
            f"{record['kernel']:<15} {record['params']:<10d} {record['residual']:<8} {record['inception']:<9}"
        )


if __name__ == "__main__":
    main()
