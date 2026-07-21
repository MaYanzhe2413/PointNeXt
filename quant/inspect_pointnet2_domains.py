"""Inspect PointNet++ incoming-feature and grouped-input quantization domains."""

import argparse
import json
import re

import torch


def scalar(state, prefix):
    scale = state[prefix + ".scale"]
    zero_point = state[prefix + ".zero_point"]
    if scale.numel() != 1 or zero_point.numel() != 1:
        raise RuntimeError(f"{prefix}: expected scalar qparams")
    return {
        "name": prefix,
        "scale": float(scale.item()),
        "zero_point": int(zero_point.item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state")
    args = parser.parse_args()
    payload = torch.load(args.state, map_location="cpu")
    state = payload["model_state_dict"]

    target_pattern = re.compile(
        r"encoder\.SA_modules\.(\d+)\.local_aggregations\.0\."
        r"SA_CONFIG_operator\.quant_feat\.activation_post_process\.scale$"
    )
    stages = sorted(
        int(match.group(1))
        for key in state
        for match in [target_pattern.match(key)]
        if match
    )
    records = []
    for stage in stages:
        target_prefix = (
            f"encoder.SA_modules.{stage}.local_aggregations.0."
            "SA_CONFIG_operator.quant_feat.activation_post_process"
        )
        if stage == 0:
            incoming_prefix = "encoder.quant_input.activation_post_process"
        else:
            conv_pattern = re.compile(
                rf"encoder\.SA_modules\.{stage - 1}\.local_aggregations\.0\."
                r"SA_CONFIG_operator\.convs\.(\d+)\.0\."
                r"activation_post_process\.scale$"
            )
            indices = [
                int(match.group(1))
                for key in state
                for match in [conv_pattern.match(key)]
                if match
            ]
            if not indices:
                raise RuntimeError(f"SA{stage}: previous output domain not found")
            incoming_prefix = (
                f"encoder.SA_modules.{stage - 1}.local_aggregations.0."
                f"SA_CONFIG_operator.convs.{max(indices)}.0."
                "activation_post_process"
            )
        incoming = scalar(state, incoming_prefix)
        target = scalar(state, target_prefix)
        records.append({
            "stage": f"SA{stage + 1}",
            "incoming_feature": incoming,
            "grouped_dp_feature": target,
            "scale_ratio_incoming_to_grouped": (
                incoming["scale"] / target["scale"]
            ),
        })
    print(json.dumps({"stages": records}, indent=2))


if __name__ == "__main__":
    main()
