import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.quantization as quant

sys.path.insert(0, os.path.dirname(__file__))

from kdpoint_export import (  # noqa: E402
    build_feature_requant_export,
    build_geometry_contract,
    direct_operator_payload,
    flatten_signed_bytes,
    fused_lift_payload,
    interp_qcat_record,
    prepare_calibrated_model,
    q31_multiplier,
    residual_record,
    rtl_requant,
    sha256_file,
    verify_hardware_feature_requants,
    write_quant_export,
)
from ptq_general import get_hardware_qat_qconfig, get_qat_module_mapping  # noqa: E402
from openpoints.models.layers.group import QueryAndGroup  # noqa: E402


class KDPointExportTest(unittest.TestCase):
    def _prepared_conv(self, cin, cout, bias=True):
        module = nn.Sequential(nn.Conv1d(cin, cout, 1, bias=bias))
        module.qconfig = get_hardware_qat_qconfig("fbgemm")
        module.train()
        quant.prepare_qat(
            module, mapping=get_qat_module_mapping(), inplace=True
        )
        conv = module[0]
        with torch.no_grad():
            conv.weight.copy_(torch.linspace(
                -0.7, 0.8, steps=cin * cout
            ).reshape(cout, cin, 1))
            if conv.bias is not None:
                conv.bias.copy_(torch.linspace(-0.2, 0.3, steps=cout))
        conv.weight_fake_quant(conv.weight)
        conv.activation_post_process(torch.tensor([-1.0, 0.0, 3.0]))
        input_fq = get_hardware_qat_qconfig("fbgemm").activation()
        input_fq(torch.tensor([-2.0, 0.0, 4.0]))
        for fake_quant in (conv.weight_fake_quant,
                           conv.activation_post_process, input_fq):
            fake_quant.disable_observer()
            fake_quant.enable_fake_quant()
        return conv, input_fq

    def test_q31_and_rtl_negative_rounding(self):
        for ratio in (1e-5, 0.125, 0.5, 1.0, 3.75):
            multiplier, shift = q31_multiplier(ratio)
            fixed = multiplier * (2.0 ** -shift)
            self.assertLessEqual(abs(fixed - ratio), 0.5 * (2.0 ** -shift))
        self.assertEqual(rtl_requant(-3, 1, 1), -1)
        self.assertEqual(rtl_requant(3, 1, 1), 2)

    def test_direct_operator_has_cin_cout_bytes_and_consistent_col_sums(self):
        conv, input_fq = self._prepared_conv(3, 4)
        payload, weight_bytes = direct_operator_payload(
            "SA1.conv1", conv, input_fq
        )
        self.assertEqual((payload["cin"], payload["cout"]), (3, 4))
        self.assertEqual(len(weight_bytes), 12)
        signed = flatten_signed_bytes(weight_bytes)
        expected_sums = [sum(signed[row * 4 + col] for row in range(3))
                         for col in range(4)]
        self.assertEqual(payload["col_sum_w"], expected_sums)

    def test_fused_lift_inserts_exact_dp_identity_block(self):
        stem, input_fq = self._prepared_conv(4, 5)
        stem.activation_post_process.load_state_dict(
            input_fq.state_dict(), strict=True
        )
        payload, weight_bytes = fused_lift_payload(
            "SA1.fused_lift", stem, input_fq
        )
        self.assertEqual((payload["cin"], payload["cout"]), (7, 8))
        matrix = flatten_signed_bytes(weight_bytes)
        for axis in range(3):
            self.assertEqual(matrix[axis * 8 + axis], 1)
            self.assertEqual(payload["m0"][axis], 1)
            self.assertEqual(payload["shift"][axis], 0)
            self.assertEqual(payload["bias_int32"][axis], 0)
        for row in range(3, 7):
            self.assertEqual(matrix[row * 8 + 0:row * 8 + 3], [0, 0, 0])

    def test_residual_bias_makes_both_zero_points_map_to_output_zero(self):
        record = residual_record(
            "SA2", (0.25, 123), (0.125, 17), (0.5, 91)
        )
        main = record["main"]
        skip = record["skip"]
        output = (rtl_requant(123, main["m0"], main["shift"])
                  + rtl_requant(17, skip["m0"], skip["shift"])
                  + record["bias"])
        self.assertEqual(output, 91)
        self.assertEqual(record["clamp_lo"], 91)

    def test_interp_and_qcat_ratios_match_rtl_accumulator_units(self):
        record = interp_qcat_record(
            "FP4", (0.5, 113), (0.25, 37), (0.125, 19)
        )
        interp = record["interpolation"]
        qcat = record["qcat_skip"]
        self.assertAlmostEqual(
            interp["m0"] * (2.0 ** -interp["shift"]),
            0.5 / (128 * 0.125),
        )
        self.assertAlmostEqual(
            qcat["m0"] * (2.0 ** -qcat["shift"]),
            0.25 / 0.125,
        )
        self.assertEqual(interp["z_out"], qcat["z_out"])

    def _geometry_model(self):
        activation = get_hardware_qat_qconfig("fbgemm").activation()
        activation(torch.tensor([-0.75, 0.0, 1.25]))
        activation.disable_observer()
        activation.enable_fake_quant()

        class DpTarget(nn.Module):
            def __init__(self, fake_quant):
                super().__init__()
                self.activation_post_process = fake_quant

        class EncoderOwner(nn.Module):
            def __init__(self, fake_quant):
                super().__init__()
                self.grouper = QueryAndGroup(
                    0.25, 32, normalize_dp=True,
                    coord_hardware_exact=True,
                )
                self.quant_feat = DpTarget(fake_quant)
                self.grouper.coord_dp_qparams = (
                    float(fake_quant.scale.item()),
                    int(fake_quant.zero_point.item()),
                )

        class FeaturePropogation(nn.Module):
            def __init__(self):
                super().__init__()
                self.coord_hardware_exact = True

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.coord_hardware_exact = True
                self.fp = nn.ModuleList([FeaturePropogation() for _ in range(4)])

        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.stage = EncoderOwner(activation)
        model.decoder = Decoder()
        return model

    def test_geometry_contract_covers_q9_sidecar_dynamic_dp_and_u17(self):
        model = self._geometry_model()
        contract = build_geometry_contract(model)
        self.assertEqual(contract["schema"], "kdpoint.geometry-contract.v1")
        self.assertEqual(contract["status"], "admissible_runtime_scene_config")
        source = contract["source_coordinate_encoding"]
        self.assertEqual(source["q9_code_range"], [0, 511])
        self.assertEqual(source["main_path"]["q8_code"], "q9 >> 1")
        self.assertEqual(source["encoder_sidecar"]["bits_per_point"], 3)
        self.assertEqual(len(contract["encoder_bq_dp"]), 1)
        self.assertEqual(
            contract["encoder_bq_dp"][0]["dp_ratio"],
            "scene_step / (radius * target_scale)",
        )
        decoder = contract["decoder_interpolation"]
        self.assertEqual(decoder["stage_count"], 4)
        self.assertEqual(decoder["weight_format"], "U1.7")
        self.assertTrue(decoder["hardware_exact"])

        model.encoder.stage.grouper.coord_hardware_exact = False
        with self.assertRaisesRegex(ValueError, "hardware geometry is disabled"):
            build_geometry_contract(model)

    def test_pointnext_feature_requant_export_is_explicitly_empty(self):
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.encoder = nn.Sequential()

        payload = build_feature_requant_export(model)

        self.assertEqual(payload["schema"], "kdpoint.feature-requant-config.v1")
        self.assertEqual(payload["stages"], [])

    def test_feature_requant_legacy_proof_is_rederived_strictly(self):
        derived = [{
            "schema": "kdpoint.feature-requant-stage.v1",
            "stage": "SA2",
            "m0": 2035247015,
            "shift": 31,
            "uint8_codebook_bit_exact": True,
        }]
        legacy = [dict(derived[0])]
        legacy[0].pop("uint8_codebook_bit_exact")
        module = SimpleNamespace(
            collect_hardware_feature_requants=lambda model, scope: derived
        )
        with patch("kdpoint_export._ptq_general_module", return_value=module):
            verify_hardware_feature_requants(nn.Module(), legacy)
            invalid = [dict(derived[0])]
            invalid[0]["uint8_codebook_bit_exact"] = False
            with self.assertRaisesRegex(ValueError, "proof is not true"):
                verify_hardware_feature_requants(nn.Module(), invalid)

    def test_missing_feature_requant_table_requires_empty_derived_contract(self):
        empty_module = SimpleNamespace(
            collect_hardware_feature_requants=lambda model, scope: []
        )
        with patch(
                "kdpoint_export._ptq_general_module", return_value=empty_module):
            self.assertEqual(
                verify_hardware_feature_requants(nn.Module(), None), []
            )

        required_module = SimpleNamespace(
            collect_hardware_feature_requants=lambda model, scope: [{
                "schema": "kdpoint.feature-requant-stage.v1",
                "stage": "SA2",
                "uint8_codebook_bit_exact": True,
            }]
        )
        with patch(
                "kdpoint_export._ptq_general_module", return_value=required_module):
            with self.assertRaisesRegex(ValueError, "lacks hardware feature"):
                verify_hardware_feature_requants(nn.Module(), None)

    def test_old_calibrated_state_without_hardware_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "old_state.pth"
            torch.save({
                "metadata": {
                    "hardware_domain_lock": True,
                    "hardware_geometry": False,
                },
                "model_state_dict": {},
            }, state_path)
            with self.assertRaisesRegex(ValueError, "hardware-geometry"):
                prepare_calibrated_model(Path(directory), state_path)

    def test_materialized_export_is_deterministic_hashed_and_no_overwrite(self):
        conv, input_fq = self._prepared_conv(3, 4)
        payload, weight_bytes = direct_operator_payload(
            "SA1.conv1", conv, input_fq
        )
        signature = "1" * 64
        expected = {
            "backend": "fbgemm",
            "calib_batches": 3,
            "opts": ["model.test=true"],
            "cfg_sha256": "2" * 64,
            "checkpoint_sha256": "3" * 64,
        }
        result = {
            "manifest_id": "S1",
            "operators": {"SA1.conv1": (payload, weight_bytes)},
            "residual": {
                "schema": "kdpoint.residual-config.v1", "stages": []},
            "interp_qcat": {
                "schema": "kdpoint.interp-qcat-config.v1", "stages": []},
            "feature_requant": {
                "schema": "kdpoint.feature-requant-config.v1", "stages": []},
            "geometry_contract": {
                "schema": "kdpoint.geometry-contract.v1",
                "status": "admissible_runtime_scene_config",
                "source_coordinate_encoding": {"q9_code_range": [0, 511]},
                "encoder_bq_dp": [{"stage": "SA1"}],
                "decoder_interpolation": None,
            },
            "metadata": {
                "backend": expected["backend"],
                "calib_batches": expected["calib_batches"],
                "opts": expected["opts"],
            },
        }
        source_contract = {
            "schema": "kdpoint.quant-source.v1",
            "hardware_quant_abi": {"activation": {"dtype": "uint8"}},
            "networks": {"S1": {
                "manifest_id": "S1",
                "logical_operator_count": 1,
                "operator_signature_sha256": signature,
                "provenance": expected,
            }},
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "source_state.pth"
            metrics_path = root / "source_metrics.json"
            contract_path = root / "source_contract.json"
            state_path.write_bytes(b"calibrated-state")
            metrics_path.write_text(
                json.dumps({"metric": 91.5}), encoding="utf-8")
            contract_path.write_text(
                json.dumps(source_contract, sort_keys=True), encoding="utf-8")

            first = root / "export_a"
            second = root / "export_b"
            manifest_path = write_quant_export(
                result, first, state_path, metrics_path, contract_path
            )
            write_quant_export(
                result, second, state_path, metrics_path, contract_path
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "kdpoint.quant-export.v1")
            self.assertEqual(manifest["manifest_id"], "S1")
            self.assertEqual(manifest["operator_signature_sha256"], signature)
            self.assertEqual(
                manifest["source_contract_sha256"], sha256_file(contract_path)
            )
            self.assertEqual(
                manifest["calibrated_state"]["sha256"], sha256_file(state_path)
            )
            self.assertEqual(
                manifest["metrics"]["sha256"], sha256_file(metrics_path)
            )
            self.assertEqual(
                json.loads((first / "special" / "dp.json").read_text(
                    encoding="utf-8"))["stages"], [{"stage": "SA1"}]
            )

            first_files = sorted(
                path.relative_to(first) for path in first.rglob("*")
                if path.is_file()
            )
            second_files = sorted(
                path.relative_to(second) for path in second.rglob("*")
                if path.is_file()
            )
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual(
                    (first / relative).read_bytes(),
                    (second / relative).read_bytes(),
                )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_quant_export(
                    result, first, state_path, metrics_path, contract_path
                )


if __name__ == "__main__":
    unittest.main()
