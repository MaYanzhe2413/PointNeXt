import io
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.quantization as quant

sys.path.insert(0, os.path.dirname(__file__))

from ptq_general import (  # noqa: E402
    QATConv1d,
    QATConvReLU1d,
    audit_quantized_operators,
    cpu_state_dict,
    get_hardware_qat_qconfig,
    get_qat_module_mapping,
)


class QuantizedOperatorAuditTest(unittest.TestCase):
    def _prepare(self, model):
        model.qconfig = get_hardware_qat_qconfig("fbgemm")
        model.train()
        quant.prepare_qat(
            model, mapping=get_qat_module_mapping(), inplace=True
        )
        return model

    def test_conv1d_conv2d_and_linear_have_valid_weight_fake_quant(self):
        model = nn.ModuleDict({
            "conv1d": nn.Conv1d(3, 5, 1),
            "conv_relu1d": nni.ConvReLU1d(
                nn.Conv1d(3, 5, 1), nn.ReLU()
            ),
            "conv2d": nn.Conv2d(3, 5, 1),
            "linear": nn.Linear(5, 2),
        })
        self._prepare(model)

        operators = audit_quantized_operators(model, stage="unit test")

        self.assertEqual(len(operators), 4)
        self.assertIsInstance(model["conv1d"], QATConv1d)
        self.assertIsInstance(model["conv_relu1d"], QATConvReLU1d)
        for _, module in operators:
            self.assertEqual(module.weight_fake_quant.dtype, torch.qint8)
            self.assertIn(module.weight_fake_quant.qscheme, (
                torch.per_channel_affine,
                torch.per_channel_symmetric,
            ))
            self.assertEqual(
                module.activation_post_process.dtype, torch.quint8
            )
            self.assertIn(module.activation_post_process.qscheme, (
                torch.per_tensor_affine,
                torch.per_tensor_symmetric,
            ))
            self.assertEqual(module.activation_post_process.quant_min, 0)
            self.assertEqual(module.activation_post_process.quant_max, 255)

    def test_reduced_range_activation_is_rejected(self):
        model = nn.Sequential(nn.Conv1d(3, 5, 1))
        model.qconfig = quant.get_default_qat_qconfig("fbgemm")
        model.train()
        quant.prepare_qat(
            model, mapping=get_qat_module_mapping(), inplace=True
        )

        with self.assertRaisesRegex(RuntimeError, r"expected \[0, 255\]"):
            audit_quantized_operators(model, stage="reduced-range unit test")

    def test_activation_observer_alone_is_rejected(self):
        model = nn.Sequential(nn.Conv1d(3, 5, 1))
        qconfig = quant.get_default_qat_qconfig("fbgemm")
        model[0].activation_post_process = qconfig.activation()

        with self.assertRaisesRegex(RuntimeError, "weight.*FakeQuantize"):
            audit_quantized_operators(model, stage="negative unit test")

    def test_calibrated_state_round_trips_strictly(self):
        model = self._prepare(nn.Sequential(nn.Conv1d(3, 5, 1)))
        model(torch.randn(2, 3, 8))

        serialized = io.BytesIO()
        torch.save({"model_state_dict": cpu_state_dict(model)}, serialized)
        serialized.seek(0)
        saved_state = torch.load(serialized)["model_state_dict"]

        restored = self._prepare(nn.Sequential(nn.Conv1d(3, 5, 1)))
        incompatible = restored.load_state_dict(saved_state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
