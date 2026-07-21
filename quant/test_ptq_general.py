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
    _copy_activation_domain,
    _pointnet2_aggregation_operator,
    _verify_feature_requant_codebook,
    apply_hardware_domain_lock,
    audit_quantized_operators,
    bind_hardware_dp_qparams,
    collect_hardware_feature_requants,
    cpu_state_dict,
    get_hardware_qat_qconfig,
    get_qat_module_mapping,
    set_hardware_geometry_mode,
)
from openpoints.models.layers.group import QueryAndGroup  # noqa: E402


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
            self.assertEqual(
                module.weight_fake_quant.qscheme,
                torch.per_channel_symmetric,
            )
            self.assertEqual(module.weight_fake_quant.quant_min, -128)
            self.assertEqual(module.weight_fake_quant.quant_max, 127)
            self.assertEqual(
                module.activation_post_process.dtype, torch.quint8
            )
            self.assertIn(module.activation_post_process.qscheme, (
                torch.per_tensor_affine,
                torch.per_tensor_symmetric,
            ))
            self.assertEqual(module.activation_post_process.quant_min, 0)
            self.assertEqual(module.activation_post_process.quant_max, 255)

    def test_calibrated_weight_buffers_are_hardware_symmetric(self):
        model = nn.ModuleDict({
            "conv1d": nn.Conv1d(3, 5, 1),
            "conv_relu1d": nni.ConvReLU1d(
                nn.Conv1d(3, 5, 1), nn.ReLU()
            ),
            "conv2d": nn.Conv2d(3, 5, 1),
            "linear": nn.Linear(5, 2),
        })
        self._prepare(model)
        model.apply(quant.disable_fake_quant)
        model.apply(quant.enable_observer)

        model["conv1d"](torch.randn(2, 3, 8))
        model["conv_relu1d"](torch.randn(2, 3, 8))
        model["conv2d"](torch.randn(2, 3, 4, 4))
        model["linear"](torch.randn(2, 5))
        model.apply(quant.enable_fake_quant)
        model.apply(quant.disable_observer)

        operators = audit_quantized_operators(
            model, stage="calibrated unit test",
            require_calibrated_buffers=True,
        )
        for _, module in operators:
            fake_quant = module.weight_fake_quant
            output_channels = module.weight.shape[0]
            self.assertEqual(fake_quant.scale.numel(), output_channels)
            self.assertEqual(fake_quant.zero_point.numel(), output_channels)
            self.assertTrue(torch.all(fake_quant.scale > 0))
            self.assertTrue(torch.all(fake_quant.zero_point == 0))
            calculated_scale, calculated_zero_point = (
                fake_quant.calculate_qparams()
            )
            self.assertTrue(torch.equal(
                fake_quant.scale,
                calculated_scale.to(dtype=fake_quant.scale.dtype),
            ))
            self.assertTrue(torch.equal(
                fake_quant.zero_point.to(dtype=torch.int64),
                calculated_zero_point.to(dtype=torch.int64),
            ))

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
        model.apply(quant.enable_fake_quant)
        model.apply(quant.disable_observer)
        audit_quantized_operators(
            model, stage="before round trip", require_calibrated_buffers=True
        )

        serialized = io.BytesIO()
        torch.save({"model_state_dict": cpu_state_dict(model)}, serialized)
        serialized.seek(0)
        saved_state = torch.load(serialized)["model_state_dict"]

        restored = self._prepare(nn.Sequential(nn.Conv1d(3, 5, 1)))
        incompatible = restored.load_state_dict(saved_state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        audit_quantized_operators(
            restored, stage="after round trip", require_calibrated_buffers=True
        )

        fake_quant = restored[0].weight_fake_quant
        fake_quant.scale[0] = torch.nextafter(
            torch.nextafter(
                fake_quant.scale[0],
                torch.tensor(float("inf"), dtype=fake_quant.scale.dtype),
            ),
            torch.tensor(float("inf"), dtype=fake_quant.scale.dtype),
        )
        with self.assertRaisesRegex(RuntimeError, "scale buffer disagrees"):
            audit_quantized_operators(
                restored, stage="corrupt round trip",
                require_calibrated_buffers=True,
            )

    def test_hardware_domain_lock_copies_exact_fake_quant_behavior(self):
        qconfig = get_hardware_qat_qconfig("fbgemm")
        source = qconfig.activation()
        target = qconfig.activation()
        source(torch.tensor([-1.0, 0.0, 2.0, 4.0]))
        target(torch.tensor([-20.0, 0.0, 20.0, 40.0]))
        source.disable_observer()
        target.disable_observer()

        record = _copy_activation_domain(
            target, source, "target", "source"
        )

        source_scale, source_zero_point = source.calculate_qparams()
        target_scale, target_zero_point = target.calculate_qparams()
        self.assertTrue(torch.equal(target_scale, source_scale))
        self.assertTrue(torch.equal(target_zero_point, source_zero_point))
        probe = torch.tensor([-3.0, -0.25, 0.0, 0.75, 5.0])
        self.assertTrue(torch.equal(target(probe), source(probe)))
        self.assertEqual(record["target"], "target")
        self.assertEqual(record["source"], "source")

    def test_hardware_domain_lock_rejects_unknown_ablation_scope(self):
        with self.assertRaisesRegex(ValueError, "invalid hardware domain lock scope"):
            apply_hardware_domain_lock(nn.Module(), scope="partial")

    def test_pointnet2_encoder_uses_feature_requant_instead_of_domain_copy(self):
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.quant_input = quant.QuantStub()
        model.encoder.SA_modules = nn.ModuleList()
        operators = []
        for _ in range(2):
            implementation = nn.Module()
            implementation.quant_feat = quant.QuantStub()
            implementation.convs = nn.Sequential(nn.Conv2d(1, 1, 1))
            wrapper = nn.Module()
            wrapper.SA_CONFIG_operator = implementation
            stage = nn.Module()
            stage.local_aggregations = nn.ModuleList([wrapper])
            model.encoder.SA_modules.append(stage)
            operators.append(implementation)

        self._prepare(model)
        root_values = torch.tensor([0.0, 1.0, 2.0, 4.0])
        model.encoder.quant_input(root_values)
        operators[0].quant_feat(root_values)
        operators[0].convs[0].activation_post_process(
            torch.tensor([0.0, 0.5, 1.0, 2.0])
        )
        operators[1].quant_feat(torch.tensor([-1.0, 0.0, 1.0, 3.0]))
        operators[1].convs[0].activation_post_process(
            torch.tensor([0.0, 1.0, 2.0, 5.0])
        )
        model.apply(quant.enable_fake_quant)
        model.apply(quant.disable_observer)

        # Use the calibrated S1 SA2 domains, whose complete uint8 codebook is
        # admissible under the RTL positive-half rounding contract. The raw
        # observer fixture above produces an exact 1/2 ratio and therefore a
        # deliberate ties-to-even mismatch at odd source codes.
        source = operators[0].convs[0].activation_post_process
        target = operators[1].quant_feat.activation_post_process
        source.scale.fill_(0.013760093599557877)
        source.zero_point.fill_(0)
        target.scale.fill_(0.014518913812935352)
        target.zero_point.fill_(13)
        target_before = (target.scale.clone(), target.zero_point.clone())
        ties = apply_hardware_domain_lock(model, scope="all")
        records = collect_hardware_feature_requants(model, scope="all")

        self.assertEqual(ties, [])
        self.assertTrue(torch.equal(target.scale, target_before[0]))
        self.assertTrue(torch.equal(target.zero_point, target_before[1]))
        self.assertEqual([record["stage"] for record in records], ["SA1", "SA2"])
        self.assertTrue(records[0]["identity"])
        self.assertEqual((records[0]["m0"], records[0]["shift"]), (1, 0))
        self.assertFalse(records[1]["identity"])
        self.assertEqual(records[1]["source_zero_point"], 0)
        self.assertGreater(records[1]["target_zero_point"], 0)
        self.assertTrue(records[1]["uint8_codebook_bit_exact"])

    def test_feature_requant_rejects_a_non_bit_exact_multiplier(self):
        with self.assertRaisesRegex(RuntimeError, "not bit-exact"):
            _verify_feature_requant_codebook(
                0.013760093599557877, 0,
                0.014518913812935352, 13,
                1, 0, "bad SA2",
            )

    def test_feature_requant_rejects_a_half_even_tie_domain(self):
        with self.assertRaisesRegex(RuntimeError, "not bit-exact"):
            _verify_feature_requant_codebook(
                1.0, 0, 2.0, 64,
                1 << 30, 31, "half-tie SA2",
            )

    def test_pointnet2_aggregation_operator_is_explicitly_unwrapped(self):
        implementation = nn.Module()
        implementation.quant_feat = quant.QuantStub()
        implementation.convs = nn.Sequential(nn.Conv2d(6, 8, 1))
        wrapper = nn.Module()
        wrapper.SA_CONFIG_operator = implementation

        self.assertIs(
            _pointnet2_aggregation_operator(wrapper, "test stage"),
            implementation,
        )
        with self.assertRaisesRegex(RuntimeError, "SA_CONFIG_operator"):
            _pointnet2_aggregation_operator(nn.Module(), "bad stage")

    def test_hardware_geometry_toggle_clears_runtime_state(self):
        model = nn.Module()
        model.grouper = QueryAndGroup(radius=0.2, nsample=8)
        model.grouper.coord_step = torch.ones(1, 1, 1)
        model.grouper.coord_dp_qparams = (0.01, 128)

        toggled = set_hardware_geometry_mode(model, True)

        self.assertEqual(toggled, 1)
        self.assertTrue(model.grouper.coord_hardware_exact)
        self.assertIsNone(model.grouper.coord_step)
        self.assertIsNone(model.grouper.coord_dp_qparams)

    def test_hardware_dp_binding_uses_runtime_fake_quant_buffers(self):
        owner = nn.Module()
        owner.grouper = QueryAndGroup(radius=0.2, nsample=8)
        owner.grouper.coord_hardware_exact = True
        owner.quant_feat = quant.QuantStub()
        owner.qconfig = get_hardware_qat_qconfig("fbgemm")
        owner.train()
        quant.prepare_qat(owner, inplace=True)
        owner.quant_feat(torch.tensor([-1.0, 0.0, 2.0, 4.0]))
        owner.apply(quant.disable_observer)

        bindings = bind_hardware_dp_qparams(owner)

        fake_quant = owner.quant_feat.activation_post_process
        expected = (
            float(fake_quant.scale.item()),
            int(fake_quant.zero_point.item()),
        )
        self.assertEqual(owner.grouper.coord_dp_qparams, expected)
        self.assertEqual(len(bindings), 1)
        self.assertEqual(bindings[0]["scale"], expected[0])
        self.assertEqual(bindings[0]["zero_point"], expected[1])


if __name__ == "__main__":
    unittest.main()
