import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from kdpoint_geometry import (  # noqa: E402
    cordic_sqrt_distance_q15,
    dp_quant_config,
    dp_quantize_scalar,
    isotropic_quantize,
    reconstruct_q9,
    sidecar_delta,
    sidecar_distance_sq,
    split_q9_sidecar,
    strict_radius_sq,
    wfu_weights_exact,
)
from openpoints.models.layers.kdpoint_geometry import (  # noqa: E402
    cordic_sqrt_distance_q15_tensor,
    dp_requantize_q9,
    isotropic_q9_encode,
    wfu_weights_exact_tensor,
)
from openpoints.models.layers.group import GroupAll  # noqa: E402


class KDPointGeometryTest(unittest.TestCase):
    def test_isotropic_grid_does_not_stretch_short_axes(self):
        points = torch.tensor([[[0.0, 0.0, -1.0],
                                [10.0, 1.0, 0.0],
                                [5.0, 0.5, 1.0]]])
        encoding = isotropic_quantize(points)
        self.assertEqual(encoding.codes[0, 1, 0].item(), 511)
        self.assertEqual(encoding.codes[0, 1, 1].item(), 51)
        self.assertEqual(encoding.codes[0, 2, 2].item(), 102)
        self.assertAlmostEqual(encoding.step.item(), 10.0 / 511.0)

    def test_q9_sidecar_reconstructs_deltas_and_distances(self):
        codes = torch.tensor([[0, 1, 2], [255, 256, 511], [37, 90, 401]])
        main, sidecar = split_q9_sidecar(codes)
        self.assertTrue(torch.equal(reconstruct_q9(main, sidecar), codes))
        delta = sidecar_delta(main[2], sidecar[2], main[0], sidecar[0])
        self.assertTrue(torch.equal(delta, codes[2] - codes[0]))
        expected = ((codes[2] - codes[0]).to(torch.int64).square().sum())
        actual = sidecar_distance_sq(
            main[2], sidecar[2], main[0], sidecar[0]
        )
        self.assertEqual(actual.item(), expected.item())

    def test_strict_radius_threshold_matches_less_than_at_integer_boundary(self):
        step = torch.tensor(0.1, dtype=torch.float64)
        threshold = strict_radius_sq(
            torch.tensor(0.2, dtype=torch.float64), step
        )
        self.assertEqual(threshold.item(), 3)
        self.assertTrue(3 <= threshold.item())
        with self.assertRaisesRegex(ValueError, "float64"):
            strict_radius_sq(torch.tensor(0.2), step)
        self.assertFalse(4 <= threshold.item())

    def test_dynamic_dp_config_uses_scene_step_and_target_domain(self):
        config = dp_quant_config(
            source_step=1.0 / 512.0,
            target_scale=1.0 / 128.0,
            target_zero_point=128,
            radius=0.25,
        )
        self.assertEqual(config["m0"], 1)
        self.assertEqual(config["shift"], 0)
        self.assertEqual(dp_quantize_scalar(-128, config), 0)
        self.assertEqual(dp_quantize_scalar(0, config), 128)
        self.assertEqual(dp_quantize_scalar(127, config), 255)

    def test_cordic_sqrt_tracks_q15_distance_scale(self):
        for distance_sq in (1, 2, 4, 17, 195075):
            actual = cordic_sqrt_distance_q15(distance_sq)
            expected = math.sqrt(distance_sq) * (1 << 15)
            self.assertLess(abs(actual - expected) / expected, 0.003)

    def test_wfu_exact_weights_are_u1_7_and_zero_first(self):
        self.assertEqual(wfu_weights_exact((100, 100, 100)), (44, 42, 42))
        self.assertEqual(wfu_weights_exact((0, 0, 9)), (128, 0, 0))
        self.assertEqual(wfu_weights_exact((5, 0, 9)), (0, 128, 0))
        weights = wfu_weights_exact((4, 16, 64))
        self.assertEqual(sum(weights), 128)
        self.assertGreaterEqual(weights[0], weights[1])
        self.assertGreaterEqual(weights[1], weights[2])

    def test_vectorized_runtime_matches_scalar_oracle(self):
        vectors = torch.tensor([
            [100, 100, 100],
            [4, 16, 64],
            [0, 5, 9],
            [5, 0, 9],
            [1, 2, 3],
            [1, 195074, 195075],
        ], dtype=torch.int64)
        roots = cordic_sqrt_distance_q15_tensor(vectors)
        for row in range(vectors.shape[0]):
            for col in range(3):
                self.assertEqual(
                    roots[row, col].item(),
                    cordic_sqrt_distance_q15(vectors[row, col].item()),
                )
        expected = torch.tensor([
            wfu_weights_exact(tuple(row.tolist())) for row in vectors
        ], dtype=torch.int64)
        self.assertTrue(torch.equal(wfu_weights_exact_tensor(vectors), expected))

    def test_runtime_isotropic_encoding_and_dynamic_dp(self):
        points = torch.tensor([[
            [-2.0, 5.0, 1.0],
            [2.0, 6.0, 1.5],
        ]], dtype=torch.float32)
        codes, dequantized, step = isotropic_q9_encode(points)
        reference = isotropic_quantize(points)
        self.assertTrue(torch.equal(codes, reference.codes))
        self.assertTrue(torch.equal(
            dequantized,
            (reference.codes.to(torch.float64) * reference.step
             + reference.origin).to(torch.float32),
        ))
        delta = torch.tensor([[[[-128, 0, 127]]]], dtype=torch.int64)
        output = dp_requantize_q9(
            delta, torch.tensor([[[1.0 / 512.0]]], dtype=torch.float64),
            radius=0.25, target_scale=1.0 / 128.0,
            target_zero_point=128, normalize_dp=True,
        )
        self.assertTrue(torch.equal(
            output,
            torch.tensor([[[[-1.0, 0.0, 127.0 / 128.0]]]]),
        ))

    def test_group_all_restores_physical_xyz_from_q9(self):
        points = torch.tensor([[
            [-2.0, 5.0, 1.0],
            [2.0, 6.0, 1.5],
        ]], dtype=torch.float32)
        codes, dequantized, step = isotropic_q9_encode(points)
        group_all = GroupAll()
        group_all.coord_hardware_exact = True
        group_all.coord_step = step
        group_all.coord_origin = points.to(torch.float64).amin(
            dim=1, keepdim=True
        )
        grouped_xyz, grouped_features = group_all(
            codes.to(torch.float32), codes.to(torch.float32), None
        )
        self.assertIsNone(grouped_features)
        self.assertTrue(torch.equal(
            grouped_xyz,
            dequantized.transpose(1, 2).unsqueeze(2),
        ))


if __name__ == "__main__":
    unittest.main()
