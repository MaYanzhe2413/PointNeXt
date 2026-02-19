"""
Quantization-Aware Training (QAT) script for PointNeXt.

Usage:
    python tools/train_qat.py \
        --cfg cfgs/s3dis/pointnext-s.yaml \
        --pretrained /path/to/checkpoint.pth \
        --qat_epochs 10 \
        --freeze_bn_epoch 5

Workflow:
    1. Build model from config
    2. Load pretrained FP32 weights
    3. Fuse Conv-BN modules
    4. Prepare model for QAT (insert FakeQuantize observers)
    5. Fine-tune with quantization simulation
    6. Convert to fully quantized INT8 model
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.quantization as quant

# Add project root to path
DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
sys.path.insert(0, ROOT)

from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg
from openpoints.models.layers.quant_utils import (
    swap_custom_convs_to_standard,
    fuse_convbn_modules,
    disable_quantization_for_geometry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description='QAT for PointNeXt'
    )
    p.add_argument(
        '--cfg', type=str, required=True,
        help='Path to YAML config file'
    )
    p.add_argument(
        '--pretrained', type=str, default=None,
        help='Path to pretrained FP32 checkpoint'
    )
    p.add_argument(
        '--qat_epochs', type=int, default=10,
        help='Number of QAT fine-tuning epochs'
    )
    p.add_argument(
        '--freeze_bn_epoch', type=int, default=5,
        help='Epoch to freeze BN stats and observers'
    )
    p.add_argument(
        '--lr', type=float, default=1e-5,
        help='Learning rate for QAT fine-tuning'
    )
    p.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size'
    )
    p.add_argument(
        '--output', type=str, default='qat_output',
        help='Output directory'
    )
    p.add_argument(
        '--backend', type=str, default='qnnpack',
        choices=['qnnpack', 'fbgemm', 'x86'],
        help='Quantization backend'
    )
    return p.parse_args()


def prepare_model_for_qat(model, backend='qnnpack'):
    """Prepare a PointNeXt model for QAT.

    Steps:
        1. Fuse Conv-BN(-ReLU)
        2. Pin geometry ops to FP32
        3. Set qconfig
        4. prepare_qat()
    """
    model.train()

    # Step 0: Swap custom Conv wrappers to standard
    logger.info('Swapping custom Conv to standard...')
    swap_custom_convs_to_standard(model)

    # Step 1: Fuse Conv-BN
    logger.info('Fusing Conv-BN modules...')
    fuse_convbn_modules(model)

    # Step 2: Pin geometry ops to FP32
    logger.info(
        'Disabling quantization for geometry ops...'
    )
    disable_quantization_for_geometry(model)

    # Step 3: Set qconfig
    torch.backends.quantized.engine = backend
    model.qconfig = quant.get_default_qat_qconfig(
        backend
    )

    # Step 4: Insert FakeQuantize observers
    logger.info('Preparing model for QAT...')
    quant.prepare_qat(model, inplace=True)

    return model


def convert_to_quantized(model):
    """Convert QAT model to quantized INT8."""
    model.eval()
    quant.convert(model, inplace=True)
    return model


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 1. Load config and build model
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    model = build_model_from_cfg(cfg.model)
    logger.info(
        f'Built model: {cfg.model.NAME}'
    )

    # 2. Load pretrained weights
    if args.pretrained:
        ckpt = torch.load(
            args.pretrained, map_location='cpu'
        )
        if 'model' in ckpt:
            ckpt = ckpt['model']
        missing, unexpected = model.load_state_dict(
            ckpt, strict=False
        )
        if missing:
            logger.warning(
                f'Missing keys: {missing[:5]}...'
            )
        logger.info('Loaded pretrained weights')

    # 3. Prepare for QAT
    model = prepare_model_for_qat(
        model, backend=args.backend
    )
    logger.info('Model prepared for QAT')

    # 4. Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )

    # 5. QAT training loop (skeleton)
    logger.info(
        f'Starting QAT for {args.qat_epochs} epochs'
    )
    for epoch in range(args.qat_epochs):
        model.train()

        # Freeze BN and observers after threshold
        if epoch >= args.freeze_bn_epoch:
            model.apply(
                torch.quantization.disable_observer
            )
            model.apply(torch.quantization.fake_quantize.disable_observer)

        # TODO: Add your dataloader here
        # for batch in dataloader:
        #     data = batch['x']     # (B, N, C)
        #     target = batch['y']   # (B, N)
        #     pred = model(data)
        #     loss = criterion(pred, target)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        logger.info(f'Epoch {epoch+1}/{args.qat_epochs}')

        # Save QAT checkpoint
        ckpt_path = os.path.join(
            args.output,
            f'qat_epoch_{epoch+1}.pth'
        )
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
        }, ckpt_path)

    # 6. Convert to quantized model
    logger.info('Converting to quantized model...')
    model_int8 = convert_to_quantized(model)

    # Save final quantized model
    out_path = os.path.join(
        args.output, 'model_quantized.pth'
    )
    torch.save(model_int8.state_dict(), out_path)
    logger.info(f'Saved quantized model to {out_path}')

    # Print model size comparison
    fp32_size = os.path.getsize(
        args.pretrained
    ) if args.pretrained else 0
    q_size = os.path.getsize(out_path)
    logger.info(
        f'FP32 size: {fp32_size/1e6:.1f} MB, '
        f'INT8 size: {q_size/1e6:.1f} MB, '
        f'Ratio: {q_size/max(fp32_size,1):.2f}x'
    )


if __name__ == '__main__':
    main()
