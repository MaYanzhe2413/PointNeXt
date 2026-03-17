 # Post-Processing Unit (PPU) Hardware Specification

## Overview

This document specifies the Post-Processing Unit (PPU) that sits between the systolic array output and the next layer's input buffer. The PPU converts the INT32 accumulator result from the MAC array into a uint8 output suitable for the next layer.

The design is based on PyTorch's fbgemm quantization scheme (Jacob et al., arXiv:1712.05877), which is the industry standard used by Google TPU, ARM CMSIS-NN, and PyTorch Mobile.

---

## System Context

```
                         ┌─────────────────────┐
  a_q[uint8] ──────────→ │                     │
                         │   Systolic Array     │──→ acc[int32]
  w_q[int8]  ──────────→ │   (INT8 MAC)        │    per output channel
                         │                     │
                         └─────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
  col_sum_w[int32] ────→ │                     │
  Z_a[uint8]       ────→ │  Post-Processing    │
  bias_q[int32]    ────→ │  Unit (PPU)         │──→ o_q[uint8]
  M0[int32]        ────→ │                     │    to next layer
  shift[int8]      ────→ │  << THIS DOCUMENT   │
  Z_o[uint8]       ────→ │                     │
                         └─────────────────────┘
```

---

## Data Types Summary

| Signal        | Type   | Bit Width | Source              | Per-Channel? | Runtime/Static |
|---------------|--------|-----------|---------------------|--------------|----------------|
| acc           | int32  | 32        | Systolic array out  | Yes          | Runtime        |
| a_q           | uint8  | 8         | Input activation    | No           | Runtime        |
| w_q           | int8   | 8         | Weight SRAM         | Yes          | Static         |
| col_sum_w     | int32  | 32        | Precomputed         | Yes          | Static         |
| Z_a           | uint8  | 8         | Prev layer param    | No (scalar)  | Static*        |
| bias_q        | int32  | 32        | Precomputed         | Yes          | Static         |
| M0            | int32  | 32        | Precomputed         | Yes          | Static         |
| shift         | uint8  | 6 (0~63)  | Precomputed         | Yes          | Static         |
| Z_o           | uint8  | 8         | This layer param    | No (scalar)  | Static         |
| o_q           | uint8  | 8         | PPU output          | Yes          | Runtime        |

*Static = loaded once per layer, does not change during inference of that layer.
*Z_a changes per layer but is constant within a layer.

---

## PPU Pipeline Stages

The PPU processes one output channel at a time (or multiple channels in parallel if area allows). Each output channel goes through 5 stages:

### Stage 1: Zero-Point Compensation (INT32 subtract)

**Purpose**: The systolic array computes `Σ(a_q × w_q)`, but we need `Σ((a_q - Z_a) × w_q)`. This stage corrects for the input zero-point.

```
acc = acc - Z_a × col_sum_w[ch]
```

**Hardware**: One INT32 multiplier + one INT32 subtractor.

**Note on col_sum_w**: This is the sum of all weight integers for one output channel:
```
col_sum_w[ch] = Σ w_q[ch, c_in, k]   (sum over all input channels and kernel positions)
```
It is precomputed offline when loading the model. For a Conv1d(C_in, C_out, kernel=1), col_sum_w has C_out entries, each being the sum of C_in int8 values.

**Why not subtract Z_a before the MAC?** Subtracting Z_a from each a_q before multiplication would require a subtractor at every PE input, and would widen a_q from uint8 to int16, complicating the MAC datapath. Compensating once after accumulation is cheaper.

**Important**: In the fbgemm scheme, weight zero_point Z_w = 0 (symmetric quantization for weights). Therefore the terms `Z_w × Σ(a_q)` and `N × Z_a × Z_w` are both zero and do NOT need hardware support.

### Stage 2: Bias Addition (INT32 add)

```
acc = acc + bias_q[ch]
```

**Hardware**: One INT32 adder.

**Note on bias_q**: The original FP32 bias is pre-quantized into int32:
```
bias_q[ch] = round(bias_fp32[ch] / (S_a × S_w[ch]))
```
This is computed offline. If the layer has no bias, bias_q = 0.

### Stage 3: Requantization Multiply (INT32 × INT32, take upper bits)

```
acc = (acc × M0[ch]) >> shift[ch]
```

**Hardware**: One INT32 × INT32 → INT64 multiplier, then right-shift and truncate to INT32.

**Note on M0 and shift**: The requantization scale M is a floating-point value:
```
M[ch] = (S_a × S_w[ch]) / S_o
```
Since 0 < M < 1 in practice, it is decomposed into fixed-point form:
```
M[ch] = M0[ch] × 2^(-shift[ch])
```
where M0 is an int32 in the range [2^30, 2^31) (i.e., the leading bit is in position 30 or 31), and shift is a small positive integer (typically 0~40).

**Precomputation (offline)**:
```python
def decompose_M(M_float):
    """Convert float M to fixed-point M0 and shift."""
    assert 0 < M_float < 1
    shift = 0
    while M_float < 0.5:
        M_float *= 2
        shift += 1
    M0 = round(M_float * (2**31))  # M0 is ~2^31 scale
    shift += 31
    return M0, shift
```

**Rounding**: Use "round-half-to-even" (banker's rounding) on the shifted result for best accuracy. If hardware cost is a concern, "round-half-up" (add 1<<(shift-1) before shifting) is acceptable:
```
acc = (acc × M0[ch] + (1 << (shift[ch] - 1))) >> shift[ch]
```

### Stage 4: Output Zero-Point Addition (INT32 add)

```
acc = acc + Z_o
```

**Hardware**: One INT32 adder (Z_o is a scalar, same for all channels in this layer).

### Stage 5: Saturating Clamp to uint8

```
o_q = clamp(acc, 0, 255)
```

**Hardware**: Comparator + mux.
```
if acc < 0:     o_q = 0
elif acc > 255: o_q = 255
else:           o_q = acc[7:0]
```

---

## Complete Pipeline Diagram

```
acc[int32]  (from systolic array, one per output channel)
    │
    ▼
┌──────────────────────────────────┐
│ Stage 1: ZP Compensation        │
│                                  │
│   ┌─────┐   ┌───────────┐       │
│   │ Z_a │──→│  INT32    │       │
│   └─────┘   │  MUL      │──→(-)│──→ acc'
│   ┌─────────┐│           │   ▲  │
│   │col_sum_w│→           │   │  │
│   │  [ch]   │└───────────┘  acc │
│   └─────────┘                   │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ Stage 2: Bias Add               │
│                                  │
│   acc' = acc' + bias_q[ch]      │
│                                  │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ Stage 3: Requantize             │
│                                  │
│   tmp[int64] = acc' × M0[ch]    │
│   acc'' = (tmp + rounding_const)│
│            >> shift[ch]          │
│                                  │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ Stage 4: Output ZP Add          │
│                                  │
│   acc'' = acc'' + Z_o            │
│                                  │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ Stage 5: Clamp                  │
│                                  │
│   o_q = clamp(acc'', 0, 255)    │
│                                  │
└──────────────────────────────────┘
    │
    ▼
o_q[uint8]  → output buffer / next layer input
```

---

## Per-Layer Static Parameters (stored in SRAM/registers)

For each layer, the following parameters must be loaded before processing:

| Parameter       | Shape           | Bit Width | Example (Conv1d 4→32, k=1) |
|-----------------|-----------------|-----------|----------------------------|
| col_sum_w[ch]   | [C_out]         | 32        | [32] int32 values          |
| bias_q[ch]      | [C_out]         | 32        | [32] int32 values          |
| M0[ch]          | [C_out]         | 32        | [32] int32 values          |
| shift[ch]       | [C_out]         | 6         | [32] uint6 values          |
| Z_a             | scalar          | 8         | 1 uint8 value              |
| Z_o             | scalar          | 8         | 1 uint8 value              |

**Total SRAM per layer** ≈ C_out × (32+32+32+6)/8 + 2 bytes ≈ C_out × 12.75 + 2 bytes.
For C_out=512 (largest layer in PointNeXt-S): ~6.4 KB.

---

## Precomputation Formulas (done by software at model load time)

Given PyTorch quantized model parameters for each layer:

```
Inputs from PyTorch model:
  S_a     = input activation scale    (from previous layer or QuantStub)
  Z_a     = input activation zp       (from previous layer or QuantStub)
  S_w[ch] = weight scale per channel  (from conv.weight().q_per_channel_scales())
  Z_w[ch] = weight zp per channel     (should be all zeros for fbgemm)
  S_o     = output activation scale   (from this conv module's .scale)
  Z_o     = output activation zp      (from this conv module's .zero_point)
  bias    = FP32 bias                 (from original model, or None)
  w_q     = INT8 weight tensor        (from conv.weight().int_repr())

Compute:
  col_sum_w[ch] = sum(w_q[ch, :, :])               # int32, sum over in_channels and kernel
  bias_q[ch]    = round(bias[ch] / (S_a * S_w[ch])) # int32, or 0 if no bias
  M[ch]         = (S_a * S_w[ch]) / S_o              # float, requantization scale
  M0[ch], shift[ch] = decompose_M(M[ch])             # fixed-point decomposition
```

---

## Numerical Example

Using actual values from the PointNeXt-S quantized model (first Conv1d layer):

```
Layer: encoder.encoder.0.0.convs.0.0  (Conv1d: 4 → 32, kernel=1)

Input:  S_a = 0.0445,  Z_a = 61     (from quant_input)
Output: S_o = 0.0286,  Z_o = 58     (from this Conv1d)

Weight channel 0: w_q = [127, -116, -34, -31]  (int8)
                  S_w[0] = (read from model)

Precompute:
  col_sum_w[0] = 127 + (-116) + (-34) + (-31) = -54
  M[0] = (0.0445 × S_w[0]) / 0.0286

Runtime (for one input sample, channel 0):
  Step 1: acc = systolic_result                        # e.g., 1523
  Step 2: acc = acc - 61 × (-54) = 1523 + 3294 = 4817 # ZP compensation
  Step 3: acc = acc + bias_q[0]                        # e.g., 4817 + 12 = 4829
  Step 4: acc = (4829 × M0[0]) >> shift[0]             # e.g., 73
  Step 5: acc = 73 + 58 = 131                          # add Z_o
  Step 6: o_q = clamp(131, 0, 255) = 131               # output uint8
```

---

## Special Cases

### 1. ConvReLU Fused Layers

For layers like `QuantizedConvReLU1d` / `QuantizedConvReLU2d`, the ReLU is fused into the clamp:

```
o_q = clamp(acc, 0, 255)   →   o_q = clamp(acc, Z_o, 255)
```

Since ReLU means output_real ≥ 0, and real=0 maps to q=Z_o, the lower bound becomes Z_o instead of 0. In practice, for ConvReLU layers Z_o = 0 (PyTorch observer learns this), so the clamp is identical: `clamp(acc, 0, 255)`.

**Hardware implication**: No change needed. The clamp lower bound can be a configurable register (0 for ConvReLU, 0 for regular Conv with unsigned output).

### 2. Layers Without Bias

Set `bias_q[ch] = 0` for all channels. Stage 2 can be skipped (or the adder just adds 0).

### 3. Geometry Operations (ball_query, FPS, KNN)

These operations are NOT quantized. They operate on FP32 coordinates (xyz positions). The PPU is not involved. Data crosses the quantized/FP32 boundary via DeQuantStub/QuantStub pairs in the model:
- Before geometry op: DeQuantStub converts uint8 features → FP32
- After geometry op: QuantStub converts FP32 features → uint8

**Hardware implication**: You need a small FP32 datapath (or dedicated module) for geometry operations, separate from the quantized MAC + PPU pipeline.

### 4. Residual Addition (QAdd)

PointNeXt uses residual connections: `output = conv_result + skip_connection`. Both operands are quantized with different scales. PyTorch uses `QFunctional.add()` which has its own output scale/zp.

The hardware for this is:
```
# Both inputs are uint8 with different scales
a_real = (a_q - Z_a) × S_a
b_real = (b_q - Z_b) × S_b

# Add in "real" domain, then requantize
sum_real = a_real + b_real
sum_q = clamp(round(sum_real / S_out) + Z_out, 0, 255)
```

This requires dequantize-add-requantize, which involves two FP32 multiplies (or two fixed-point rescales). It is a separate functional unit from the conv PPU.

### 5. Concatenation (QCat)

Similar to QAdd, but concatenates along the channel dimension. Each input chunk may have a different scale. The hardware must rescale each chunk to the common output scale before writing to the output buffer.

---

## Latency and Throughput Considerations

- **Stages 1-5 are simple INT32 operations**: Each stage is 1 cycle if pipelined.
- **The bottleneck is Stage 3** (INT32 × INT32 multiply): If a single-cycle 32×32 multiplier is too expensive, consider:
  - Using a 16×16 multiplier with 2-cycle multiply
  - Time-sharing one multiplier across multiple channels
- **Throughput target**: The PPU should match the systolic array's output rate. If the array produces one channel result per cycle, the PPU pipeline should also sustain one channel per cycle.

---

## Summary: What Software Provides, What Hardware Does

### Software (model compiler / runtime loader) provides per layer:
1. `col_sum_w[C_out]` — int32 array, precomputed weight column sums
2. `bias_q[C_out]` — int32 array, pre-quantized biases
3. `M0[C_out]` — int32 array, fixed-point requant multipliers
4. `shift[C_out]` — uint6 array, right-shift amounts
5. `Z_a` — uint8 scalar, input zero-point
6. `Z_o` — uint8 scalar, output zero-point

### Hardware (PPU) does at runtime:
1. `acc -= Z_a × col_sum_w[ch]` — 1 MUL + 1 SUB (int32)
2. `acc += bias_q[ch]` — 1 ADD (int32)
3. `acc = (acc × M0[ch]) >> shift[ch]` — 1 MUL (int32×int32→int64) + 1 SHIFT
4. `acc += Z_o` — 1 ADD (int32)
5. `o_q = clamp(acc, 0, 255)` — comparator + mux
