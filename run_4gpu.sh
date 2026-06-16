#!/bin/bash
# ============================================================================
# 6 格实验分到 4 张 GPU 并行 (每格 = coord ladder + INT8 PTQ)
# 均衡: 两个 partseg(最慢)各占一卡, 两对 seg+cls 各占一卡
#   GPU0: pn2_shape          GPU1: pnx_s3dis + pnx_mn40
#   GPU3: pnx_shape          GPU2: pn2_s3dis + pn2_mn40
#
# 用法: bash run_4gpu.sh            # 后台起 4 个 setsid 任务
#       DRY=1 bash run_4gpu.sh      # 只打印
# 监控: tail -f gpu*.log   /   watch -n5 nvidia-smi
# ============================================================================
SEED=7895
DRY="${DRY:-0}"

# ---- checkpoint 目录 (与 run_all_remaining.sh 一致) ----
CK_PNX_S3DIS="log/s3dis/s3dis-train-pointnext-s-ngpus3-20260612-102404-GmdxU7SJa3BUuadeLTdQgY"
CK_PNX_MN40="log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus4-seed7895-20260331-221531-JvuawAeqYqbwPKDYod2R2f"
CK_PNX_SHAPE="log/shapenetpart/shapenetpart-train-pointnext-s_baseline-ngpus2-seed7895-20260325-143647-7ym2g4MNj4VqbreHGx4N92"
CK_PN2_S3DIS="log/s3dis/s3dis-train-pointnet++-ngpus3-20260615-103940-kKwLcsaKbXWhfZuMKrxcNt"
CK_PN2_MN40="log/modelnet40ply2048/modelnet40ply2048-train-pointnet++-ngpus4-seed527-20250914-155417-bebhZhPnyPhBmLNisKNyQG"
CK_PN2_SHAPE="log/shapenetpart/shapenetpart-train-pointnet++-ngpus2-seed7895-20260615-220231-6DEjnjKUveG6UEHbGkyF4z"

pth() { echo "$1/checkpoint/$(basename $1)_ckpt_best.pth"; }

# 一格 = ladder + PTQ. 参数: gpu task model dataset leaf ckptdir tag
one_cell() {
  local g="$1" task="$2" model="$3" ds="$4" leaf="$5" cdir="$6" tag="$7"
  local ckpt=$(pth "$cdir")
  echo "bash batch_ablation_general.sh --task $task --model $model --dataset $ds -g $g -S $SEED -p $ckpt --sweep && CUDA_VISIBLE_DEVICES=$g python quant/ptq_general.py --task $task --cfg cfgs/$ds/$model.yaml --ckpt $ckpt --tag ${tag}_blockfps model.encoder_args.sampler=kdtree model.encoder_args.sampler_args.leaf_size=$leaf model.encoder_args.sampler_args.strategy=fps"
}

# 每卡的任务串 (一卡可串多格)
GPU0=$(one_cell 0 partseg pointnet++ shapenetpart 128 "$CK_PN2_SHAPE" pn2_shape)
GPU1="$(one_cell 1 seg pointnext-s s3dis 1500 "$CK_PNX_S3DIS" pnx_s3dis) && $(one_cell 1 cls pointnext-s modelnet40ply2048 64 "$CK_PNX_MN40" pnx_mn40)"
GPU2="$(one_cell 2 seg pointnet++ s3dis 1500 "$CK_PN2_S3DIS" pn2_s3dis) && $(one_cell 2 cls pointnet++ modelnet40ply2048 64 "$CK_PN2_MN40" pn2_mn40)"
GPU3=$(one_cell 3 partseg pointnext-s shapenetpart 128 "$CK_PNX_SHAPE" pnx_shape)

launch() {  # gpu_idx cmd
  echo "==== GPU $1 ===="
  echo "$2"
  echo ""
  if [ "$DRY" = "1" ]; then return; fi
  setsid bash -c "$2" > "gpu$1.log" 2>&1 < /dev/null &
  echo "  -> launched, log: gpu$1.log"
}

launch 0 "$GPU0"
launch 1 "$GPU1"
launch 2 "$GPU2"
launch 3 "$GPU3"

echo ""
echo "全部起在后台. 监控: tail -f gpu0.log gpu1.log gpu2.log gpu3.log"
echo "GPU 占用: watch -n5 nvidia-smi"
