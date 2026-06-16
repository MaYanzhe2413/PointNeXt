#!/bin/bash
# ============================================================================
# 批量执行清单: 剩余的 coord ladder + INT8 PTQ
# 矩阵 = {pointnext-s, pointnet++} x {s3dis, modelnet40, shapenet}
#
# 用法:
#   1. 填好下面 CKPT_* 变量 (已知的预填, 缺的留空会自动跳过)
#   2. 如数据路径非默认, 填 DR_* (data_root override)
#   3. bash run_all_remaining.sh            # 跑全部
#      bash run_all_remaining.sh ladder     # 只跑 coord ladder
#      bash run_all_remaining.sh ptq        # 只跑 PTQ
#      在命令前加 DRY=1 只打印不执行: DRY=1 bash run_all_remaining.sh
#
# 找 checkpoint 路径的助手 (先跑这个确认):
#   for p in s3dis modelnet40ply2048 shapenetpart; do
#     echo "== $p =="; ls -d log/$p/*pointnext-s-*  log/$p/*pointnet++-* 2>/dev/null | grep -vE "_kdtree|_random|_simple|coord_quant|abl_|_baseline"
#   done
# ============================================================================

GPU="${GPU:-0}"
SEED=7895
DRY="${DRY:-0}"

# ---------------- checkpoint 目录 (填 run 目录名, 不含 /checkpoint/...) ----------------
# PointNeXt-S (全局 FPS 训练)
CKPT_PNX_S3DIS="log/s3dis/s3dis-train-pointnext-s-ngpus3-20260612-102404-GmdxU7SJa3BUuadeLTdQgY"
CKPT_PNX_MN40="log/modelnet40ply2048/modelnet40ply2048-train-pointnext-s-ngpus4-seed7895-20260331-221531-JvuawAeqYqbwPKDYod2R2f"
CKPT_PNX_SHAPE="log/shapenetpart/shapenetpart-train-pointnext-s-ngpus2-seed7895-20260323-215423-nGmdDiEbvB4encTtDjDuT4"
# PointNet++ (全局 FPS 训练)
CKPT_PN2_S3DIS="log/s3dis/s3dis-train-pointnet++-ngpus3-20260615-103940-kKwLcsaKbXWhfZuMKrxcNt"
CKPT_PN2_MN40="log/modelnet40ply2048/modelnet40ply2048-train-pointnet++-ngpus4-seed527-20250914-155417-bebhZhPnyPhBmLNisKNyQG"
CKPT_PN2_SHAPE="log/shapenetpart/shapenetpart-train-pointnet++-ngpus2-seed7895-20260615-220231-6DEjnjKUveG6UEHbGkyF4z"

# ---------------- data_root override (默认空=用 config 默认; 换机器时填) ----------------
DR_S3DIS=""       # e.g. /workspace/dataset/s3disfull
DR_MN40=""        # e.g. /workspace/dataset/modelnet40_ply_hdf5_2048  (注意 modelnet 用 data_dir)
DR_SHAPE=""       # e.g. /home/mayz/shapenetcore_..._normal

# ---------------- 每数据集的 task / cfg 名 / block-FPS leaf ----------------
# 行: 标签|task|dataset目录|leaf|ckpt变量|data_root变量|dr_key
ROWS=(
  "pnx_s3dis|seg|s3dis|pointnext-s|1500|$CKPT_PNX_S3DIS|$DR_S3DIS|dataset.common.data_root"
  "pnx_mn40|cls|modelnet40ply2048|pointnext-s|64|$CKPT_PNX_MN40|$DR_MN40|dataset.common.data_dir"
  "pnx_shape|partseg|shapenetpart|pointnext-s|128|$CKPT_PNX_SHAPE|$DR_SHAPE|dataset.common.data_root"
  "pn2_s3dis|seg|s3dis|pointnet++|1500|$CKPT_PN2_S3DIS|$DR_S3DIS|dataset.common.data_root"
  "pn2_mn40|cls|modelnet40ply2048|pointnet++|64|$CKPT_PN2_MN40|$DR_MN40|dataset.common.data_dir"
  "pn2_shape|partseg|shapenetpart|pointnet++|128|$CKPT_PN2_SHAPE|$DR_SHAPE|dataset.common.data_root"
)

MODE="${1:-all}"   # all | ladder | ptq

run() {
  echo ">>> $1"
  if [ "$DRY" = "1" ]; then return 0; fi
  eval "$1"
}

ckpt_pth() {  # 从 run 目录名拼出 ckpt_best.pth
  local d="$1"; echo "$d/checkpoint/$(basename $d)_ckpt_best.pth"
}

for ROW in "${ROWS[@]}"; do
  IFS='|' read -r TAG TASK DS MODEL LEAF CKPTDIR DR DRKEY <<< "$ROW"
  echo ""
  echo "============================================================"
  echo " $TAG  (task=$TASK, model=$MODEL, dataset=$DS, leaf=$LEAF)"
  echo "============================================================"
  if [ -z "$CKPTDIR" ]; then
    echo "  [跳过] 无 checkpoint (CKPT 变量为空)"
    continue
  fi
  CKPT=$(ckpt_pth "$CKPTDIR")
  DROPT=""; [ -n "$DR" ] && DROPT="$DRKEY=$DR"

  # ---------- coord ladder ----------
  if [ "$MODE" = "all" ] || [ "$MODE" = "ladder" ]; then
    echo "--- coord ladder (A0->A3-safe + leaf sweep) ---"
    LCMD="bash batch_ablation_general.sh --task $TASK --model $MODEL --dataset $DS -g $GPU -S $SEED -p $CKPT --sweep"
    [ -n "$DROPT" ] && LCMD="$LCMD --extra \"$DROPT\""
    run "$LCMD"
  fi

  # ---------- INT8 PTQ (叠在 block FPS 上) ----------
  if [ "$MODE" = "all" ] || [ "$MODE" = "ptq" ]; then
    echo "--- INT8 PTQ (stacked on block FPS leaf=$LEAF) ---"
    KD="model.encoder_args.sampler=kdtree model.encoder_args.sampler_args.leaf_size=$LEAF model.encoder_args.sampler_args.strategy=fps"
    run "CUDA_VISIBLE_DEVICES=$GPU python quant/ptq_general.py --task $TASK --cfg cfgs/$DS/$MODEL.yaml --ckpt $CKPT --tag ${TAG}_blockfps $KD $DROPT"
  fi
done

echo ""
echo "============================================================"
echo " 全部完成. 取结果:"
echo "   coord ladder:  grep -hE 'Best ckpt|E@.*OA:|Instance mIoU' experiment_logs/abl_*.log"
echo "   PTQ:           cat quant/output/ptq_*.json | python -m json.tool 2>/dev/null; ls quant/output/ptq_*.json"
echo "============================================================"
