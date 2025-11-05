# PNN point_config generator (unified)

Use a single entrypoint: `tools/gen_point_config.py`

It instantiates the model from your YAML, observes real intermediate tensor shapes via a dummy forward pass, and emits a simulator-friendly `OrderedDict` describing:
- Per-stage Groupers (S, K, radius, Tree)
- Conv2D/BN/ReLU stacks on grouped features
- MaxPool over K
- Linear head

Input points N0 (num_points)
- Use `--num-points` to set N0 explicitly
- Or `--auto-num-points` to parse it from a full training YAML (dataset/train/val num_points)
- Or `--infer-from-data` to build a DataLoader and peek a batch (requires data_dir to be valid)

Examples
- PointNet++ (ModelNet40, 1024 points):
  ```bash
  python tools/gen_point_config.py \
    --cfg PointNeXt/cfgs/modelnet40ply2048/pointnet++.yaml \
    --num-points 1024 --device cuda \
    --out simulator/PNN_Acc_Base/generated/pointnetpp_from_model.py
  ```
- PointNeXt-S (kdtree):
  ```bash
  python tools/gen_point_config.py \
    --cfg PointNeXt/cfgs/modelnet40ply2048/pointnext-s_kdtree.yaml \
    --num-points 1024 --device cuda \
    --out simulator/PNN_Acc_Base/generated/pointnext_s_from_model.py
  ```
- Auto-detect N0 from a full training YAML (has dataset/dataloader):
  ```bash
  python tools/gen_point_config.py \
    --cfg simulator/PNN_Acc_Base/generated/pointnext-s_kdtree_traincfg.yaml \
    --infer-from-data --device cuda \
    --out simulator/PNN_Acc_Base/generated/pointnext_s_from_model_inferdata.py
  ```

Deprecated scripts
- gen_point_config_from_pointnet2.py
- gen_point_config_from_pointnext.py
- gen_point_config_from_json.py

These now print a deprecation message; use `gen_point_config.py` instead.

## S3DIS example

Segmentation configs also work for stage shape capture, but the classifier head differs by task. You can still generate the stage-wise point_config; the head will follow what the cfg exposes.

- PointNet (S3DIS), set an explicit N0 (adjust to your pipeline, e.g., 4096 or 8192):
  ```bash
  python tools/gen_point_config.py \
    --cfg PointNeXt/cfgs/s3dis/pointnet.yaml \
    --num-points 4096 --device cuda \
    --out simulator/PNN_Acc_Base/generated/s3dis_pointnet_from_model.py
  ```

- If you have a full training YAML that includes dataset/dataloader with a valid `data_dir`, you can infer N0 from a real batch:
  ```bash
  python tools/gen_point_config.py \
    --cfg <your_s3dis_training_yaml_with_dataset> \
    --infer-from-data --device cuda \
    --out simulator/PNN_Acc_Base/generated/s3dis_pointnet_from_model_inferdata.py
  ```

Notes
- Ensure the dataset path in the cfg points to your local S3DIS folder; otherwise `--infer-from-data` cannot fetch a batch.
- If your segmentation model uses different decoders/heads, the generated head will reflect the cfg-provided `cls_args` (if present); otherwise only encoder stages are emitted.
