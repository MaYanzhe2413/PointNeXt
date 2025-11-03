import __init__
import os
import argparse
import yaml
import numpy as np
from torch import multiprocessing as mp
from types import SimpleNamespace
from examples.classification.train import main as train
from examples.classification.pretrain import main as pretrain
from openpoints.utils import (
    EasyConfig,
    dist_utils,
    find_free_port,
    generate_exp_directory,
    resume_exp_directory,
    Wandb,
    load_checkpoint,
)
from openpoints.models import build_model_from_cfg  # 官方工具：根据 cfg 构造网络

# --- 兼容 torchinfo 不存在的环境 ---------------------------------------------
try:
    from torchinfo import summary  # 美观的网络层级表
except ImportError:  # 没装 torchinfo 也能跑，只是少了漂亮的表格
    summary = None

# -----------------------------------------------------------------------------
# 辅助：把 dict 递归变成 SimpleNamespace (支持 a.b 点式访问且仍是 Mapping)
# -----------------------------------------------------------------------------
from collections.abc import Mapping

def dict_to_ns(d):
    """递归将嵌套 dict ➜ SimpleNamespace，其他类型保持不变"""
    if isinstance(d, Mapping):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_ns(v) for v in d]
    else:
        return d

# -----------------------------------------------------------------------------
# 打印模型结构（新增：支持 torchinfo & cfg.input_size；并解决 group_args 属性报错）
# -----------------------------------------------------------------------------

def print_model(rank: int, cfg: EasyConfig, *args, **kwargs):
    """Instantiate the model,打印网络结构并输出每个局部邻域(QueryAndGroup/KNNGroup)的尺寸信息。
    只在 rank 0 执行，支持 cfg.input_size=(B,C,N)。
    """
    if rank != 0:
        return

    # ---- 1. 确保 group_args 支持点式访问 ----
    enc_cfg = cfg.model.get('encoder_args', None)
    if enc_cfg and 'group_args' in enc_cfg:
        enc_cfg['group_args'] = dict_to_ns(enc_cfg['group_args'])

    # ---- 2. 构造模型 ----
    model = build_model_from_cfg(cfg.model).cuda().eval()

    # ---- 3. 打印网络结构 ----
    print("
=================== Model Architecture ===================")
    if summary and hasattr(cfg, 'input_size'):
        try:
            summary(model, input_size=cfg.input_size)
        except Exception as e:
            print('[torchinfo.summary failed]:', e)
            print(model)
    else:
        print(model)
    print("========================================================")

    # ---- 4. 注册邻域 hook 收集尺寸 ----
    from openpoints.models.layers.group import QueryAndGroup, KNNGroup, DilatedKNN
    import collections, torch
    log = collections.OrderedDict()

    def hook(name, module, inputs, output):
        idx = output if isinstance(output, torch.Tensor) else output[0]
        _, M, K = idx.shape  # (B, M, K)
        Cin = inputs[2].shape[1] if inputs[2] is not None else '-'
        log[name] = dict(M=M, K=K, Cin=Cin)

    for n, m in model.named_modules():
        if isinstance(m, (QueryAndGroup, KNNGroup, DilatedKNN)):
            m.register_forward_hook(partial(hook, n))

    # ---- 5. 单次 dummy 推理触发 hook ----
    if hasattr(cfg, 'input_size'):
        B, C, N = cfg.input_size  # e.g. (1,3,2048)
    else:
        B, C, N = 1, 3, 1024
    dummy = torch.randn(B, C, N).cuda()
    try:
        _ = model(dummy)
    except Exception as e:
        print('
[Forward failed on dummy data]:', e)
        print('无法统计邻域信息，仅打印网络结构
')
        return

    # ---- 6. 打印邻域统计 ----
    if log:
        print("
=================== 邻域尺寸统计 ===================")
        for n, v in log.items():
            print(f"{n:<55} | Cin={v['Cin']:^4} → M={v['M']:^5}, K={v['K']}")
        print("========================================================
")

# -----------------------------------------------------------------------------
#  主入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Point cloud training / utility entry")
    parser.add_argument("--cfg", required=True, help="config yaml")
    parser.add_argument("--profile", action="store_true", help="profile speed")
    args, opts = parser.parse_known_args()

    # 1. 载入 cfg
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # 2. Dist 信息
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # 3. 目录 & Wandb（print 模式跳过磁盘开销）
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]

    tags = [cfg.task_name, cfg.mode, cfg.exp_name, f'ngpus{cfg.world_size}', f'seed{cfg.seed}']

    if cfg.mode in ["resume", "val", "test"]:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    elif cfg.mode != "print":
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT'))
        cfg.wandb.tags = tags

    # 保存 cfg
    if cfg.mode != 'print':
        os.environ['JOB_LOG_DIR'] = cfg.log_dir
        cfg_path = os.path.join(cfg.run_dir, 'cfg.yaml')
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, indent=2)
            os.system(f'cp {args.cfg} {cfg.run_dir}')
        cfg.cfg_path = cfg_path
        cfg.wandb.name = cfg.run_name

    # 4. 选择主函数
    if cfg.mode == 'pretrain':
        main_fn = pretrain
    elif cfg.mode == 'print':
        main_fn = print_model
        cfg.mp = False                # 单进程即可
    else:
        main_fn = train

    # 5. 启动
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f'tcp://localhost:{port}'
        print('using mp spawn for distributed training')
        mp.spawn(main_fn, nprocs=cfg.world_size, args=(cfg, args.profile))
    else:
        main_fn(0, cfg, profile=args.profile)
