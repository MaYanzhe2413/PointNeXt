import __init__
import os
import argparse
import yaml
import numpy as np
from torch import multiprocessing as mp
from examples.classification.train import main as train
from examples.classification.pretrain import main as pretrain
from openpoints.utils import (
    EasyConfig,
    dist_utils,
    find_free_port,
    generate_exp_directory,
    resume_exp_directory,
    Wandb,
)
from openpoints.models import build_model_from_cfg as build_model  # util to instantiate network
from torchinfo import summary  # lightweight model summarizer


def print_model(rank: int, cfg: EasyConfig, *args, **kwargs):
    if rank != 0:
        return                                          # 只让 rank-0 打印

    # ---- 0. 把 group_args 转成可点式访问 ----
    from types import SimpleNamespace
    def dict2ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict2ns(v) for k, v in d.items()})
        if isinstance(d, list):
            return [dict2ns(v) for v in d]
        return d
    enc = cfg.model.get('encoder_args', None)
    if enc and 'group_args' in enc:
        enc['group_args'] = EasyConfig(enc['group_args'])

    cpu_model = build_model(cfg.model).cpu().eval()
    print("\n================ Network Architecture ================ ")
    # try:
    #     from torchinfo import summary
    #     summary(cpu_model, input_size=cfg.input_size)
    # except Exception as e:
    #     print("torchinfo.summary failed -> fallback\n", e)
    #     print(cpu_model)
    print("=======================================================\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Point cloud training / utility entry")
    parser.add_argument("--cfg", type=str, required=True, help="config file")
    parser.add_argument(
        "--profile", action="store_true", default=False, help="set to True to profile speed"
    )
    args, opts = parser.parse_known_args()

    # ----------------------------------------------------------------------
    #  Load & merge config
    # ----------------------------------------------------------------------
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # ----------------------------------------------------------------------
    #  Distributed setup (must happen before logger, dataloader, etc.)
    # ----------------------------------------------------------------------
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # ----------------------------------------------------------------------
    #  Experiment folder / logging
    # ----------------------------------------------------------------------
    cfg.task_name = args.cfg.split(".")[-2].split("/")[-2]
    cfg.exp_name = args.cfg.split(".")[-2].split("/")[-1]

    tags = [
        cfg.task_name,
        cfg.mode,
        cfg.exp_name,
        f"ngpus{cfg.world_size}",
        f"seed{cfg.seed}",
    ]

    opt_list = []
    for opt in opts:
        if not any(ignore in opt for ignore in ["rank", "dir", "root", "pretrain", "path", "wandb", "/"]):
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = "-".join(opt_list)

    if cfg.mode in ["resume", "val", "test"]:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    elif cfg.mode != "print":  # 'print' 模式不需要生成exp目录
        generate_exp_directory(cfg, tags, additional_id=os.environ.get("MASTER_PORT", None))
        cfg.wandb.tags = tags

    # Save merged cfg for record (except in 'print' mode)
    if cfg.mode != "print":
        os.environ["JOB_LOG_DIR"] = cfg.log_dir
        cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, indent=2)
            os.system(f"cp {args.cfg} {cfg.run_dir}")
        cfg.cfg_path = cfg_path
        cfg.wandb.name = cfg.run_name

    # ----------------------------------------------------------------------
    #  Select main entry according to mode
    # ----------------------------------------------------------------------
    if cfg.mode == "pretrain":
        main_fn = pretrain
    elif cfg.mode == "print":
        # Simply print the model and exit (single process is enough)
        main_fn = print_model
        cfg.mp = False  # 强制单进程，打印一次即可
    else:
        main_fn = train

    # ----------------------------------------------------------------------
    #  Launch (multi‑processing or single)
    # ----------------------------------------------------------------------
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print("using mp spawn for distributed training")
        mp.spawn(main_fn, nprocs=cfg.world_size, args=(cfg, args.profile))
    else:
        main_fn(0, cfg, profile=args.profile)
