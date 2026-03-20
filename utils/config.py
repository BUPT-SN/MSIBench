# utils/config.py
import argparse
import os
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def parse_args():
    """
    Unified CLI:
      --mode train|test
      train:
        --config
        --run_name (optional)
        --resume_from (optional)
      test:
        --config
        --run_dir (optional)
        --checkpoint (optional)
        --split (optional)
    """
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, choices=["train", "test"], required=True)

    p.add_argument("--config", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None, help="override run name (train)")
    p.add_argument("--resume_from", type=str, default=None, help="resume from checkpoint dir (train)")

    p.add_argument("--run_dir", type=str, default=None, help="run dir created by training (test)")
    p.add_argument("--checkpoint", type=str, default=None, help="direct checkpoint dir (e.g., .../best) (test)")
    p.add_argument("--split", type=str, default=None, help="split name for evaluation (test)")

    return p.parse_args()


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)]

def _normalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)

    proj = cfg.setdefault("project", {})
    out_dir = str(proj.get("output_dir", "outputs"))
    proj["output_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)

    data = cfg.setdefault("data", {})
    if "data_root" not in data:
        raise ValueError("Missing cfg.data.data_root")
    if "dataset" not in data:
        raise ValueError("Missing cfg.data.dataset")

    model = cfg.setdefault("model", {})
    if "method" not in model:
        raise ValueError("Missing cfg.model.method (cross_encoder | bi_encoder)")
    if model["method"] not in {"cross_encoder", "bi_encoder"}:
        raise ValueError(f"Unsupported cfg.model.method={model['method']}")

    model.setdefault("configs", {})
    cfg.setdefault("train", {})
    cfg.setdefault("wandb", {})

    # 删除所有 legacy 字段兼容，不再 pop，不再推断
    return cfg


def resolve_config_for_train(args) -> Dict[str, Any]:
    cfg = load_yaml(args.config)
    cfg = _normalize_cfg(cfg)

    if args.run_name is not None:
        cfg.setdefault("project", {})
        cfg["project"]["run_name"] = args.run_name
    if args.resume_from is not None:
        cfg.setdefault("project", {})
        cfg["project"]["resume_from"] = args.resume_from

    return cfg


def resolve_config_for_test(args) -> Dict[str, Any]:
    cfg = load_yaml(args.config)
    cfg = _normalize_cfg(cfg)
    return cfg
