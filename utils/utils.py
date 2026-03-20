import os
import random
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def now_string():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_group_name(cfg: Dict[str, Any]) -> str:
    """
    A "group run" can contain multiple methods.
    """
    ts = now_string()
    method = cfg["model"]["method"]
    group = cfg['model']['configs'][method]['train_mode']
    dataset = cfg['data']['dataset']
    run_name = f"{dataset}-{method}-{group}-{ts}"
    if "run_name" in cfg.get("project", {}):
        if cfg["project"]["run_name"] is not None and len(cfg["project"]["run_name"]) > 0:
            run_name = f"{dataset}-{method}-{group}-{cfg['project']['run_name']}-{ts}"
    return run_name

def pick_metric_name(metric_for_best_model: str) -> str:
    """
    No alias/cleanup. Use exact metric key produced by evaluator/metrics.
    Example valid keys:
      - "mean"
      - "roc-auc"
      - "f1"
      - "brier"
      - "c@1"
      - "f05u"
    """
    name = (metric_for_best_model or "").strip()
    return name if name else "mean"
