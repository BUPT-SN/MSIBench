import os
from typing import Any, Dict, Optional

import wandb

from utils.utils import is_main_process


def init_wandb(cfg: Dict[str, Any], run_name: str, output_dir: str):
    wb_cfg = cfg.get("wandb", {})
    enabled = bool(wb_cfg.get("enabled", True))

    if not enabled:
        os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "disabled")
        return None

    if not is_main_process():
        return None

    project = wb_cfg.get("project") or os.environ.get("WANDB_PROJECT")
    entity = wb_cfg.get("entity") or os.environ.get("WANDB_ENTITY")
    tags = wb_cfg.get("tags", [])

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,
        config=cfg,
        dir=output_dir,
    )
    return run


def finish_wandb():
    if is_main_process() and wandb.run is not None:
        wandb.finish()
