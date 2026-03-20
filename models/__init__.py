# models/__init__.py
from __future__ import annotations

from typing import Any, Dict

from models.bi_encoder import BiEncoder
from models.cross_encoder import CrossEncoder


def get_train_mode(cfg: Dict[str, Any]) -> str:
    """
    Decide training mode from:
      cfg.model.method
      cfg.model.configs.<method>.train_mode

    Returns: "group" or "pair"
    """
    model_cfg = cfg.get("model", {}) or {}
    method = str(model_cfg.get("method", "cross_encoder")).strip()
    configs = model_cfg.get("configs", {}) or {}

    if method == "cross_encoder":
        c = configs.get("cross_encoder", {}) or {}
    elif method == "bi_encoder":
        c = configs.get("bi_encoder", {}) or {}
    else:
        raise ValueError(f"Unsupported model.method={method}")

    tm = str(c.get("train_mode", "pair")).strip().lower()
    if tm not in {"group", "pair"}:
        raise ValueError(f"Invalid train_mode={tm}. Use 'group' or 'pair'.")
    return tm


def build_model(cfg: Dict[str, Any]):
    """
    Build model wrapper from config.

    Required:
      cfg['model']['method'] in {'cross_encoder','bi_encoder'}
      cfg['model']['configs'][<method>] exists
    """
    model_cfg = cfg.get("model", {}) or {}
    method = str(model_cfg.get("method", "")).strip()
    all_cfgs = model_cfg.get("configs", {}) or {}

    if method == "cross_encoder":
        c = all_cfgs.get("cross_encoder", {}) or {}
        return CrossEncoder(
            pretrained_name=str(c.get("pretrained_name", "microsoft/deberta-v3-base")),
            max_length=int(c.get("max_length", 512)),
            dropout=float(c.get("dropout", 0.1)),
            cls_weight=float(c.get("cls_weight", 1.0)),
            rank_weight=float(c.get("rank_weight", 0.0)),
            mismatch_rank_weight=float(c.get("mismatch_rank_weight", 1.0)),
            listwise_temperature=float(c.get("listwise_temperature", 1.0)),
        )

    if method == "bi_encoder":
        c = all_cfgs.get("bi_encoder", {}) or {}
        return BiEncoder(
            pretrained_name=str(c.get("pretrained_name", "BAAI/bge-large-en-v1.5")),
            max_length=int(c.get("max_length", 256)),
            pooling=str(c.get("pooling", "mean")),
            proj_dim=int(c.get("proj_dim", 0)),
            temperature=float(c.get("temperature", 0.07)),
            learnable_temperature=bool(c.get("learnable_temperature", False)),
            normalize=bool(c.get("normalize", True)),

            cls_weight=float(c.get("cls_weight", 1.0)),
            rank_weight=float(c.get("rank_weight", 0.0)),
            mismatch_rank_weight=float(c.get("mismatch_rank_weight", 0.0)),
            listwise_temperature=float(c.get("listwise_temperature", 1.0)),

            # -------------------------
            # In-batch InfoNCE (主分支 NEG route; 非 mm 分支)
            # -------------------------
            use_inbatch_nce=bool(c.get("use_inbatch_nce", False)),
            nce_weight=float(c.get("nce_weight", 0.1)),
            nce_symmetric=bool(c.get("nce_symmetric", True)),
        )

    raise ValueError(f"Unsupported model.method={method}. Use 'cross_encoder' or 'bi_encoder'.")
