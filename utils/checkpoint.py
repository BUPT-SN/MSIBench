# utils/checkpoint.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch


def save_model_checkpoint(wrapper: Any, output_dir: str, method: str) -> None:
    """
    Save model checkpoint (encoder + tokenizer + wrapper.model weights + hyperparams config)
    into output_dir.

    Directory layout:
      output_dir/
        encoder/                  (HF encoder save_pretrained)
        tokenizer files...        (HF tokenizer save_pretrained)
        <WEIGHTS_NAME>            (torch state_dict)
        <CONFIG_NAME>             (json hyperparams)
    """
    method = str(method).strip()
    os.makedirs(output_dir, exist_ok=True)

    # --- save encoder ---
    enc_dir = os.path.join(output_dir, "encoder")
    os.makedirs(enc_dir, exist_ok=True)

    if method == "cross_encoder":
        # wrapper.model is _CrossEncoderCore with .encoder
        wrapper.model.encoder.save_pretrained(enc_dir)
        weights_name = getattr(wrapper, "WEIGHTS_NAME", "pytorch_model.bin")
        config_name = getattr(wrapper, "CONFIG_NAME", "saved_config.json")
        cfg_obj = getattr(wrapper, "_saved_cfg", None)
    elif method == "bi_encoder":
        # wrapper.model is _BiEncoderCore with .encoder
        wrapper.model.encoder.save_pretrained(enc_dir)
        weights_name = getattr(wrapper, "WEIGHTS_NAME", "pytorch_model.bin")
        config_name = getattr(wrapper, "CONFIG_NAME", "tstd_biencoder_config.json")
        cfg_obj = getattr(wrapper, "_saved_cfg", None)
    else:
        raise ValueError(f"Unsupported method={method} for save_model_checkpoint.")

    # --- save tokenizer ---
    if not hasattr(wrapper, "tokenizer") or wrapper.tokenizer is None:
        raise AttributeError("wrapper.tokenizer is required for checkpoint saving.")
    wrapper.tokenizer.save_pretrained(output_dir)

    # --- save weights ---
    if not hasattr(wrapper, "model") or wrapper.model is None:
        raise AttributeError("wrapper.model is required for checkpoint saving.")
    torch.save(wrapper.model.state_dict(), os.path.join(output_dir, weights_name))

    # --- save hyperparams config ---
    if cfg_obj is None or not hasattr(cfg_obj, "to_dict"):
        raise AttributeError("wrapper._saved_cfg (with .to_dict()) is required for checkpoint saving.")
    with open(os.path.join(output_dir, config_name), "w", encoding="utf-8") as f:
        json.dump(cfg_obj.to_dict(), f, indent=2, ensure_ascii=False)


def load_model_checkpoint(
    cfg: Dict[str, Any],
    checkpoint_dir: str,
    device: Optional[str] = None,
) -> Any:
    """
    Load wrapper from checkpoint_dir (saved by save_model_checkpoint).

    Note:
      - checkpoint_dir is the model dir (e.g. .../best or .../checkpoint-epochX/model)
      - method is inferred from cfg.model.method (single source of truth)
    """
    model_cfg = cfg.get("model", {}) or {}
    method = str(model_cfg.get("method", "cross_encoder")).strip()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    enc_dir = os.path.join(checkpoint_dir, "encoder")
    if not os.path.isdir(enc_dir):
        raise FileNotFoundError(f"Missing encoder dir in checkpoint: {enc_dir}")

    if method == "cross_encoder":
        from models.cross_encoder import CrossEncoder, CrossEncoderSavedConfig

        config_name = getattr(CrossEncoder, "CONFIG_NAME", "saved_config.json")
        weights_name = getattr(CrossEncoder, "WEIGHTS_NAME", "pytorch_model.bin")
        cfg_path = os.path.join(checkpoint_dir, config_name)
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Missing {config_name} in checkpoint_dir={checkpoint_dir}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            saved = CrossEncoderSavedConfig.from_dict(json.load(f))

        wrapper = CrossEncoder(
            pretrained_name=checkpoint_dir,  # load tokenizer from checkpoint_dir
            encoder_name_or_path=enc_dir,    # load encoder from encoder/
            max_length=saved.max_length,
            dropout=saved.dropout,
            cls_weight=saved.cls_weight,
            rank_weight=saved.rank_weight,
            mismatch_rank_weight=saved.mismatch_rank_weight,
            listwise_temperature=saved.listwise_temperature,
        )

    elif method == "bi_encoder":
        from models.bi_encoder import BiEncoder, BiEncoderSavedConfig

        config_name = getattr(BiEncoder, "CONFIG_NAME", "tstd_biencoder_config.json")
        weights_name = getattr(BiEncoder, "WEIGHTS_NAME", "pytorch_model.bin")
        cfg_path = os.path.join(checkpoint_dir, config_name)
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Missing {config_name} in checkpoint_dir={checkpoint_dir}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            saved = BiEncoderSavedConfig.from_dict(json.load(f))

        wrapper = BiEncoder(
            pretrained_name=checkpoint_dir,  # load tokenizer from checkpoint_dir
            encoder_name_or_path=enc_dir,    # load encoder from encoder/
            max_length=saved.max_length,
            pooling=saved.pooling,
            proj_dim=saved.proj_dim,
            temperature=saved.temperature,
            learnable_temperature=saved.learnable_temperature,
            normalize=saved.normalize,
            cls_weight=saved.cls_weight,
            rank_weight=saved.rank_weight,
            mismatch_rank_weight=saved.mismatch_rank_weight,
            listwise_temperature=saved.listwise_temperature,
        )

    else:
        raise ValueError(f"Unsupported cfg.model.method={method}")

    weights_path = os.path.join(checkpoint_dir, weights_name)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing weights file in checkpoint: {weights_path}")

    state = torch.load(weights_path, map_location="cpu")
    wrapper.model.load_state_dict(state)

    wrapper.to(device)
    return wrapper
