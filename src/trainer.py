# src/trainer.py
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from models import build_model, get_train_mode
from utils.dataloader import (
    load_group_dataset,
    load_split_dataset,
    make_group_collator,
    make_pair_collator,
)
from utils.metrics import evaluate_all, find_best_threshold_grid_clipped
from utils.utils import set_seed, make_run_group_name
from utils.wandb_logger import init_wandb, finish_wandb
from utils.config import save_yaml


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def resolve_train_mode(cfg: Dict[str, Any]) -> str:
    model_cfg = cfg.get("model", {}) or {}
    method = str(model_cfg.get("method", "")).strip()
    configs = model_cfg.get("configs", {}) or {}
    c = configs.get(method, {}) or {}

    tm = str(c.get("train_mode", "pair")).strip().lower()
    if tm not in {"pair", "group"}:
        raise ValueError(f"Invalid train_mode={tm}. Use 'pair' or 'group'.")
    return tm


@dataclass
class _BestTracker:
    metric_name: str
    greater_is_better: bool
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None

    def is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.greater_is_better:
            return value > self.best_value
        return value < self.best_value

    def update(self, value: float, epoch: int) -> None:
        self.best_value = float(value)
        self.best_epoch = int(epoch)


def _save_checkpoint(
    run_dir: str,
    epoch: int,
    wrapper,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    method: str,
) -> str:
    ckpt_dir = os.path.join(run_dir, f"checkpoint-epoch{epoch}")
    _ensure_dir(ckpt_dir)

    model_dir = os.path.join(ckpt_dir, "model")
    _ensure_dir(model_dir)

    from utils.checkpoint import save_model_checkpoint
    save_model_checkpoint(wrapper, model_dir, method=method)

    state = {
        "epoch": int(epoch),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
    return ckpt_dir


def _rotate_checkpoints(run_dir: str, save_total_limit: int) -> None:
    if save_total_limit <= 0:
        return
    entries = []
    for name in os.listdir(run_dir):
        if name.startswith("checkpoint-epoch"):
            full = os.path.join(run_dir, name)
            if os.path.isdir(full):
                try:
                    ep = int(name.replace("checkpoint-epoch", ""))
                except Exception:
                    ep = 0
                entries.append((ep, full))
    entries.sort(key=lambda x: x[0])
    while len(entries) > save_total_limit:
        _, path = entries.pop(0)
        shutil.rmtree(path)


def _load_resume(
    resume_from: str,
    cfg: Dict[str, Any],
    device: str,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    model_dir = resume_from
    if os.path.isdir(os.path.join(resume_from, "model")):
        model_dir = os.path.join(resume_from, "model")

    from utils.checkpoint import load_model_checkpoint
    wrapper = load_model_checkpoint(cfg, model_dir, device=device)

    st_path = os.path.join(resume_from, "trainer_state.pt")
    if os.path.isfile(st_path):
        state = torch.load(st_path, map_location="cpu")
        return wrapper, state
    return wrapper, None


def _logits_to_probs_np(logits: torch.Tensor) -> np.ndarray:
    """
    Convert model outputs to probabilities for binary tasks.

    Supports:
      - logits shape [N] or [N, 1]  (BCE-style)
      - logits shape [N, 2]         (CE-style): score = logit_pos - logit_neg
    """
    x = logits.detach().float().cpu().numpy()
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim == 2 and x.shape[1] == 2:
        score = x[:, 1] - x[:, 0]
        return _sigmoid_np(score.astype(np.float32))
    if x.ndim == 1:
        return _sigmoid_np(x.astype(np.float32))
    raise ValueError(f"Unsupported logits shape for binary probs: {x.shape}")


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


@torch.no_grad()
def _compute_group_diagnostics(
    out: Dict[str, Any],
    *,
    listwise_temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Diagnostics for group-mode training.

    Expects model forward to return:
      - neg_scores: [B, K1]  (index 0 is POS, others NEG)
      - mm_scores:  [B, K2]  (index 0 is POS, others mismatch NEG)
    """

    def _diag_from_scores(scores_t: torch.Tensor, prefix: str) -> Dict[str, float]:
        scores = scores_t.detach().float().cpu().numpy()
        B, K = scores.shape
        if K < 2:
            return {
                f"{prefix}/gap_mean": 0.0,
                f"{prefix}/gap_p50": 0.0,
                f"{prefix}/gap_p90": 0.0,
                f"{prefix}/gap_gt0": 0.0,
                f"{prefix}/gap_gt1": 0.0,
                f"{prefix}/gap_gt2": 0.0,
                f"{prefix}/posprob_mean": 1.0,
                f"{prefix}/posprob_p50": 1.0,
                f"{prefix}/posprob_p90": 1.0,
                f"{prefix}/posprob_gt099": 1.0,
            }

        pos = scores[:, 0]
        max_neg = np.max(scores[:, 1:], axis=1)
        gap = pos - max_neg  # [B]

        T = float(max(listwise_temperature, 1e-6))
        probs = _softmax_np(scores / T, axis=1)
        pos_prob = probs[:, 0]

        def _pct(a: np.ndarray, q: float) -> float:
            return float(np.percentile(a, q))

        return {
            f"{prefix}/gap_mean": float(np.mean(gap)),
            f"{prefix}/gap_p50": _pct(gap, 50),
            f"{prefix}/gap_p90": _pct(gap, 90),
            f"{prefix}/gap_gt0": float(np.mean(gap > 0.0)),
            f"{prefix}/gap_gt1": float(np.mean(gap > 1.0)),
            f"{prefix}/gap_gt2": float(np.mean(gap > 2.0)),
            f"{prefix}/posprob_mean": float(np.mean(pos_prob)),
            f"{prefix}/posprob_p50": _pct(pos_prob, 50),
            f"{prefix}/posprob_p90": _pct(pos_prob, 90),
            f"{prefix}/posprob_gt099": float(np.mean(pos_prob > 0.99)),
        }

    diags: Dict[str, float] = {}

    if "neg_scores" in out and out["neg_scores"] is not None:
        diags.update(_diag_from_scores(out["neg_scores"], "neg"))
    if "mm_scores" in out and out["mm_scores"] is not None:
        diags.update(_diag_from_scores(out["mm_scores"], "mm"))

    return diags




@torch.no_grad()
def _eval_pairwise(
    wrapper,
    loader: DataLoader,
    device: str,
    desc: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    wrapper.eval()

    all_probs: List[float] = []
    all_labels: List[int] = []
    all_types: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    loss_sum = 0.0
    loss_n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        labels = batch.get("labels")
        if labels is None:
            raise ValueError("Eval batch missing 'labels'.")

        # meta fields (if present)
        sample_uid = batch.get("sample_uid", None)
        neg_type = batch.get("neg_type", None)
        group_id = batch.get("group_id", None)
        s_uid = batch.get("style_ref_text_uid", None)
        q_uid = batch.get("query_text_uid", None)

        # both names supported
        is_mismatch_s = batch.get("is_mismatch_s", None)
        if is_mismatch_s is None:
            is_mismatch_s = batch.get("is_mismatch", None)

        batch = _to_device(batch, device)

        # IMPORTANT: exclude meta keys from model feed
        meta_keys = {
            "sample_uid",
            "split",
            "neg_type",
            "labels",
            "group_id",
            "style_ref_text_uid",
            "query_text_uid",
            "style_uid",
            "query_uid",
            "is_mismatch_s",  # keep is_mismatch for model if exists
        }
        feed = {k: v for k, v in batch.items() if k not in meta_keys}

        out = wrapper.forward(**feed, labels=labels.to(device=device))
        logits = out["logits"]

        if "loss" in out and out["loss"] is not None:
            loss_sum += float(out["loss"].detach().float().cpu().item())
            loss_n += 1

        probs_np = _logits_to_probs_np(logits)
        labels_np = labels.detach().float().cpu().numpy().reshape(-1)

        if probs_np.shape[0] != labels_np.shape[0]:
            raise ValueError(f"Eval probs/labels length mismatch: probs={probs_np.shape} labels={labels_np.shape}")

        all_probs.extend([float(x) for x in probs_np.tolist()])
        all_labels.extend([int(x) for x in labels_np.tolist()])

        # neg_type bookkeeping
        if neg_type is not None:
            if isinstance(neg_type, (list, tuple)):
                cur_types = [str(x) for x in neg_type]
            else:
                cur_types = [str(neg_type) for _ in range(int(labels_np.shape[0]))]
        else:
            cur_types = ["__UNKNOWN__" for _ in range(int(labels_np.shape[0]))]
        all_types.extend(cur_types)

        bsz = int(labels_np.shape[0])

        def _get_list(x, default=None):
            if x is None:
                return [default] * bsz
            if isinstance(x, (list, tuple)):
                return list(x)
            if torch.is_tensor(x):
                return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
            return [x] * bsz

        sample_uid_l = _get_list(sample_uid, default=None)
        group_id_l = _get_list(group_id, default=None)
        s_uid_l = _get_list(s_uid, default=None)
        q_uid_l = _get_list(q_uid, default=None)
        is_mm_l = _get_list(is_mismatch_s, default=0)

        for i in range(bsz):
            all_meta.append(
                {
                    "sample_uid": sample_uid_l[i],
                    "neg_type": cur_types[i],
                    "group_id": group_id_l[i],
                    "style_ref_text_uid": s_uid_l[i],
                    "query_text_uid": q_uid_l[i],
                    "is_mismatch_s": int(is_mm_l[i]) if is_mm_l[i] is not None else 0,
                }
            )

        if loss_n > 0:
            pbar.set_postfix({"loss": loss_sum / max(loss_n, 1)})

    probs = np.asarray(all_probs, dtype=np.float32)
    labs = np.asarray(all_labels, dtype=np.float32)
    types_arr = np.asarray(all_types, dtype=object)

    # ---- 原始二分类 metrics：保持不变 ----
    metrics = evaluate_all(labs, probs, threshold=float(threshold))

    from utils.metrics import evaluate_all_extended, evaluate_ranking_attribution

    ext = evaluate_all_extended(
        labs,
        probs,
        neg_types=[str(x) for x in types_arr.tolist()],
        threshold=float(threshold),
        undecided_eps=0.0,
        ece_bins=15,
    )

    preds = (probs >= float(threshold)).astype(np.int32)
    labs_i = labs.astype(np.int32)

    by_type: Dict[str, Any] = {}
    uniq = sorted(set([str(x) for x in types_arr.tolist()]))

    for t in uniq:
        mask = (types_arr == t)
        n = int(np.sum(mask))
        if n <= 0:
            continue
        y = labs_i[mask]
        p = preds[mask]

        acc = float(np.mean((y == p).astype(np.float32)))

        tp = int(np.sum((y == 1) & (p == 1)))
        tn = int(np.sum((y == 0) & (p == 0)))
        fp = int(np.sum((y == 0) & (p == 1)))
        fn = int(np.sum((y == 1) & (p == 0)))

        by_type[str(t)] = {
            "n": n,
            "acc": acc,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "pos_rate": float(np.mean((y == 1).astype(np.float32))),
            "pred_pos_rate": float(np.mean((p == 1).astype(np.float32))),
        }

    task2 = evaluate_ranking_attribution(
        labs_i,
        probs,
        all_meta,
        threshold=float(threshold),
        k_recall=3,
    )

    return {
        "loss": (loss_sum / loss_n) if loss_n > 0 else None,
        "metrics": metrics,
        "extra_metrics": ext["extra"],
        "n_samples": int(len(all_labels)),
        "probs": probs,
        "labels": labs,
        "by_neg_type": by_type,
        "task2_metrics": task2,
        "meta": all_meta,
    }
def train_and_eval(cfg: Dict[str, Any]) -> Dict[str, Any]:
    project_cfg = cfg.get("project", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    seed = int(project_cfg.get("seed", 42))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = make_run_group_name(cfg)
    output_root = str(project_cfg.get("output_dir", "outputs"))
    run_dir = os.path.join(output_root, run_name)
    _ensure_dir(run_dir)

    save_yaml(cfg, os.path.join(run_dir, "config_used.yaml"))

    wb_run = init_wandb(cfg, run_name=run_name, output_dir=run_dir)

    resume_from = project_cfg.get("resume_from")
    if resume_from:
        wrapper, resume_state = _load_resume(str(resume_from), cfg, device=device)
    else:
        wrapper = build_model(cfg).to(device)
        resume_state = None

    train_mode = get_train_mode(cfg)
    method = str(model_cfg.get("method", "cross_encoder")).strip()
    max_len = int(getattr(wrapper, "max_length", 512))

    train_split = str(data_cfg.get("train_split", "train"))
    dev_split = str(data_cfg.get("dev_split", "dev"))
    test_split = str(data_cfg.get("test_split", "test"))

    train_bs = int(train_cfg.get("train_batch_size", 8))
    eval_bs = int(train_cfg.get("eval_batch_size", 32))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    epochs = int(train_cfg.get("epochs", 1))

    fp16 = bool(train_cfg.get("fp16", False)) and (device == "cuda")
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    lr = float(train_cfg.get("lr", 2e-5))
    wd = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.06))

    metric_for_best = str(train_cfg.get("metric_for_best_model", "mean"))
    greater_is_better = bool(train_cfg.get("greater_is_better", True))
    early_patience = int(train_cfg.get("early_stopping_patience", 5))

    save_total_limit = int(train_cfg.get("save_total_limit", 2))
    auto_test_best = bool(train_cfg.get("auto_test_best", True))

    threshold_metric = str(train_cfg.get("threshold_metric", "f1")).strip().lower()
    if threshold_metric not in {"f1", "f05u", "c@1"}:
        raise ValueError("train.threshold_metric must be one of: f1, f05u, c@1")

    # --------- NEW: hard_mining config (epoch gating) ----------
    hm_cfg = ((cfg.get("data", {}) or {}).get("hard_mining", {}) or {})
    hm_start_epoch = int(hm_cfg.get("start_epoch", 1))
    hm_train_level = str(hm_cfg.get("train_level", "easy")).strip().lower()

    # --------- build dev/test loaders once (eval_level is handled inside dataloader) ----------
    dev_ds = load_split_dataset(cfg, dev_split)
    dev_collate = make_pair_collator(method, wrapper.tokenizer, max_len, cfg=cfg)
    dev_loader = DataLoader(dev_ds, batch_size=eval_bs, shuffle=False, collate_fn=dev_collate)

    test_ds = load_split_dataset(cfg, test_split)
    test_collate = make_pair_collator(method, wrapper.tokenizer, max_len, cfg=cfg)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, collate_fn=test_collate)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {
            "params": [p for n, p in wrapper.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": wd,
        },
        {
            "params": [p for n, p in wrapper.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)

    # NOTE: total steps depends on train loader length; but train loader changes by epoch (hard gating).
    # To keep minimal changes & stable behavior, we estimate using "hard-enabled" length if possible.
    # This keeps scheduler roughly consistent without invasive refactor.
    try:
        if train_mode == "group":
            est_train_ds = load_group_dataset(cfg, train_split, epoch=max(hm_start_epoch, 1))
        else:
            est_train_ds = load_split_dataset(cfg, train_split, epoch=max(hm_start_epoch, 1))
        est_len = max(1, int(np.ceil(len(est_train_ds) / max(train_bs, 1))))
    except Exception:
        est_len = 1

    num_update_steps_per_epoch = int(np.ceil(est_len / max(grad_accum, 1)))
    total_steps = int(epochs * num_update_steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    start_epoch = 1
    global_step = 0

    if resume_state is not None:
        if resume_state.get("optimizer") is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state.get("scheduler") is not None and scheduler is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
        if resume_state.get("scaler") is not None and scaler is not None:
            scaler.load_state_dict(resume_state["scaler"])
        if resume_state.get("epoch") is not None:
            start_epoch = int(resume_state["epoch"]) + 1

    best = _BestTracker(metric_name=metric_for_best, greater_is_better=greater_is_better)
    bad_epochs = 0

    best_json_path = os.path.join(run_dir, "best_metrics.json")

    best_threshold = 0.5
    if os.path.isfile(best_json_path):
        try:
            with open(best_json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if metric_for_best in obj:
                best.best_value = float(obj[metric_for_best])
                best.best_epoch = int(obj.get("epoch", 0))
            if "threshold" in obj:
                best_threshold = float(obj["threshold"])
        except Exception:
            pass

    logging_steps = int(train_cfg.get("logging_steps", 20))
    listwise_T = float(getattr(wrapper, "listwise_temperature", 1.0))

    for epoch in range(start_epoch, epochs + 1):
        # --------- NEW: rebuild train loader each epoch (for hard epoch gating & group random fill) ----------
        hard_enabled_this_epoch = (epoch >= hm_start_epoch)

        if train_mode == "group":
            train_ds = load_group_dataset(cfg, train_split, epoch=epoch, hard_level_override=hm_train_level)
            train_collate = make_group_collator(method, wrapper.tokenizer, max_len, cfg=cfg)
        else:
            train_ds = load_split_dataset(cfg, train_split, epoch=epoch, hard_level_override=hm_train_level)
            train_collate = make_pair_collator(method, wrapper.tokenizer, max_len, cfg=cfg)

        train_loader = DataLoader(
            train_ds,
            batch_size=train_bs,
            shuffle=True,
            collate_fn=train_collate,
            drop_last=False,
        )

        wrapper.train()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}", leave=True)
        running_loss = 0.0
        running_n = 0

        diag_epoch_sum: Dict[str, float] = {}
        diag_epoch_n: int = 0

        for step, batch in enumerate(pbar, start=1):
            batch = _to_device(batch, device)

            meta_keys = ("sample_uid", "neg_type", "group_key")
            feed = {k: v for k, v in batch.items() if k not in meta_keys}

            with torch.cuda.amp.autocast(enabled=fp16):
                out = wrapper.forward(**feed)
                loss = out["loss"]
                if loss is None or loss < 0:
                    raise ValueError("Model forward must return 'loss' during training.")
                loss = loss / max(grad_accum, 1)

            scaler.scale(loss).backward()

            running_loss += float(loss.detach().float().cpu().item())
            running_n += 1

            if step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

            lr_now = optimizer.param_groups[0]["lr"]

            postfix = {"loss": running_loss / max(running_n, 1), "lr": lr_now}
            if hard_enabled_this_epoch:
                postfix["hard_mm"] = 1
            else:
                postfix["hard_mm"] = 0

            sub_keys = ("cls_loss", "neg_rank_loss", "mm_rank_loss", "rank_loss", "mismatch_rank_loss")
            for k in sub_keys:
                if k in out and out[k] is not None:
                    postfix[k] = float(out[k].detach().float().cpu().item())

            diag_obj: Dict[str, float] = {}
            if train_mode == "group":
                if ("neg_scores" in out and out["neg_scores"] is not None) or ("mm_scores" in out and out["mm_scores"] is not None):
                    diag_obj = _compute_group_diagnostics(out, listwise_temperature=listwise_T)

                    for dk, dv in diag_obj.items():
                        diag_epoch_sum[dk] = diag_epoch_sum.get(dk, 0.0) + float(dv)
                    diag_epoch_n += 1

                    if "neg/gap_mean" in diag_obj:
                        postfix["neg_gap"] = float(diag_obj["neg/gap_mean"])
                        postfix["neg_p0.99"] = float(diag_obj["neg/posprob_gt099"])
                    if "mm/gap_mean" in diag_obj:
                        postfix["mm_gap"] = float(diag_obj["mm/gap_mean"])
                        postfix["mm_p0.99"] = float(diag_obj["mm/posprob_gt099"])

            pbar.set_postfix(postfix)

            if wb_run is not None and global_step > 0:
                if (global_step % logging_steps) == 0:
                    log_obj = {"train/loss": running_loss / max(running_n, 1), "train/lr": lr_now, "step": global_step}
                    log_obj["train/hard_enabled"] = 1.0 if hard_enabled_this_epoch else 0.0
                    for k in ("cls_loss", "rank_loss", "mismatch_rank_loss", "neg_rank_loss", "mm_rank_loss"):
                        if k in out and out[k] is not None:
                            log_obj[f"train/{k}"] = float(out[k].detach().float().cpu().item())

                    for dk, dv in diag_obj.items():
                        log_obj[f"train_diag/{dk}"] = float(dv)

                    wb_run.log(log_obj, step=global_step)

        if train_mode == "group" and diag_epoch_n > 0:
            diag_epoch_avg = {k: (v / float(diag_epoch_n)) for k, v in diag_epoch_sum.items()}

            key_show = [
                "neg/gap_mean", "neg/gap_p90", "neg/posprob_mean", "neg/posprob_gt099",
                "mm/gap_mean", "mm/gap_p90", "mm/posprob_mean", "mm/posprob_gt099",
            ]
            compact = {k: diag_epoch_avg[k] for k in key_show if k in diag_epoch_avg}
            tqdm.write(f"[epoch {epoch}] train group diagnostics(avg over steps): {json.dumps(compact, ensure_ascii=False)}")

            if wb_run is not None:
                wb_run.log({f"train_epoch_diag/{k}": float(v) for k, v in diag_epoch_avg.items()}, step=global_step)

        ckpt_dir = _save_checkpoint(run_dir, epoch, wrapper, optimizer, scheduler, scaler, method=method)
        _rotate_checkpoints(run_dir, save_total_limit=save_total_limit)

        dev_raw = _eval_pairwise(wrapper, dev_loader, device=device, desc=f"dev epoch {epoch}", threshold=0.5)

        t_star, t_val = find_best_threshold_grid_clipped(
            dev_raw["labels"],
            dev_raw["probs"],
            metric=threshold_metric,
            step=0.01,
            min_t=0.1,
            max_t=0.9,
        )
        dev_report = _eval_pairwise(wrapper, dev_loader, device=device, desc=f"dev epoch {epoch} (t*)", threshold=t_star)
        dev_metrics = dev_report["metrics"]

        tqdm.write(
            f"[epoch {epoch}] dev threshold_search: metric={threshold_metric} best_t={t_star:.6f} best_{threshold_metric}={t_val:.6f}"
        )
        tqdm.write(f"[epoch {epoch}] dev metrics@t*: {json.dumps(dev_metrics, ensure_ascii=False)}")

        if wb_run is not None:
            wb_run.log({f"dev/{k}": v for k, v in dev_metrics.items()}, step=global_step)
            wb_run.log({f"dev/threshold": float(t_star), f"dev/threshold_metric": threshold_metric}, step=global_step)

        metric_value = float(dev_metrics.get(metric_for_best, dev_metrics.get("mean", 0.0)))

        by_type = dev_report.get("by_neg_type", {}) or {}
        if by_type:
            ordered_keys = sorted(by_type.keys())
            compact = {}
            for k in ordered_keys:
                v = by_type[k]
                compact[k] = {
                    "n": v["n"],
                    "acc": round(float(v["acc"]), 4),
                    "fp": v["fp"],
                    "fn": v["fn"],
                    "pos_rate": round(float(v["pos_rate"]), 4),
                    "pred_pos_rate": round(float(v["pred_pos_rate"]), 4),
                }

            tqdm.write(f"[epoch {epoch}] dev by_neg_type@t*: {json.dumps(compact, ensure_ascii=False)}")

            if wb_run is not None:
                log_obj = {}
                for k, v in by_type.items():
                    kk = str(k).replace("/", "_")
                    log_obj[f"dev_bytype/{kk}_acc"] = float(v["acc"])
                    log_obj[f"dev_bytype/{kk}_fp"] = float(v["fp"])
                    log_obj[f"dev_bytype/{kk}_fn"] = float(v["fn"])
                wb_run.log(log_obj, step=global_step)

        if best.is_better(metric_value):
            best.update(metric_value, epoch)
            bad_epochs = 0
            best_threshold = float(t_star)

            best_dir = os.path.join(run_dir, "best")
            if os.path.isdir(best_dir):
                shutil.rmtree(best_dir)
            _ensure_dir(best_dir)

            src_model = os.path.join(ckpt_dir, "model")
            shutil.copytree(src_model, best_dir, dirs_exist_ok=True)

            with open(best_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        metric_for_best: metric_value,
                        "metrics": dev_metrics,
                        "threshold": best_threshold,
                        "threshold_metric": threshold_metric,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            tqdm.write(
                f"[epoch {epoch}] NEW BEST: {metric_for_best}={metric_value:.4f} -> saved to {best_dir} (threshold={best_threshold:.6f})"
            )
        else:
            bad_epochs += 1
            tqdm.write(f"[epoch {epoch}] no improvement ({bad_epochs}/{early_patience}) best={best.best_value} at epoch={best.best_epoch}")

        if bad_epochs >= early_patience:
            tqdm.write(f"Early stopping triggered at epoch {epoch}.")
            break

    test_report = None
    best_dir = os.path.join(run_dir, "best")
    if auto_test_best and os.path.isdir(best_dir):
        wrapper_best, _ = _load_resume(best_dir, cfg, device=device)
        test_report = _eval_pairwise(wrapper_best, test_loader, device=device, desc="test(best@t*)", threshold=best_threshold)
        tqdm.write(f"[best] test metrics@t*: {json.dumps(test_report['metrics'], ensure_ascii=False)} (threshold={best_threshold:.6f})")
        if wb_run is not None:
            wb_run.log({f"test/{k}": v for k, v in test_report["metrics"].items()}, step=global_step)
            wb_run.log({f"test/threshold": float(best_threshold), f"test/threshold_metric": threshold_metric}, step=global_step)

    finish_wandb()

    test_report_slim = None
    if isinstance(test_report, dict):
        test_report_slim = dict(test_report)
        test_report_slim.pop("probs", None)
        test_report_slim.pop("labels", None)

    summary = {
        "run_dir": run_dir,
        "run_name": run_name,
        "train_mode": train_mode,
        "best_epoch": best.best_epoch,
        "best_value": best.best_value,
        "metric_for_best_model": metric_for_best,
        "dev_best_path": os.path.join(run_dir, "best"),
        "best_threshold": float(best_threshold),
        "threshold_metric": threshold_metric,
        "model_config": model_cfg["configs"][method],
        "test_report": test_report_slim,
    }
    return summary
