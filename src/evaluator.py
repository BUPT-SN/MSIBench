# src/evaluator.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataloader import load_split_dataset, make_pair_collator
from utils.metrics import evaluate_all


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logits_to_probs_np(logits: torch.Tensor) -> np.ndarray:
    """
    Convert logits to probabilities for binary tasks.

    Supports:
      - logits shape [N] or [N,1]  (BCE-style): prob=sigmoid(logit)
      - logits shape [N,2]         (CE-style): score=logit_pos-logit_neg; prob=sigmoid(score)
    """
    x = logits.detach().float().cpu().numpy()
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim == 2 and x.shape[1] == 2:
        score = x[:, 1] - x[:, 0]
        return _sigmoid(score.astype(np.float32))
    if x.ndim == 1:
        return _sigmoid(x.astype(np.float32))
    raise ValueError(f"Unsupported logits shape for binary probs: {x.shape}")


def _safe_json_dump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            x = json.load(f)
        return x if isinstance(x, dict) else None
    except Exception:
        return None


def _update_best_metrics_json(best_metrics_path: str, payload: Dict[str, Any]) -> None:
    """
    Update/overwrite best_metrics.json with richer fields while keeping existing fields if needed.
    Minimal policy:
      - load existing dict if any
      - shallow-merge new payload (new keys overwrite old keys)
    """
    old = _safe_read_json(best_metrics_path) or {}
    merged = dict(old)
    merged.update(payload)
    _safe_json_dump(best_metrics_path, merged)


def _neg_type_acc_summary(
    labels: np.ndarray,
    probs: np.ndarray,
    meta: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """
    For each neg_type (including POS if present), record:
      - total_n
      - acc
      - correct_est (use exact correct count as “estimated correct”)
    """
    y = labels.astype(np.int32).reshape(-1)
    p = probs.astype(np.float32).reshape(-1)
    pred_y = (p >= float(threshold)).astype(np.int32)

    type2idx: Dict[str, List[int]] = {}
    for i, m in enumerate(meta):
        nt = str(m.get("neg_type", "__UNKNOWN__"))
        type2idx.setdefault(nt, []).append(i)

    out: Dict[str, Dict[str, Any]] = {}
    for nt, idxs in type2idx.items():
        idxs_np = np.asarray(idxs, dtype=np.int64)
        n = int(idxs_np.size)
        if n <= 0:
            continue
        correct = int(np.sum(pred_y[idxs_np] == y[idxs_np]))
        acc = float(correct / max(n, 1))
        out[nt] = {
            "total_n": n,
            "acc": round(acc, 6),
            "correct_est": correct,
        }
    return out


def _make_text_summary(
    split: str,
    n_samples: int,
    threshold: float,
    threshold_source: str,
    threshold_search: Dict[str, Any],
    metrics: Dict[str, Any],
    extra_metrics: Dict[str, Any],
    task2_metrics: Dict[str, Any],
    neg_type_acc: Dict[str, Dict[str, Any]],
    paths: Dict[str, str],
) -> str:
    """
    Human-readable summary text. Keep it stable for log/compare.
    """
    lines: List[str] = []
    lines.append(f"=== TEST SUMMARY ({split}) ===")
    lines.append(f"n_samples: {n_samples}")
    lines.append(f"threshold: {threshold:.6f}")
    lines.append(f"threshold_source: {threshold_source}")
    if threshold_search:
        lines.append(f"threshold_search: {json.dumps(threshold_search, ensure_ascii=False)}")
    lines.append("")

    lines.append("[Binary metrics]")
    for k, v in metrics.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("[Extra metrics]")
    # keep top-level extra keys concise
    for k in ["auroc", "auprc", "tpr", "gba", "macro_tnr", "worst_tnr", "wgba", "ece"]:
        if k in extra_metrics:
            lines.append(f"- {k}: {extra_metrics[k]}")
    lines.append("")

    lines.append("[Neg-type accuracy summary]  (total_n / acc / correct_est)")
    for nt in sorted(neg_type_acc.keys()):
        s = neg_type_acc[nt]
        lines.append(f"- {nt}: {s['total_n']} / {s['acc']} / {s['correct_est']}")
    lines.append("")

    lines.append("[Task2 attribution / ranking metrics]")
    # 直接 dump，避免你后续扩展字段时 text_summary 不同步
    lines.append(json.dumps(task2_metrics, ensure_ascii=False, indent=2))
    lines.append("")

    lines.append("[Artifacts]")
    for k, v in paths.items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines) + "\n"


def _resolve_threshold_plan(cfg: Dict[str, Any], split: str) -> Tuple[float, str, Dict[str, Any]]:
    """
    Threshold policy (fix for cross-dataset evaluation):
      - If evaluating test split: search best threshold on dev split with step=0.01, metric=f1,
        then apply that threshold to test.
      - If evaluating dev split: search best threshold on dev split itself (same grid).
      - Otherwise: default threshold=0.5 (no search).

    Returns:
      (threshold, threshold_source, threshold_search_info)
    """
    data_cfg = cfg.get("data", {}) or {}
    dev_split = str(data_cfg.get("dev_split", "dev"))
    test_split = str(data_cfg.get("test_split", "test"))

    split = str(split).strip()
    if split == test_split:
        # search on dev, apply to test (the required fix)
        return 0.5, "searched_on_dev", {"search_split": dev_split}
    if split == dev_split:
        # helpful for reporting dev metrics too
        return 0.5, "searched_on_self_dev", {"search_split": dev_split}
    return 0.5, "default_0.5", {}


@torch.no_grad()
def _run_inference(
    cfg: Dict[str, Any],
    wrapper,
    split: str,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Run model inference on a given split, returning:
      probs: float32 [N]
      labels: float32 [N]
      meta: list[dict] per-sample
    """
    from tqdm.auto import tqdm

    model_cfg = cfg.get("model", {}) or {}
    method = str(model_cfg.get("method", "cross_encoder")).strip()

    ds = load_split_dataset(cfg, split)
    max_len = int(getattr(wrapper, "max_length", 512))
    collate_fn = make_pair_collator(method, wrapper.tokenizer, max_len, cfg=cfg)

    loader = DataLoader(
        ds,
        batch_size=int((cfg.get("train", {}) or {}).get("eval_batch_size", 32)),
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_probs: List[float] = []
    all_labels: List[int] = []
    all_meta: List[Dict[str, Any]] = []

    running_loss_sum = 0.0
    running_logit_sum = 0.0
    running_n = 0

    pbar = tqdm(loader, desc=f"Eval[{split}]", dynamic_ncols=True)

    for batch in pbar:
        labels = batch.pop("labels")
        sample_uids = batch.pop("sample_uid", None)
        neg_types = batch.pop("neg_type", None)

        group_ids = batch.pop("group_id", None)
        style_uids = batch.pop("style_ref_text_uid", None)
        query_uids = batch.pop("query_text_uid", None)
        is_mismatch = batch.pop("is_mismatch_s", None)

        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        labels_t = labels.to(device=device)
        out = wrapper.forward(**batch, labels=labels_t)

        logits = out["logits"]

        if logits.dim() == 2 and logits.shape[1] == 2:
            score = logits[:, 1] - logits[:, 0]
            running_logit_sum += float(score.detach().float().sum().cpu())
            bs = int(score.shape[0])
        else:
            running_logit_sum += float(logits.detach().float().sum().cpu())
            bs = int(logits.shape[0])

        running_n += bs

        if "loss" in out:
            loss_val = float(out["loss"].detach().cpu())
            running_loss_sum += loss_val * bs

        if running_n > 0:
            postfix = {"mean_logit": running_logit_sum / running_n}
            if running_loss_sum > 0:
                postfix["loss"] = running_loss_sum / running_n
            pbar.set_postfix(postfix)

        probs_np = _logits_to_probs_np(logits)
        labels_np = labels.detach().float().cpu().numpy().reshape(-1)

        if probs_np.shape[0] != labels_np.shape[0]:
            raise ValueError(f"Eval probs/labels length mismatch: probs={probs_np.shape} labels={labels_np.shape}")

        all_probs.extend([float(x) for x in probs_np.tolist()])
        all_labels.extend([int(x) for x in labels_np.tolist()])

        def _to_list(x, default=None):
            if x is None:
                return [default] * int(labels_np.shape[0])
            if isinstance(x, (list, tuple)):
                return list(x)
            if torch.is_tensor(x):
                return [v for v in x.detach().cpu().reshape(-1).tolist()]
            return [x] * int(labels_np.shape[0])

        sample_uids_l = _to_list(sample_uids, default=None)
        neg_types_l = _to_list(neg_types, default="__UNKNOWN__")
        group_ids_l = _to_list(group_ids, default=None)
        style_uids_l = _to_list(style_uids, default=None)
        query_uids_l = _to_list(query_uids, default=None)
        is_mismatch_l = _to_list(is_mismatch, default=0)

        for i in range(int(labels_np.shape[0])):
            all_meta.append(
                {
                    "sample_uid": sample_uids_l[i],
                    "neg_type": str(neg_types_l[i]),
                    "group_id": group_ids_l[i],
                    "style_ref_text_uid": style_uids_l[i],
                    "query_text_uid": query_uids_l[i],
                    "is_mismatch_s": int(is_mismatch_l[i]) if is_mismatch_l[i] is not None else 0,
                    "label": int(labels_np[i]),
                    "logit": float((logits[i, 1] - logits[i, 0]).detach().float().cpu().item())
                    if (logits.dim() == 2 and logits.shape[1] == 2)
                    else float(logits[i].detach().float().cpu().item()),
                }
            )

    probs = np.asarray(all_probs, dtype=np.float32)
    labels_arr = np.asarray(all_labels, dtype=np.float32)
    return probs, labels_arr, all_meta


@torch.no_grad()
def evaluate_checkpoint(
    cfg: Dict[str, Any],
    checkpoint_dir: str,
    split: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load wrapper, run inference, compute metrics.

    FIX (this request):
      - Do NOT read threshold from best_metrics.json for evaluation.
      - Instead, search best threshold on the validation/dev split with step=0.01 (metric=f1),
        then apply that threshold to test evaluation.
      - For dev split evaluation, search on dev itself (same grid) for reporting.

    额外输出保持不变（你原来新增的那套）：
      1) best_metrics.json 记录更全：metrics / extra_metrics / task2_metrics / neg_type_acc_summary / paths
      2) 输出 text_summary_{split}.txt
      3) 记录每个样本预测结果到 predictions_{split}.json（dict: sample_uid -> {score,pred_y,label}）
      4) 保留 predictions_{split}.jsonl（逐行详细 meta）
    """
    from utils.checkpoint import load_model_checkpoint
    from utils.metrics import evaluate_all_extended, evaluate_ranking_attribution, find_best_threshold_grid_clipped

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wrapper = load_model_checkpoint(cfg, checkpoint_dir, device=device)
    wrapper.eval()

    # 1) Decide threshold policy (search vs default)
    threshold, threshold_source, threshold_search_info = _resolve_threshold_plan(cfg, split)

    # 2) If need search: run inference on dev split and grid-search best threshold
    if threshold_source in {"searched_on_dev", "searched_on_self_dev"}:
        search_split = str(threshold_search_info.get("search_split", "dev"))
        dev_probs, dev_labels, _ = _run_inference(cfg, wrapper, search_split, device=str(device))

        best_t, best_v = find_best_threshold_grid_clipped(
            true_y=dev_labels,
            pred_y=dev_probs,
            metric="f1",
            step=0.01,
            min_t=0.1,
            max_t=0.9,
            undecided_eps=0.0,
        )
        threshold = float(best_t)
        threshold_search_info = {
            "metric": "f1",
            "step": 0.01,
            "min_t": 0.1,
            "max_t": 0.9,
            "search_split": search_split,
            "best_value": float(best_v),
        }

    # 3) Run inference on the requested split and evaluate using the (searched) threshold
    probs, labels_arr, all_meta = _run_inference(cfg, wrapper, split, device=str(device))

    metrics = evaluate_all(labels_arr, probs, threshold=threshold)

    ext = evaluate_all_extended(
        labels_arr,
        probs,
        neg_types=[m.get("neg_type", "__UNKNOWN__") for m in all_meta],
        threshold=threshold,
        undecided_eps=0.0,
        ece_bins=15,
    )

    task2 = evaluate_ranking_attribution(
        labels_arr.astype(np.int32),
        probs,
        all_meta,
        threshold=threshold,
        k_recall=3,
    )

    neg_type_acc = _neg_type_acc_summary(labels_arr, probs, all_meta, threshold=threshold)

    # ============ outputs ============
    out_jsonl_path = os.path.join(checkpoint_dir, f"predictions_{split}.jsonl")
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for i, m in enumerate(all_meta):
            mm = dict(m)
            mm["prob"] = float(probs[i])
            f.write(json.dumps(mm, ensure_ascii=False) + "\n")

    out_json_path = os.path.join(checkpoint_dir, f"predictions_{split}.json")
    pred_y_arr = (probs >= float(threshold)).astype(np.int32)
    sample_pred_dict: Dict[str, Dict[str, Any]] = {}
    for i, m in enumerate(all_meta):
        uid = m.get("sample_uid", None)
        if uid is None:
            uid = str(i)
        uid = str(uid)
        sample_pred_dict[uid] = {
            "score": float(probs[i]),
            "pred_y": int(pred_y_arr[i]),
            "label": int(m.get("label", int(labels_arr[i]))),
        }
    _safe_json_dump(out_json_path, sample_pred_dict)

    text_summary_path = os.path.join(checkpoint_dir, f"text_summary_{split}.txt")
    text_summary = _make_text_summary(
        split=split,
        n_samples=int(labels_arr.shape[0]),
        threshold=float(threshold),
        threshold_source=str(threshold_source),
        threshold_search=dict(threshold_search_info or {}),
        metrics=metrics,
        extra_metrics=ext["extra"],
        task2_metrics=task2,
        neg_type_acc=neg_type_acc,
        paths={
            "predictions_jsonl": out_jsonl_path,
            "predictions_json": out_json_path,
            "text_summary": text_summary_path,
        },
    )
    os.makedirs(os.path.dirname(text_summary_path), exist_ok=True)
    with open(text_summary_path, "w", encoding="utf-8") as f:
        f.write(text_summary)

    # enrich best_metrics.json
    best_metrics_path = os.path.join(os.path.dirname(checkpoint_dir), "best_metrics.json")
    _update_best_metrics_json(
        best_metrics_path,
        {
            "split": split,
            "n_samples": int(labels_arr.shape[0]),
            "threshold": float(threshold),
            "threshold_source": str(threshold_source),
            "threshold_search": dict(threshold_search_info or {}),
            "metrics": metrics,
            "extra_metrics": ext["extra"],
            "task2_metrics": task2,
            "neg_type_acc_summary": neg_type_acc,
            "artifacts": {
                "predictions_jsonl": out_jsonl_path,
                "predictions_json": out_json_path,
                "text_summary": text_summary_path,
            },
        },
    )

    return {
        "split": split,
        "n_samples": int(labels_arr.shape[0]),
        "threshold": float(threshold),
        "threshold_source": str(threshold_source),
        "threshold_search": dict(threshold_search_info or {}),
        "metrics": metrics,
        "extra_metrics": ext["extra"],
        "task2_metrics": task2,
        "neg_type_acc_summary": neg_type_acc,
        "predictions_path": out_jsonl_path,
        "predictions_json_path": out_json_path,
        "text_summary_path": text_summary_path,
        "best_metrics_path": best_metrics_path,
    }
