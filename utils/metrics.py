# utils/metrics.py
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss


def auc(true_y, pred_y):
    return roc_auc_score(true_y, pred_y)


def c_at_1(true_y, pred_y, threshold=0.5, undecided_eps=0.0):
    """
    c@1 with optional undecided region around threshold:
      - If undecided_eps > 0, then predictions within [threshold-eps, threshold+eps] are undecided.
      - If undecided_eps == 0, undecided happens only when pred == threshold (almost never for floats).
    """
    true_y = np.asarray(true_y, dtype=np.float32)
    pred_y = np.asarray(pred_y, dtype=np.float32)

    if undecided_eps > 0:
        nu_mask = np.abs(pred_y - float(threshold)) <= float(undecided_eps)
    else:
        nu_mask = pred_y == float(threshold)

    decided_mask = ~nu_mask
    decided_pred = (pred_y > float(threshold)).astype(np.float32)

    nc = np.sum((true_y == decided_pred)[decided_mask])
    nu = np.sum(nu_mask)

    # original c@1 definition:
    # (1/N) * (nc + nu * nc / N)
    N = len(true_y)
    if N <= 0:
        return 0.0
    return (1.0 / N) * (nc + (nu * nc / N))


def f1(true_y, pred_y, threshold=0.5):
    true_y = np.asarray(true_y, dtype=np.float32)
    pred_y = np.asarray(pred_y, dtype=np.float32)
    return f1_score(true_y, pred_y > float(threshold))


def f05u(true_y, pred_y, threshold=0.5, undecided_eps=0.0):
    """
    F0.5u: treat undecided as special.
    If undecided_eps == 0, undecided happens only when pred == threshold (almost never).
    If you want a real undecided zone, set undecided_eps > 0.
    """
    true_y = np.asarray(true_y, dtype=np.float32)
    pred_y = np.asarray(pred_y, dtype=np.float32)

    if undecided_eps > 0:
        undecided = np.abs(pred_y - float(threshold)) <= float(undecided_eps)
    else:
        undecided = pred_y == float(threshold)

    pred_pos = pred_y > float(threshold)
    pred_neg = pred_y < float(threshold)

    n_tp = np.sum(true_y * pred_pos)
    n_fn = np.sum(true_y * pred_neg)
    n_fp = np.sum((1.0 - true_y) * pred_pos)
    n_u = np.sum(undecided)

    denom = (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)
    if denom <= 0:
        return 0.0
    return (1.25 * n_tp) / denom


def brier_score(true_y, pred_y):
    try:
        return 1 - brier_score_loss(true_y, pred_y)
    except ValueError:
        return 0.0


def evaluate_all(true_y, pred_y, threshold=0.5, undecided_eps=0.0):
    """
    Compute all metrics, using `threshold` for threshold-based ones.
    AUC and Brier are threshold-free.
    """
    true_y = np.asarray(true_y, dtype=np.float32)
    pred_y = np.asarray(pred_y, dtype=np.float32)

    results = {
        "roc-auc": auc(true_y, pred_y),
        "brier": brier_score(true_y, pred_y),
        "c@1": c_at_1(true_y, pred_y, threshold=threshold, undecided_eps=undecided_eps),
        "f1": f1(true_y, pred_y, threshold=threshold),
        "f05u": f05u(true_y, pred_y, threshold=threshold, undecided_eps=undecided_eps),
    }
    results["mean"] = np.mean(list(results.values()))

    for k, v in results.items():
        results[k] = round(float(v), 3)
    return results


def _candidate_thresholds_from_probs(probs: np.ndarray) -> np.ndarray:
    """
    Generate thresholds that can change classification outcomes:
      - sort unique probs
      - take midpoints between consecutive unique probs
      - add a bit below min and above max
    This is robust even when probs are very close.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return np.asarray([0.5], dtype=np.float64)

    uniq = np.unique(p)
    if uniq.size == 1:
        # any threshold gives same predictions; return 0.5 + extremes
        u = float(uniq[0])
        return np.asarray([0.0, u, 0.5, 1.0], dtype=np.float64)

    uniq.sort()
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    # add endpoints outside range
    eps = 1e-12
    cands = np.concatenate(
        [
            np.asarray([max(0.0, uniq[0] - eps)], dtype=np.float64),
            mids.astype(np.float64),
            np.asarray([min(1.0, uniq[-1] + eps)], dtype=np.float64),
        ],
        axis=0,
    )
    # clamp
    cands = np.clip(cands, 0.0, 1.0)
    return cands


def find_best_threshold(true_y, pred_y, metric="f1", undecided_eps=0.0):
    true_y = np.asarray(true_y, dtype=np.float32)
    pred_y = np.asarray(pred_y, dtype=np.float32)

    metric = str(metric).strip().lower()
    if metric not in {"f1", "f05u", "c@1"}:
        raise ValueError(f"Unsupported metric for threshold search: {metric}")

    cands = _candidate_thresholds_from_probs(pred_y)
    # avoid extreme thresholds that are usually unstable / unhelpful
    cands = np.clip(cands, 0.01, 0.99)

    best_t = 0.5
    best_v = -1.0
    best_dist = float("inf")

    for t in cands.tolist():
        if metric == "f1":
            v = f1(true_y, pred_y, threshold=t)
        elif metric == "f05u":
            v = f05u(true_y, pred_y, threshold=t, undecided_eps=undecided_eps)
        else:
            v = c_at_1(true_y, pred_y, threshold=t, undecided_eps=undecided_eps)

        dist = abs(float(t) - 0.5)
        if (v > best_v) or (v == best_v and dist < best_dist):
            best_v = float(v)
            best_t = float(t)
            best_dist = float(dist)

    return best_t, best_v


def find_best_threshold_grid_clipped(
    true_y,
    pred_y,
    metric="f1",
    step=0.01,
    min_t=0.1,
    max_t=0.9,
    undecided_eps=0.0,
):
    """
    Clipped fixed-grid threshold search (stable version).

    Default:
      threshold ∈ [0.1, 0.9], step = 0.01
    """
    true_y = np.asarray(true_y, dtype=np.float32)
    pred_y = np.asarray(pred_y, dtype=np.float32)

    metric = str(metric).strip().lower()
    if metric not in {"f1", "f05u", "c@1"}:
        raise ValueError(f"Unsupported metric for threshold search: {metric}")

    thresholds = np.arange(min_t, max_t + 1e-8, step, dtype=np.float32)

    best_t = 0.5
    best_v = -1.0

    for t in thresholds:
        if metric == "f1":
            v = f1(true_y, pred_y, threshold=t)
        elif metric == "f05u":
            v = f05u(true_y, pred_y, threshold=t, undecided_eps=undecided_eps)
        else:
            v = c_at_1(true_y, pred_y, threshold=t, undecided_eps=undecided_eps)

        if v > best_v:
            best_v = float(v)
            best_t = float(t)

    return best_t, best_v


def auprc(true_y, pred_y):
    from sklearn.metrics import average_precision_score

    true_y = np.asarray(true_y, dtype=np.float32)
    pred_y = np.asarray(pred_y, dtype=np.float32)
    return average_precision_score(true_y, pred_y)


def _safe_roc_auc(true_y: np.ndarray, pred_y: np.ndarray) -> float:
    # roc_auc_score 会在只有单一类别时抛异常
    try:
        return roc_auc_score(true_y, pred_y)
    except Exception:
        return 0.0


def _safe_auprc(true_y: np.ndarray, pred_y: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(true_y, pred_y))
    except Exception:
        return 0.0


def expected_calibration_error(true_y, pred_y, n_bins: int = 15) -> float:
    """
    ECE: Expected Calibration Error (binary).
    Bin by confidence (pred_y), compare avg confidence vs empirical accuracy.
    """
    y = np.asarray(true_y, dtype=np.float32).reshape(-1)
    p = np.asarray(pred_y, dtype=np.float32).reshape(-1)

    if y.size == 0:
        return 0.0

    p = np.clip(p, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float32)

    ece = 0.0
    n = float(y.size)

    for i in range(int(n_bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])

        if i == int(n_bins) - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        cnt = int(np.sum(mask))
        if cnt <= 0:
            continue

        conf = float(np.mean(p[mask]))
        acc = float(np.mean((p[mask] >= 0.5).astype(np.float32) == y[mask]))
        ece += (cnt / n) * abs(acc - conf)

    return float(ece)


def evaluate_all_extended(
    true_y,
    pred_y,
    *,
    neg_types=None,
    threshold: float = 0.5,
    undecided_eps: float = 0.0,
    ece_bins: int = 15,
):
    """
    在不改变 evaluate_all 的前提下，额外返回一套扩展指标：
      - AUPRC, AUROC
      - 每个 NEG 子类的 FPR/TNR
      - MacroTNR, Worst-group TNR
      - TPR, GBA, WGBA
      - ECE

    返回：
      {
        "binary": <evaluate_all 原样输出>,
        "extra": {...扩展...}
      }
    """
    true_y = np.asarray(true_y, dtype=np.float32).reshape(-1)
    pred_y = np.asarray(pred_y, dtype=np.float32).reshape(-1)

    # 1) 原始二分类评估（保持不变）
    binary = evaluate_all(true_y, pred_y, threshold=float(threshold), undecided_eps=float(undecided_eps))

    # 2) 扩展指标（排序/校准/组公平）
    y = true_y.astype(np.int32)
    p = pred_y.astype(np.float32)
    pred = (p >= float(threshold)).astype(np.int32)

    # TPR (Recall on POS)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tpr = float(tp / max(tp + fn, 1))

    # AUROC / AUPRC
    auroc = float(_safe_roc_auc(true_y, pred_y))
    auprc_val = float(_safe_auprc(true_y, pred_y))

    # ECE
    ece = float(expected_calibration_error(true_y, pred_y, n_bins=int(ece_bins)))

    # per-neg-type FPR/TNR + Macro/Worst
    per_type = {}
    tnr_list = []

    if neg_types is None:
        neg_types_arr = np.asarray(["__UNKNOWN__"] * int(y.size), dtype=object)
    else:
        neg_types_arr = np.asarray(neg_types, dtype=object).reshape(-1)
        if neg_types_arr.size != y.size:
            # 容错：长度不一致就退化为 UNKNOWN
            neg_types_arr = np.asarray(["__UNKNOWN__"] * int(y.size), dtype=object)

    uniq_types = sorted(set([str(x) for x in neg_types_arr.tolist()]))
    for t in uniq_types:
        if t == "POS":
            continue

        mask_t = (neg_types_arr == t)
        if int(np.sum(mask_t)) <= 0:
            continue

        # 只对该类型内的真负样本计算 FPR/TNR
        mask_neg = mask_t & (y == 0)
        denom = int(np.sum(mask_neg))
        if denom <= 0:
            per_type[str(t)] = {"n": int(np.sum(mask_t)), "neg_n": 0, "fpr": 0.0, "tnr": 0.0}
            continue

        fp_t = int(np.sum(mask_neg & (pred == 1)))
        tn_t = int(np.sum(mask_neg & (pred == 0)))

        fpr_t = float(fp_t / max(fp_t + tn_t, 1))
        tnr_t = float(tn_t / max(fp_t + tn_t, 1))

        per_type[str(t)] = {
            "n": int(np.sum(mask_t)),
            "neg_n": int(denom),
            "fp": fp_t,
            "tn": tn_t,
            "fpr": fpr_t,
            "tnr": tnr_t,
        }
        tnr_list.append(tnr_t)

    macro_tnr = float(np.mean(tnr_list)) if len(tnr_list) > 0 else 0.0
    worst_tnr = float(np.min(tnr_list)) if len(tnr_list) > 0 else 0.0

    # GBA / WGBA
    gba = 0.5 * (tpr + macro_tnr)
    wgba = 0.5 * (tpr + worst_tnr)

    extra = {
        "auroc": round(float(auroc), 6),
        "auprc": round(float(auprc_val), 6),
        "tpr": round(float(tpr), 6),
        "gba": round(float(gba), 6),
        "macro_tnr": round(float(macro_tnr), 6),
        "worst_tnr": round(float(worst_tnr), 6),
        "wgba": round(float(wgba), 6),
        "ece": round(float(ece), 6),
        "per_neg_type": per_type,
    }

    return {"binary": binary, "extra": extra}


def evaluate_all_by_group(
    labels: np.ndarray,
    probs: np.ndarray,
    meta: List[Dict[str, Any]],
    *,
    group_id_key: str = "group_id",
    threshold: float = 0.5,
    undecided_eps: float = 0.0,
    ece_bins: int = 15,
):
    """
    你要求的统一入口：每次评估一个 group（或遍历所有 group）。

    返回：
      {
        "groups": {
          "<gid>": {
            "binary": ...原 evaluate_all...,
            "extra": ...扩展...,
          },
          ...
        }
      }
    """
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)

    if len(meta) != int(labels.size):
        raise ValueError(f"meta length mismatch: meta={len(meta)} labels={labels.size}")

    # group -> indices
    gid2idx = {}
    for i, m in enumerate(meta):
        gid = m.get(group_id_key, None)
        gid = str(gid) if gid is not None else "__NO_GROUP__"
        gid2idx.setdefault(gid, []).append(i)

    out = {"groups": {}}

    for gid, idxs in gid2idx.items():
        idxs_np = np.asarray(idxs, dtype=np.int64)
        y_g = labels[idxs_np]
        p_g = probs[idxs_np]
        neg_types_g = [str(meta[i].get("neg_type", "__UNKNOWN__")) for i in idxs]

        out["groups"][gid] = evaluate_all_extended(
            y_g,
            p_g,
            neg_types=neg_types_g,
            threshold=float(threshold),
            undecided_eps=float(undecided_eps),
            ece_bins=int(ece_bins),
        )

    return out


def evaluate_ranking_attribution(
    labels: np.ndarray,
    probs: np.ndarray,
    meta: List[Dict[str, Any]],
    *,
    threshold: float = 0.5,
    group_id_key: str = "group_id",
    style_uid_key: str = "style_ref_text_uid",
    query_uid_key: str = "query_text_uid",
    is_mismatch_key: str = "is_mismatch_s",
    k_recall: int = 3,
    score_key_in_meta: str = "score",
):
    """
    （归因/排序）评估：
      - Hit@1, MRR, Recall@K (K=3)
      - 支持两条线分别评估（只要数据里存在）：
          A) 固定 q，错配 s（is_mismatch_s==1 + POS）
          B) 固定 s，错配 q（is_mismatch_s==0 + POS）
      - 额外补充：整组候选（同一 group_id 下全部候选）的 ranking 指标（在保留区分的前提下新增）
      - 同时给出：按候选集难度分组（用 neg_type 近似）的 macro + worst-group
      - 在 meta 中保留每个样本的所得分值（score），通过返回 meta_with_score 实现（不修改入参 meta）

    说明：
      - “候选集难度”这里默认用该候选集里负样本的 neg_type 作为 difficulty key
        （如果你之后在 CSV 加字段例如 mismatch_level，也可以把下面 difficulty 取值改成那个字段）
    """
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)

    if len(meta) != int(labels.size):
        raise ValueError(f"meta length mismatch: meta={len(meta)} labels={labels.size}")

    # 1) meta 带分数（不修改入参 meta，避免外部副作用）
    meta_with_score: List[Dict[str, Any]] = []
    for i, m in enumerate(meta):
        mm = dict(m)
        mm[score_key_in_meta] = float(probs[i])
        meta_with_score.append(mm)

    # group -> indices
    gid2idx: Dict[str, List[int]] = {}
    for i, m in enumerate(meta):
        gid = m.get(group_id_key, None)
        gid = str(gid) if gid is not None else "__NO_GROUP__"
        gid2idx.setdefault(gid, []).append(i)

    def _rank_metrics_for_group(idxs: List[int], pos_idx: int) -> Dict[str, float]:
        # rank by score desc
        scores = probs[np.asarray(idxs, dtype=np.int64)]
        order = np.argsort(-scores)  # desc
        # locate pos
        pos_local = int(np.where(np.asarray(idxs, dtype=np.int64) == int(pos_idx))[0][0])
        rank0 = int(np.where(order == pos_local)[0][0])  # 0-based rank
        rank1 = rank0 + 1

        hit1 = 1.0 if rank1 == 1 else 0.0
        mrr = 1.0 / float(rank1)
        recallk = 1.0 if rank1 <= int(k_recall) else 0.0
        return {"hit@1": hit1, "mrr": mrr, f"recall@{k_recall}": recallk, "rank": float(rank1)}

    def _difficulty_key(idxs: List[int], pos_idx: int) -> str:
        # 取该候选集里负样本的 neg_type（多数票），作为难度分组 key
        neg_types = []
        for i in idxs:
            if int(i) == int(pos_idx):
                continue
            nt = str(meta[i].get("neg_type", "__UNKNOWN__"))
            neg_types.append(nt)
        if len(neg_types) == 0:
            return "__NO_NEG__"
        # majority
        vals, counts = np.unique(np.asarray(neg_types, dtype=object), return_counts=True)
        return str(vals[int(np.argmax(counts))])

    def _aggregate(per_group: Dict[str, Dict[str, float]], key: str) -> Dict[str, float]:
        vs = [float(v[key]) for v in per_group.values() if key in v]
        if len(vs) == 0:
            return {"macro": 0.0, "worst": 0.0}
        return {"macro": float(np.mean(vs)), "worst": float(np.min(vs))}

    # per direction
    per_group_q2s: Dict[str, Dict[str, float]] = {}
    per_group_s2q: Dict[str, Dict[str, float]] = {}

    # whole-group ranking
    per_group_whole: Dict[str, Dict[str, float]] = {}

    # difficulty -> per_group metrics
    diff_q2s: Dict[str, Dict[str, Dict[str, float]]] = {}
    diff_s2q: Dict[str, Dict[str, Dict[str, float]]] = {}
    diff_whole: Dict[str, Dict[str, Dict[str, float]]] = {}

    # optional line-wise binary classification (POS vs rest) using evaluate_all
    per_group_q2s_cls: Dict[str, Dict[str, float]] = {}
    per_group_s2q_cls: Dict[str, Dict[str, float]] = {}
    per_group_whole_cls: Dict[str, Dict[str, float]] = {}

    for gid, idxs in gid2idx.items():
        # find POS in this group
        pos_candidates = [i for i in idxs if int(labels[i]) == 1]
        if len(pos_candidates) == 0:
            continue
        pos_idx = int(pos_candidates[0])

        pos_style = meta[pos_idx].get(style_uid_key, None)
        pos_query = meta[pos_idx].get(query_uid_key, None)

        # ========== NEW: whole-group ranking (整组候选集) ==========
        if len(idxs) >= 2 and pos_idx in idxs:
            m_whole = _rank_metrics_for_group(idxs, pos_idx)
            per_group_whole[gid] = m_whole

            dkey_whole = _difficulty_key(idxs, pos_idx)
            diff_whole.setdefault(dkey_whole, {})[gid] = m_whole

            y_whole = np.asarray([1 if i == pos_idx else 0 for i in idxs], dtype=np.float32)
            p_whole = probs[np.asarray(idxs, dtype=np.int64)]
            per_group_whole_cls[gid] = evaluate_all(y_whole, p_whole, threshold=float(threshold))

        # A) q -> s line: same query uid, and mismatched-s (is_mismatch==1) + POS
        q2s_idxs: List[int] = []
        if pos_query is not None:
            for i in idxs:
                if meta[i].get(query_uid_key, None) != pos_query:
                    continue
                is_mm = int(meta[i].get(is_mismatch_key, 0) or 0)
                if int(i) == pos_idx or is_mm == 1:
                    q2s_idxs.append(i)

        if len(q2s_idxs) >= 2 and pos_idx in q2s_idxs:
            m = _rank_metrics_for_group(q2s_idxs, pos_idx)
            per_group_q2s[gid] = m

            dkey = _difficulty_key(q2s_idxs, pos_idx)
            diff_q2s.setdefault(dkey, {})[gid] = m

            # line-wise classification: POS vs rest
            y_line = np.asarray([1 if i == pos_idx else 0 for i in q2s_idxs], dtype=np.float32)
            p_line = probs[np.asarray(q2s_idxs, dtype=np.int64)]
            per_group_q2s_cls[gid] = evaluate_all(y_line, p_line, threshold=float(threshold))

        # B) s -> q line: same style uid, and mismatched-q (is_mismatch==0) + POS
        s2q_idxs: List[int] = []
        if pos_style is not None:
            for i in idxs:
                if meta[i].get(style_uid_key, None) != pos_style:
                    continue
                is_mm = int(meta[i].get(is_mismatch_key, 0) or 0)
                if int(i) == pos_idx or is_mm == 0:
                    s2q_idxs.append(i)

        if len(s2q_idxs) >= 2 and pos_idx in s2q_idxs:
            m = _rank_metrics_for_group(s2q_idxs, pos_idx)
            per_group_s2q[gid] = m

            dkey = _difficulty_key(s2q_idxs, pos_idx)
            diff_s2q.setdefault(dkey, {})[gid] = m

            y_line = np.asarray([1 if i == pos_idx else 0 for i in s2q_idxs], dtype=np.float32)
            p_line = probs[np.asarray(s2q_idxs, dtype=np.int64)]
            per_group_s2q_cls[gid] = evaluate_all(y_line, p_line, threshold=float(threshold))

    # aggregate overall
    overall_q2s = {
        "hit@1": _aggregate(per_group_q2s, "hit@1"),
        "mrr": _aggregate(per_group_q2s, "mrr"),
        f"recall@{k_recall}": _aggregate(per_group_q2s, f"recall@{k_recall}"),
    }
    overall_s2q = {
        "hit@1": _aggregate(per_group_s2q, "hit@1"),
        "mrr": _aggregate(per_group_s2q, "mrr"),
        f"recall@{k_recall}": _aggregate(per_group_s2q, f"recall@{k_recall}"),
    }
    overall_whole = {
        "hit@1": _aggregate(per_group_whole, "hit@1"),
        "mrr": _aggregate(per_group_whole, "mrr"),
        f"recall@{k_recall}": _aggregate(per_group_whole, f"recall@{k_recall}"),
    }

    # aggregate by difficulty
    by_diff_q2s: Dict[str, Dict[str, Any]] = {}
    for dkey, gdict in diff_q2s.items():
        by_diff_q2s[str(dkey)] = {
            "hit@1": _aggregate(gdict, "hit@1"),
            "mrr": _aggregate(gdict, "mrr"),
            f"recall@{k_recall}": _aggregate(gdict, f"recall@{k_recall}"),
            "n_groups": int(len(gdict)),
        }

    by_diff_s2q: Dict[str, Dict[str, Any]] = {}
    for dkey, gdict in diff_s2q.items():
        by_diff_s2q[str(dkey)] = {
            "hit@1": _aggregate(gdict, "hit@1"),
            "mrr": _aggregate(gdict, "mrr"),
            f"recall@{k_recall}": _aggregate(gdict, f"recall@{k_recall}"),
            "n_groups": int(len(gdict)),
        }

    by_diff_whole: Dict[str, Dict[str, Any]] = {}
    for dkey, gdict in diff_whole.items():
        by_diff_whole[str(dkey)] = {
            "hit@1": _aggregate(gdict, "hit@1"),
            "mrr": _aggregate(gdict, "mrr"),
            f"recall@{k_recall}": _aggregate(gdict, f"recall@{k_recall}"),
            "n_groups": int(len(gdict)),
        }

    return {
        # NEW: meta with per-sample score (kept separate to avoid mutating input meta)
        "meta_with_score": meta_with_score,
        # NEW: whole-group ranking metrics
        "whole_group": {
            "overall": overall_whole,
            "by_difficulty": by_diff_whole,
            "per_group": per_group_whole,
            "per_group_binary": per_group_whole_cls,
        },
        # original separated directions (kept)
        "q_to_s": {
            "overall": overall_q2s,
            "by_difficulty": by_diff_q2s,
            "per_group": per_group_q2s,
            "per_group_binary": per_group_q2s_cls,
        },
        "s_to_q": {
            "overall": overall_s2q,
            "by_difficulty": by_diff_s2q,
            "per_group": per_group_s2q,
            "per_group_binary": per_group_s2q_cls,
        },
    }
