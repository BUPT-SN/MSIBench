# utils/dataloader.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# -------------------------
# Pairwise dataset (for dev/test)
# -------------------------
@dataclass
class SampleItem:
    sample_uid: str
    split: str
    label: int
    neg_type: str
    is_mismatch_s: int

    group_id: str

    style_uid: str
    query_uid: str

    style_text: str
    query_text: str


class PairTextDataset(Dataset):
    def __init__(self, items: List[SampleItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        return {
            "sample_uid": it.sample_uid,
            "split": it.split,
            "label": int(it.label),
            "neg_type": it.neg_type,
            "is_mismatch_s": int(it.is_mismatch_s),

            "group_id": it.group_id,

            "style_uid": it.style_uid,
            "query_uid": it.query_uid,

            "style_ref_text_uid": it.style_uid,
            "query_text_uid": it.query_uid,

            "style_text": it.style_text,
            "query_text": it.query_text,
        }


# -------------------------
# Group dataset (for train)
# -------------------------
@dataclass
class GroupItem:
    group_key: str

    style_uid: str
    style_text: str

    pos_query_uid: str
    pos_query_text: str

    neg_query_uids: List[str]
    neg_query_texts: List[str]
    neg_query_types: List[str]

    mismatch_style_uids: List[str]
    mismatch_style_texts: List[str]
    mismatch_style_types: List[str]


class GroupTextDataset(Dataset):
    """
    Each item corresponds to one group_id.
    It produces TWO listwise tasks:
      - (POS + N_NEG)         fixed s_true, varying q
      - (POS + N_MISMATCH)    fixed q_pos, varying s (mismatch_s line)
    """

    def __init__(self, items: List[GroupItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        return {
            "group_key": it.group_key,

            "style_uid": it.style_uid,
            "style_text": it.style_text,

            "pos_query_uid": it.pos_query_uid,
            "pos_query_text": it.pos_query_text,

            "neg_query_uids": list(it.neg_query_uids),
            "neg_query_texts": list(it.neg_query_texts),
            "neg_query_types": list(it.neg_query_types),

            "mismatch_style_uids": list(it.mismatch_style_uids),
            "mismatch_style_texts": list(it.mismatch_style_texts),
            "mismatch_style_types": list(it.mismatch_style_types),
        }


# -------------------------
# CSV helpers
# -------------------------
def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _build_text_map(texts_df: pd.DataFrame) -> Dict[str, str]:
    req = {"text_uid", "text"}
    missing = req - set(texts_df.columns)
    if missing:
        raise ValueError(f"texts_dedup.cleaned.csv missing columns: {sorted(missing)}")

    df = texts_df.copy()
    df["text_uid"] = df["text_uid"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return dict(zip(df["text_uid"].tolist(), df["text"].tolist()))


def _resolve_paths(cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    d = cfg.get("data", {}) or {}
    data_root = str(d.get("data_root", "datasets"))
    dataset_name = str(d.get("dataset", "")).strip()
    if not dataset_name:
        raise ValueError("cfg.data.dataset is required.")
    ds_dir = os.path.join(data_root, dataset_name)
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Dataset dir not found: {ds_dir}")

    samples_csv = str(d.get("samples_csv", "samples_use.cleaned.csv"))
    texts_csv = str(d.get("texts_csv", "texts_dedup.cleaned.csv"))
    hard_csv = str(d.get("hard_minings_csv", "hard_minings.csv"))

    return (
        os.path.join(ds_dir, samples_csv),
        os.path.join(ds_dir, texts_csv),
        os.path.join(ds_dir, hard_csv),
    )


def _validate_samples_df(samples_df: pd.DataFrame, cfg_data: Dict[str, Any]) -> None:
    style_uid_col = str(cfg_data.get("style_uid_col", "style_ref_text_uid"))
    query_uid_col = str(cfg_data.get("query_uid_col", "query_text_uid"))
    label_col = str(cfg_data.get("label_col", "label"))
    split_col = str(cfg_data.get("split_col", "split"))
    neg_type_col = str(cfg_data.get("neg_type_col", "neg_type"))

    required = {style_uid_col, query_uid_col, label_col, split_col, neg_type_col}
    missing = required - set(samples_df.columns)
    if missing:
        raise ValueError(f"samples csv missing columns: {sorted(missing)}")


def _apply_split_and_excludes(cfg: Dict[str, Any], df: pd.DataFrame, split: str) -> pd.DataFrame:
    data_cfg = cfg.get("data", {}) or {}
    split_col = str(data_cfg.get("split_col", "split"))
    neg_type_col = str(data_cfg.get("neg_type_col", "neg_type"))

    split = str(split).strip()
    out = df[df[split_col].astype(str) == split].copy()

    exclude = []
    if split == str(data_cfg.get("train_split", "train")):
        exclude = data_cfg.get("train_exclude_neg_types", []) or []
    elif split == str(data_cfg.get("test_split", "test")):
        exclude = data_cfg.get("test_exclude_neg_types", []) or []
    exclude = [str(x) for x in exclude]
    if exclude:
        out = out[~out[neg_type_col].astype(str).isin(exclude)].copy()
    return out


# -------------------------
# Hard mining selection helpers
# -------------------------
def _parse_hard_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {}) or {}
    hm = (data_cfg.get("hard_mining", {}) or {}).copy()

    def _norm_level(x: Any, default: str) -> str:
        v = str(x or default).strip().lower()
        if v not in {"easy", "mid", "hard"}:
            raise ValueError(f"hard_mining level must be one of easy/mid/hard, got {v}")
        return v

    return {
        "start_epoch": int(hm.get("start_epoch", 1)),
        "train_level": _norm_level(hm.get("train_level", "easy"), "easy"),
        "eval_level": _norm_level(hm.get("eval_level", "easy"), "easy"),
        "include_pos": bool(hm.get("include_pos", True)),
    }


def _level_to_allowed_neg_types(level: str, *, downward: bool) -> List[str]:
    """
    hard_minings.csv uses neg_type like:
      - NEG-Mismatch-S         (easy)
      - NEG-Mismatch-S-mid     (mid)
      - NEG-Mismatch-S-hard    (hard)
    """
    level = str(level).strip().lower()
    if level == "easy":
        return ["NEG-Mismatch-S"]
    if level == "mid":
        return ["NEG-Mismatch-S", "NEG-Mismatch-S-mid"] if downward else ["NEG-Mismatch-S-mid"]
    if level == "hard":
        return ["NEG-Mismatch-S", "NEG-Mismatch-S-mid", "NEG-Mismatch-S-hard"] if downward else ["NEG-Mismatch-S-hard"]
    raise ValueError(level)


def _filter_hard_df(
    cfg: Dict[str, Any],
    hard_df: pd.DataFrame,
    split: str,
    *,
    level: str,
    downward: bool,
    include_pos: bool,
) -> pd.DataFrame:
    """
    Extra data only comes from:
      - mismatch_s line samples (is_mismatch_s==1) with selected neg_type levels
      - POS samples (label==1) if include_pos=True
    Excludes:
      - NEG-Anchor / NEG-neutral / NEG-wrong_ref_s from hard_df (even if present)
    """
    data_cfg = cfg.get("data", {}) or {}
    grouping = (data_cfg.get("grouping", {}) or {})
    mismatch_col = str(grouping.get("mismatch_flag_col", "is_mismatch_s"))

    neg_type_col = str(data_cfg.get("neg_type_col", "neg_type"))
    label_col = str(data_cfg.get("label_col", "label"))

    if mismatch_col not in hard_df.columns:
        hard_df = hard_df.copy()
        hard_df[mismatch_col] = 0

    df = _apply_split_and_excludes(cfg, hard_df, split)

    allowed_mm_types = set(_level_to_allowed_neg_types(level, downward=downward))

    mm_part = df[(df[mismatch_col].astype(int) == 1) & (df[label_col].astype(int) == 0)].copy()
    mm_part = mm_part[mm_part[neg_type_col].astype(str).isin(list(allowed_mm_types))].copy()

    if include_pos:
        pos_part = df[df[label_col].astype(int) == 1].copy()
        # 强制只保留 POS（避免 hard_df 中混入其它 neg_type）
        pos_part = pos_part[pos_part[neg_type_col].astype(str) == "POS"].copy() if (neg_type_col in pos_part.columns) else pos_part
        out = pd.concat([pos_part, mm_part], axis=0, ignore_index=True)
    else:
        out = mm_part

    return out


def _drop_mismatch_from_base_samples(cfg: Dict[str, Any], base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Requirement #4:
    新 hard_minings.csv 已包含 easy mismatch，所以 base samples 不再读取 mismatch_s 线，避免混乱。
    """
    data_cfg = cfg.get("data", {}) or {}
    grouping = (data_cfg.get("grouping", {}) or {})
    mismatch_col = str(grouping.get("mismatch_flag_col", "is_mismatch_s"))

    if mismatch_col not in base_df.columns:
        return base_df

    out = base_df[base_df[mismatch_col].astype(int) == 0].copy()
    return out


# -------------------------
# Public loaders
# -------------------------
def load_split_dataset(
    cfg: Dict[str, Any],
    split: str,
    *,
    epoch: Optional[int] = None,
    use_hard_override: Optional[bool] = None,
    hard_level_override: Optional[str] = None,
    hard_downward_override: Optional[bool] = None,
) -> PairTextDataset:
    """
    Pair dataset for train/dev/test.
    - Base data from samples_csv (s->q line + POS), but mismatch_s line is DROPPED (req #4).
    - Extra data from hard_minings_csv:
        - mismatch_s samples of selected difficulty
        - (optional) POS if hard_minings.csv contains extra POS

    Config:
      data.hard_mining.start_epoch / train_level / eval_level / include_pos
    """
    data_cfg = cfg.get("data", {}) or {}
    hm_cfg = _parse_hard_cfg(cfg)

    train_split = str(data_cfg.get("train_split", "train"))
    dev_split = str(data_cfg.get("dev_split", "dev"))
    test_split = str(data_cfg.get("test_split", "test"))

    samples_path, texts_path, hard_path = _resolve_paths(cfg)

    base_df = _read_csv(samples_path)
    texts_df = _read_csv(texts_path)
    hard_df = _read_csv(hard_path) if os.path.isfile(hard_path) else pd.DataFrame()

    _validate_samples_df(base_df, data_cfg)
    if not hard_df.empty:
        _validate_samples_df(hard_df, data_cfg)

    text_map = _build_text_map(texts_df)

    style_uid_col = str(data_cfg.get("style_uid_col", "style_ref_text_uid"))
    query_uid_col = str(data_cfg.get("query_uid_col", "query_text_uid"))
    label_col = str(data_cfg.get("label_col", "label"))
    split_col = str(data_cfg.get("split_col", "split"))
    neg_type_col = str(data_cfg.get("neg_type_col", "neg_type"))

    sample_uid_col = "sample_uid" if "sample_uid" in base_df.columns else None
    group_id_col = "group_id" if "group_id" in base_df.columns else None

    grouping = (data_cfg.get("grouping", {}) or {})
    mismatch_col = str(grouping.get("mismatch_flag_col", "is_mismatch_s"))

    # base: apply split/excludes, then drop mismatch line
    base_part = _apply_split_and_excludes(cfg, base_df, split)
    base_part = _drop_mismatch_from_base_samples(cfg, base_part)

    # decide whether to use hard extra
    if use_hard_override is not None:
        use_hard = bool(use_hard_override)
    else:
        if split == train_split:
            ep = int(epoch or 1)
            use_hard = ep >= int(hm_cfg["start_epoch"])
        else:
            use_hard = True  # dev/test 默认引入（但由 eval_level 控制“用哪级”）

    extra_part = pd.DataFrame()
    if use_hard and (not hard_df.empty):
        if split == train_split:
            level = str(hard_level_override or hm_cfg["train_level"])
            downward = bool(hard_downward_override) if hard_downward_override is not None else True
        else:
            level = str(hard_level_override or hm_cfg["eval_level"])
            downward = bool(hard_downward_override) if hard_downward_override is not None else False

        extra_part = _filter_hard_df(
            cfg,
            hard_df,
            split,
            level=level,
            downward=downward,
            include_pos=bool(hm_cfg["include_pos"]),
        )

    # merge
    if extra_part is not None and len(extra_part) > 0:
        df = pd.concat([base_part, extra_part], axis=0, ignore_index=True)
    else:
        df = base_part

    # build items
    items: List[SampleItem] = []

    # for hard part, sample_uid column should exist too; but to be safe, detect per-row
    has_uid_col = "sample_uid" in df.columns
    has_group_col = "group_id" in df.columns
    has_mismatch_col = mismatch_col in df.columns

    for i, row in df.iterrows():
        s_uid = str(row[style_uid_col])
        q_uid = str(row[query_uid_col])

        if s_uid not in text_map:
            raise KeyError(f"style uid not found in texts map: {s_uid} (row={i})")
        if q_uid not in text_map:
            raise KeyError(f"query uid not found in texts map: {q_uid} (row={i})")

        style_text = text_map[s_uid].strip()
        query_text = text_map[q_uid].strip()
        if not style_text:
            raise ValueError(f"Empty style_text for uid={s_uid} (row={i})")
        if not query_text:
            raise ValueError(f"Empty query_text for uid={q_uid} (row={i})")

        sample_uid = str(row["sample_uid"]) if has_uid_col else (str(row[sample_uid_col]) if sample_uid_col else str(i))
        neg_type = str(row[neg_type_col]) if neg_type_col in df.columns else ""
        label = int(row[label_col])

        is_mismatch_s = int(row[mismatch_col]) if has_mismatch_col else 0

        if has_group_col:
            group_id = str(row["group_id"])
        elif group_id_col is not None and group_id_col in df.columns:
            group_id = str(row[group_id_col])
        else:
            group_id = str(sample_uid)

        items.append(
            SampleItem(
                sample_uid=sample_uid,
                split=str(row[split_col]) if split_col in df.columns else str(split),
                label=label,
                neg_type=neg_type,
                is_mismatch_s=is_mismatch_s,
                group_id=group_id,
                style_uid=s_uid,
                query_uid=q_uid,
                style_text=style_text,
                query_text=query_text,
            )
        )

    return PairTextDataset(items)


def load_group_dataset(
    cfg: Dict[str, Any],
    split: str,
    *,
    epoch: Optional[int] = None,
    use_hard_override: Optional[bool] = None,
    hard_level_override: Optional[str] = None,
) -> GroupTextDataset:
    """
    Group dataset for training.

    Base samples from samples_csv:
      - provides POS + NEG query-side (s->q) line
      - mismatch_s line is removed (req #4)

    mismatch_s pool from hard_minings_csv:
      - uses selected difficulty (train: downward include)
      - in train+hard enabled: mismatch candidates are RANDOMLY sampled to fill n_mismatch (req #3)
    """
    import hashlib

    data_cfg = cfg.get("data", {}) or {}
    hm_cfg = _parse_hard_cfg(cfg)
    grouping = (data_cfg.get("grouping", {}) or {}).copy()

    group_id_col = str(grouping.get("group_id_col", "group_id"))
    mismatch_col = str(grouping.get("mismatch_flag_col", "is_mismatch_s"))
    pos_neg_types = [str(x) for x in (grouping.get("pos_neg_types", []) or [])]
    sampling = str(grouping.get("sampling", "truncate")).strip()

    n_neg = int(grouping.get("n_neg", 3))
    n_mm = int(grouping.get("n_mismatch", 3))

    seed = int((cfg.get("project", {}) or {}).get("seed", 42))

    samples_path, texts_path, hard_path = _resolve_paths(cfg)
    base_df = _read_csv(samples_path)
    texts_df = _read_csv(texts_path)
    hard_df = _read_csv(hard_path) if os.path.isfile(hard_path) else pd.DataFrame()

    _validate_samples_df(base_df, data_cfg)
    if not hard_df.empty:
        _validate_samples_df(hard_df, data_cfg)

    text_map = _build_text_map(texts_df)

    if group_id_col not in base_df.columns:
        raise ValueError(f"group_id_col='{group_id_col}' not found in samples csv columns")

    style_uid_col = str(data_cfg.get("style_uid_col", "style_ref_text_uid"))
    query_uid_col = str(data_cfg.get("query_uid_col", "query_text_uid"))
    label_col = str(data_cfg.get("label_col", "label"))
    neg_type_col = str(data_cfg.get("neg_type_col", "neg_type"))

    # base split + excludes, then drop mismatch line
    df = _apply_split_and_excludes(cfg, base_df, split)
    df = _drop_mismatch_from_base_samples(cfg, df)

    # decide whether to use hard for mismatch pool
    train_split = str(data_cfg.get("train_split", "train"))
    if use_hard_override is not None:
        use_hard = bool(use_hard_override)
    else:
        if split == train_split:
            ep = int(epoch or 1)
            use_hard = ep >= int(hm_cfg["start_epoch"])
        else:
            use_hard = True

    # build mismatch pool map from hard_df (only when enabled)
    mm_map: Dict[str, List[Tuple[str, str]]] = {}
    if use_hard and (not hard_df.empty):
        level = str(hard_level_override or hm_cfg["train_level"]).strip().lower()
        # train: downward include
        allowed_mm_types = set(_level_to_allowed_neg_types(level, downward=True))

        hd = _apply_split_and_excludes(cfg, hard_df, split)
        if group_id_col not in hd.columns:
            raise ValueError(f"hard_minings.csv missing group_id_col='{group_id_col}'")

        if mismatch_col not in hd.columns:
            hd = hd.copy()
            hd[mismatch_col] = 0

        # keep only mismatch_s line
        mm_only = hd[(hd[mismatch_col].astype(int) == 1) & (hd[label_col].astype(int) == 0)].copy()
        mm_only = mm_only[mm_only[neg_type_col].astype(str).isin(list(allowed_mm_types))].copy()

        # exclude any unexpected neg_types explicitly (req #1)
        # （hard_minings 里若混入 NEG-Anchor/NEG-neutral/NEG-wrong_ref_s，也不会进来）
        for _, r in mm_only.iterrows():
            gk = str(r[group_id_col])
            s_uid = str(r[style_uid_col])
            t = str(r[neg_type_col])
            if s_uid in text_map and text_map.get(s_uid, "").strip():
                mm_map.setdefault(gk, []).append((s_uid, t))

    def stable_int(s: str) -> int:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def pick_fixed_indices(n_total: int, n: int, rng: random.Random) -> List[int]:
        if n <= 0:
            return []
        if n_total <= 0:
            raise ValueError("Required pool is empty but n>0. Fix data or set n=0.")
        idxs = list(range(n_total))
        if n_total >= n:
            if sampling == "random":
                return rng.sample(idxs, n)
            return idxs[:n]
        out = list(idxs)
        while len(out) < n:
            out.append(out[len(out) % n_total])
        return out

    def pick_random_fill(n_total: int, n: int, rng: random.Random) -> List[int]:
        """
        For group mode + hard enabled:
          randomly choose until fill n (with replacement allowed if pool < n).
        """
        if n <= 0:
            return []
        if n_total <= 0:
            raise ValueError("Required pool is empty but n>0. Fix data or set n=0.")
        if n_total >= n:
            return rng.sample(list(range(n_total)), n)
        # with replacement
        return [rng.randrange(0, n_total) for _ in range(n)]

    items: List[GroupItem] = []
    grouped = df.groupby(df[group_id_col].astype(str), sort=False)

    for gkey, gdf in grouped:
        # s_true
        style_uids = gdf[style_uid_col].astype(str).unique().tolist()
        style_uid = style_uids[0]
        if style_uid not in text_map:
            raise KeyError(f"[group {gkey}] style uid not found: {style_uid}")
        style_text = text_map[style_uid].strip()
        if not style_text:
            raise ValueError(f"[group {gkey}] empty style text for uid={style_uid}")

        gdf2 = gdf.copy()
        gdf2["__label"] = gdf2[label_col].astype(int)
        gdf2["__neg_type"] = gdf2[neg_type_col].astype(str)

        # pos row (from base)
        pos_rows = gdf2[gdf2["__label"] == 1]
        if len(pos_rows) != 1:
            raise ValueError(f"[group {gkey}] expected exactly 1 POS (from base samples), got {len(pos_rows)}")
        pos_row = pos_rows.iloc[0]

        pos_uid = str(pos_row[query_uid_col])
        if pos_uid not in text_map:
            raise KeyError(f"[group {gkey}] pos query uid not found: {pos_uid}")
        pos_text = text_map[pos_uid].strip()
        if not pos_text:
            raise ValueError(f"[group {gkey}] empty pos query text for uid={pos_uid}")

        # NEG pool (s->q line): label==0, m==0 已经保证（因为我们 drop 过 mismatch 线）
        neg_pool = gdf2[gdf2["__label"] == 0]
        if pos_neg_types:
            neg_pool = neg_pool[neg_pool["__neg_type"].isin(pos_neg_types)]

        neg_uids_all = [str(x) for x in neg_pool[query_uid_col].tolist()]
        neg_types_all = [str(x) for x in neg_pool["__neg_type"].tolist()]

        neg_pairs: List[Tuple[str, str]] = []
        for u, t in zip(neg_uids_all, neg_types_all):
            if u in text_map and text_map.get(u, "").strip():
                neg_pairs.append((u, t))

        rng = random.Random((seed * 1000003) ^ stable_int(str(gkey)))

        # NEG pick: keep old behavior
        try:
            neg_pick_idx = pick_fixed_indices(len(neg_pairs), n_neg, rng) if n_neg > 0 else []
        except ValueError as e:
            raise ValueError(f"[group {gkey}] NEG pool empty but n_neg={n_neg}. {e}")

        neg_uids = [neg_pairs[j][0] for j in neg_pick_idx] if neg_pick_idx else []
        neg_types = [neg_pairs[j][1] for j in neg_pick_idx] if neg_pick_idx else []
        neg_texts = [text_map[u].strip() for u in neg_uids]

        # MISMATCH pool from hard_minings
        mm_pairs = mm_map.get(str(gkey), []) if use_hard else []
        if not mm_pairs:
            # hard 未启用或没数据：为了不破坏原训练流程，允许 mm 为空，但如果 n_mismatch>0 会报错（与旧逻辑一致）
            if n_mm > 0:
                raise ValueError(f"[group {gkey}] MISMATCH pool empty but n_mismatch={n_mm}. Check hard_minings.csv / config.")
            mm_style_uids, mm_types, mm_style_texts = [], [], []
        else:
            # req #3: group 模式固定数量，hard 启用时“随机选直到填满”
            pick_fn = pick_random_fill if (split == train_split and use_hard) else pick_fixed_indices
            try:
                mm_pick_idx = pick_fn(len(mm_pairs), n_mm, rng) if n_mm > 0 else []
            except ValueError as e:
                raise ValueError(f"[group {gkey}] MISMATCH pool empty but n_mismatch={n_mm}. {e}")

            mm_style_uids = [mm_pairs[j][0] for j in mm_pick_idx] if mm_pick_idx else []
            mm_types = [mm_pairs[j][1] for j in mm_pick_idx] if mm_pick_idx else []
            mm_style_texts = [text_map[u].strip() for u in mm_style_uids]

        # sanity: align lengths
        if len(neg_uids) != len(neg_texts) or len(neg_uids) != len(neg_types):
            raise ValueError(f"[group {gkey}] neg alignment mismatch")
        if len(mm_style_uids) != len(mm_style_texts) or len(mm_style_uids) != len(mm_types):
            raise ValueError(f"[group {gkey}] mismatch alignment mismatch")

        items.append(
            GroupItem(
                group_key=str(gkey),
                style_uid=style_uid,
                style_text=style_text,
                pos_query_uid=pos_uid,
                pos_query_text=pos_text,
                neg_query_uids=neg_uids,
                neg_query_texts=neg_texts,
                neg_query_types=neg_types if len(neg_types) == len(neg_uids) else ["__UNKNOWN__"] * len(neg_uids),
                mismatch_style_uids=mm_style_uids,
                mismatch_style_texts=mm_style_texts,
                mismatch_style_types=mm_types if len(mm_types) == len(mm_style_uids) else ["__UNKNOWN__"] * len(mm_style_uids),
            )
        )

    if len(items) == 0:
        raise ValueError(f"No group items built for split={split}. Check group_id_col/labels.")

    return GroupTextDataset(items)


# -------------------------
# Manual features (unchanged)
# -------------------------
class _ManualFeatureStore:
    """
    Loads manual features from:
      texts_dedup.style.uids.csv
      texts_dedup.style.scalars.npy
      texts_dedup.style.char_hash.npy
      texts_dedup.style.word_hash.npy
      texts_dedup.style.scaler_params.npz
      texts_dedup.style.scalar_keys.json (only for dim check; optional)
    Provides:
      get_raw(uid_list) -> np.ndarray [B, raw_dim]
    raw = [scalars_scaled ; l2(char_hash) ; l2(word_hash)]
    """
    def __init__(self, style_dir: str):
        import json
        import pandas as pd
        import os

        uids_path = os.path.join(style_dir, "texts_dedup.style.uids.csv")
        scalars_path = os.path.join(style_dir, "texts_dedup.style.scalars.npy")
        char_path = os.path.join(style_dir, "texts_dedup.style.char_hash.npy")
        word_path = os.path.join(style_dir, "texts_dedup.style.word_hash.npy")
        scaler_path = os.path.join(style_dir, "texts_dedup.style.scaler_params.npz")
        keys_path = os.path.join(style_dir, "texts_dedup.style.scalar_keys.json")

        if not os.path.isfile(uids_path):
            raise FileNotFoundError(f"manual feature uids not found: {uids_path}")
        if not os.path.isfile(scalars_path):
            raise FileNotFoundError(f"manual feature scalars not found: {scalars_path}")
        if not os.path.isfile(char_path):
            raise FileNotFoundError(f"manual feature char_hash not found: {char_path}")
        if not os.path.isfile(word_path):
            raise FileNotFoundError(f"manual feature word_hash not found: {word_path}")
        if not os.path.isfile(scaler_path):
            raise FileNotFoundError(f"manual feature scaler_params not found: {scaler_path}")

        uids = pd.read_csv(uids_path)["text_uid"].astype(str).tolist()
        self.uid2idx = {u: i for i, u in enumerate(uids)}

        self.scalars = np.load(scalars_path, mmap_mode="r")
        self.char_hash = np.load(char_path, mmap_mode="r")
        self.word_hash = np.load(word_path, mmap_mode="r")

        sc = np.load(scaler_path)
        self.scaler_type = str(sc["type"][0])
        self.hash_dim = int(sc["hash_dim"][0])

        self.mean = sc["mean"].astype(np.float32) if "mean" in sc else None
        self.std = sc["std"].astype(np.float32) if "std" in sc else None
        self.median = sc["median"].astype(np.float32) if "median" in sc else None
        self.iqr = sc["iqr"].astype(np.float32) if "iqr" in sc else None

        if os.path.isfile(keys_path):
            with open(keys_path, "r", encoding="utf-8") as f:
                keys = json.load(f)
            if int(self.scalars.shape[1]) != len(keys):
                raise ValueError(f"scalar dim mismatch: scalars.npy={self.scalars.shape[1]} vs keys={len(keys)}")

        if int(self.char_hash.shape[1]) != self.hash_dim or int(self.word_hash.shape[1]) != self.hash_dim:
            raise ValueError(
                f"hash dim mismatch: scaler_params.hash_dim={self.hash_dim}, "
                f"char_hash={self.char_hash.shape}, word_hash={self.word_hash.shape}"
            )

    @staticmethod
    def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        denom = np.sqrt(np.sum(x * x, axis=1, keepdims=True)) + eps
        return x / denom

    def _scale_scalars(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        if self.scaler_type == "standard":
            if self.mean is None or self.std is None:
                raise ValueError("standard scaler missing mean/std")
            return (x - self.mean) / (self.std + 1e-12)
        if self.scaler_type == "robust":
            if self.median is None or self.iqr is None:
                raise ValueError("robust scaler missing median/iqr")
            return (x - self.median) / (self.iqr + 1e-12)
        raise ValueError(self.scaler_type)

    def get_raw(self, uid_list: List[str]) -> np.ndarray:
        idx = []
        for u in uid_list:
            if u not in self.uid2idx:
                raise KeyError(f"manual feature uid not found in uids.csv: {u}")
            idx.append(self.uid2idx[u])

        sc = np.asarray(self.scalars[idx], dtype=np.float32)
        sc = self._scale_scalars(sc)

        ch = np.asarray(self.char_hash[idx], dtype=np.float32)
        wd = np.asarray(self.word_hash[idx], dtype=np.float32)

        ch = self._l2norm(ch)
        wd = self._l2norm(wd)

        return np.concatenate([sc, ch, wd], axis=1).astype(np.float32, copy=False)


# -------------------------
# Collators (UNCHANGED)
# -------------------------
def make_pair_collator(
    model_method: str,
    tokenizer,
    max_length: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    method = str(model_method).strip()

    use_feats = False
    store = None
    if cfg is not None:
        m = (cfg.get("model", {}) or {}).get("configs", {}) or {}
        bc = (m.get("bi_encoder", {}) or {})
        use_feats = bool(bc.get("use_manual_features", False))
        if use_feats and method == "bi_encoder":
            data_cfg = cfg.get("data", {}) or {}
            data_root = str(data_cfg.get("data_root", "datasets"))
            dataset_name = str(data_cfg.get("dataset", "")).strip()
            subdir = str(bc.get("manual_feature_subdir", "style"))
            style_dir = os.path.join(data_root, dataset_name, subdir)
            store = _ManualFeatureStore(style_dir)

    def _meta_list(batch: List[Dict[str, Any]], key: str, default=None) -> List[Any]:
        out = []
        for b in batch:
            out.append(b.get(key, default))
        return out

    def collate_cross(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        styles = [b["style_text"] for b in batch]
        queries = [b["query_text"] for b in batch]
        labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)

        is_mismatch_s = torch.tensor([int(b.get("is_mismatch_s", 0)) for b in batch], dtype=torch.int64)

        enc = tokenizer(
            styles,
            queries,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,

            "is_mismatch": is_mismatch_s,

            "is_mismatch_s": is_mismatch_s,
            "sample_uid": _meta_list(batch, "sample_uid", default=None),
            "neg_type": _meta_list(batch, "neg_type", default="__UNKNOWN__"),
            "group_id": _meta_list(batch, "group_id", default=None),
            "style_ref_text_uid": _meta_list(batch, "style_ref_text_uid", default=None),
            "query_text_uid": _meta_list(batch, "query_text_uid", default=None),

            "style_uid": _meta_list(batch, "style_uid", default=None),
            "query_uid": _meta_list(batch, "query_uid", default=None),
            "split": _meta_list(batch, "split", default=None),
        }

    def collate_bi(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        styles = [b["style_text"] for b in batch]
        queries = [b["query_text"] for b in batch]
        labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)

        is_mismatch_s = torch.tensor([int(b.get("is_mismatch_s", 0)) for b in batch], dtype=torch.int64)

        s_enc = tokenizer(styles, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")
        q_enc = tokenizer(queries, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")

        out = {
            "style_input_ids": s_enc["input_ids"],
            "style_attention_mask": s_enc["attention_mask"],
            "query_input_ids": q_enc["input_ids"],
            "query_attention_mask": q_enc["attention_mask"],
            "labels": labels,

            "is_mismatch": is_mismatch_s,

            "is_mismatch_s": is_mismatch_s,
            "sample_uid": _meta_list(batch, "sample_uid", default=None),
            "neg_type": _meta_list(batch, "neg_type", default="__UNKNOWN__"),
            "group_id": _meta_list(batch, "group_id", default=None),
            "style_ref_text_uid": _meta_list(batch, "style_ref_text_uid", default=None),
            "query_text_uid": _meta_list(batch, "query_text_uid", default=None),

            "style_uid": _meta_list(batch, "style_uid", default=None),
            "query_uid": _meta_list(batch, "query_uid", default=None),
            "split": _meta_list(batch, "split", default=None),
        }

        if use_feats:
            if store is None:
                raise RuntimeError("use_manual_features=True but feature store not initialized")
            style_uids = [str(b["style_uid"]) for b in batch]
            query_uids = [str(b["query_uid"]) for b in batch]
            s_raw = torch.from_numpy(store.get_raw(style_uids)).float()
            q_raw = torch.from_numpy(store.get_raw(query_uids)).float()
            out["style_feats_raw"] = s_raw
            out["query_feats_raw"] = q_raw

        return out

    if method == "cross_encoder":
        return collate_cross
    if method == "bi_encoder":
        return collate_bi
    raise ValueError(f"Unknown model_method={model_method} for pair collator.")


def make_group_collator(
    model_method: str,
    tokenizer,
    max_length: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    method = str(model_method).strip()

    use_feats = False
    store = None
    if cfg is not None:
        m = (cfg.get("model", {}) or {}).get("configs", {}) or {}
        bc = (m.get("bi_encoder", {}) or {})
        use_feats = bool(bc.get("use_manual_features", False))
        if use_feats and method == "bi_encoder":
            data_cfg = cfg.get("data", {}) or {}
            data_root = str(data_cfg.get("data_root", "datasets"))
            dataset_name = str(data_cfg.get("dataset", "")).strip()
            subdir = str(bc.get("manual_feature_subdir", "style"))
            style_dir = os.path.join(data_root, dataset_name, subdir)
            store = _ManualFeatureStore(style_dir)

    def collate_cross(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        styles_true = [b["style_text"] for b in batch]
        neg_lists = [([b["pos_query_text"]] + list(b["neg_query_texts"])) for b in batch]
        B = len(batch)
        K_neg = len(neg_lists[0])

        flat_styles_neg, flat_queries_neg = [], []
        for i in range(B):
            flat_styles_neg.extend([styles_true[i]] * K_neg)
            flat_queries_neg.extend(neg_lists[i])

        enc_neg = tokenizer(
            flat_styles_neg,
            flat_queries_neg,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )

        L1 = enc_neg["input_ids"].shape[-1]
        neg_input_ids = enc_neg["input_ids"].reshape(B, K_neg, L1)
        neg_attention_mask = enc_neg["attention_mask"].reshape(B, K_neg, L1)

        mm_style_lists = [([b["style_text"]] + list(b["mismatch_style_texts"])) for b in batch]
        K_mm = len(mm_style_lists[0])

        flat_styles_mm, flat_queries_mm = [], []
        for i in range(B):
            q_pos = batch[i]["pos_query_text"]
            flat_styles_mm.extend(mm_style_lists[i])
            flat_queries_mm.extend([q_pos] * K_mm)

        enc_mm = tokenizer(
            flat_styles_mm,
            flat_queries_mm,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )

        L2 = enc_mm["input_ids"].shape[-1]
        mm_input_ids = enc_mm["input_ids"].reshape(B, K_mm, L2)
        mm_attention_mask = enc_mm["attention_mask"].reshape(B, K_mm, L2)

        mm_is_mismatch = torch.zeros((B, K_mm), dtype=torch.bool)
        if K_mm > 1:
            mm_is_mismatch[:, 1:] = True

        group_keys = [b["group_key"] for b in batch]

        return {
            "group_key": group_keys,
            "group_id": group_keys,

            "style_ref_text_uid": [b["style_uid"] for b in batch],
            "pos_query_uid": [b["pos_query_uid"] for b in batch],
            "neg_query_uids": [list(b["neg_query_uids"]) for b in batch],
            "mismatch_style_uids": [list(b["mismatch_style_uids"]) for b in batch],

            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_attention_mask,
            "mm_input_ids": mm_input_ids,
            "mm_attention_mask": mm_attention_mask,
            "mm_is_mismatch": mm_is_mismatch,
        }

    def collate_bi(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        styles_true = [b["style_text"] for b in batch]
        B = len(batch)

        s_enc = tokenizer(
            styles_true,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )

        neg_lists = [([b["pos_query_text"]] + list(b["neg_query_texts"])) for b in batch]
        K_neg = len(neg_lists[0])
        flat_neg_q = [q for qs in neg_lists for q in qs]

        q_neg = tokenizer(
            flat_neg_q,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )

        L1 = q_neg["input_ids"].shape[-1]
        neg_query_input_ids = q_neg["input_ids"].reshape(B, K_neg, L1)
        neg_query_attention_mask = q_neg["attention_mask"].reshape(B, K_neg, L1)

        mm_style_lists = [([b["style_text"]] + list(b["mismatch_style_texts"])) for b in batch]
        K_mm = len(mm_style_lists[0])

        flat_mm_s = [s for ss in mm_style_lists for s in ss]
        mm_s_enc = tokenizer(
            flat_mm_s,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )

        L2 = mm_s_enc["input_ids"].shape[-1]
        mm_style_input_ids = mm_s_enc["input_ids"].reshape(B, K_mm, L2)
        mm_style_attention_mask = mm_s_enc["attention_mask"].reshape(B, K_mm, L2)

        fixed_qs = [b["pos_query_text"] for b in batch]
        mm_q_enc = tokenizer(
            fixed_qs,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )

        mm_is_mismatch = torch.zeros((B, K_mm), dtype=torch.bool)
        if K_mm > 1:
            mm_is_mismatch[:, 1:] = True

        group_keys = [b["group_key"] for b in batch]

        out = {
            "group_key": group_keys,
            "group_id": group_keys,

            "style_ref_text_uid": [b["style_uid"] for b in batch],
            "pos_query_uid": [b["pos_query_uid"] for b in batch],
            "neg_query_uids": [list(b["neg_query_uids"]) for b in batch],
            "mismatch_style_uids": [list(b["mismatch_style_uids"]) for b in batch],

            "style_input_ids": s_enc["input_ids"],
            "style_attention_mask": s_enc["attention_mask"],
            "neg_query_input_ids": neg_query_input_ids,
            "neg_query_attention_mask": neg_query_attention_mask,

            "mm_style_input_ids": mm_style_input_ids,
            "mm_style_attention_mask": mm_style_attention_mask,
            "mm_fixed_query_input_ids": mm_q_enc["input_ids"],
            "mm_fixed_query_attention_mask": mm_q_enc["attention_mask"],

            "mm_is_mismatch": mm_is_mismatch,
        }

        if use_feats:
            if store is None:
                raise RuntimeError("use_manual_features=True but feature store not initialized")

            style_uids = [str(b["style_uid"]) for b in batch]
            out["style_feats_raw"] = torch.from_numpy(store.get_raw(style_uids)).float()

            neg_uid_lists = [([str(b["pos_query_uid"])] + [str(u) for u in b["neg_query_uids"]]) for b in batch]
            flat_neg_uids = [u for us in neg_uid_lists for u in us]
            neg_raw = torch.from_numpy(store.get_raw(flat_neg_uids)).float()
            out["neg_query_feats_raw"] = neg_raw.reshape(B, K_neg, -1)

            mm_uid_lists = [([str(b["style_uid"])] + [str(u) for u in b["mismatch_style_uids"]]) for b in batch]
            flat_mm_s_uids = [u for us in mm_uid_lists for u in us]
            mm_s_raw = torch.from_numpy(store.get_raw(flat_mm_s_uids)).float()
            out["mm_style_feats_raw"] = mm_s_raw.reshape(B, K_mm, -1)

            fixed_q_uids = [str(b["pos_query_uid"]) for b in batch]
            out["mm_fixed_query_feats_raw"] = torch.from_numpy(store.get_raw(fixed_q_uids)).float()

        return out

    if method == "cross_encoder":
        return collate_cross
    if method == "bi_encoder":
        return collate_bi
    raise ValueError(f"Unknown model_method={model_method} for group collator.")
