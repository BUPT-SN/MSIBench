from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any, Iterator
import os
import re
import json
import glob
import gzip

import pandas as pd

from .utils import token_len, set_seed


def load_csv(path: str) -> pd.DataFrame:
    """
    通用 CSV loader（用于未来可能加入的 Blog50 等已预处理数据集）。
    最少需要字段：
      - dataset, split, author_id, text_id, text
    title/rating 允许缺失（会补默认值）。
    """
    df = pd.read_csv(path)
    required = ["dataset", "split", "author_id", "text_id", "text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["author_id"] = df["author_id"].astype(str)
    df["text_id"] = df["text_id"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["split"] = df["split"].astype(str)

    if "title" not in df.columns:
        df["title"] = ""
    else:
        df["title"] = df["title"].fillna("").astype(str)

    if "rating" not in df.columns:
        df["rating"] = None

    return df


def load_ccat50_csv(path: str, split_override: Optional[str] = None) -> pd.DataFrame:
    """Load CCAT50 CSV into the standard schema used by this pipeline.

    Expected columns in source CSV: author, split, text_id, text
    The CCAT50 file already contains split; if split_override is provided, it
    overwrites the split column (useful when separate files are provided).
    """
    df = pd.read_csv(path)
    required = ["author", "text_id", "text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    if "split" not in df.columns and split_override is None:
        raise ValueError(f"Missing 'split' column in {path}; provide split_override or include the column.")

    df = df.copy()
    df["author_id"] = df["author"].astype(str)
    df["text_id"] = df["text_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["dataset"] = "CCAT50"

    if split_override is not None:
        df["split"] = str(split_override)
    else:
        df["split"] = df["split"].astype(str)

    # CCAT50 没有 title/rating
    df["title"] = ""
    df["rating"] = None
    return df[["dataset", "split", "author_id", "text_id", "title", "rating", "text"]]


# -----------------------------
# C4 realnewslike (news) loader
# -----------------------------

_C4_TRAIN_FILE_RE = re.compile(r"c4-train\.(\d{5})-of-00512\.json\.gz$")


def _iter_jsonl_gz(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _c4_make_text_id(file_id: int, line_id: int) -> str:
    # ✅ 按你的规则：集id(=shard id) + 文本顺序id(=行号，5位)
    # 为避免 shard 拼接歧义，shard 固定 3 位（000~511）
    return f"{int(file_id):03d}{int(line_id):05d}"


def load_c4_realnewslike(
    train_glob: str,
    validation_path: str,
    sample_sizes: Dict[str, int],
    char_len_min: int,
    char_len_max: int,
    seed: int,
) -> pd.DataFrame:
    """
    读取 C4 realnewslike 新闻数据（json.gz，每行一个 JSON）：
      {"text": "...", "timestamp": "...", "url": "..."}
    - 忽略 timestamp/url
    - text 长度按字符过滤 [char_len_min, char_len_max]
    - train/dev 从 train shards 采样构造；test 从 validation 文件采样构造
    - author_id 置空字符串（不提供作者信息）
    - text_id 按 shard+line 规则构造：{shard:03d}{line:05d}
    """
    set_seed(int(seed))

    n_train = int(sample_sizes.get("train", 0))
    n_dev = int(sample_sizes.get("dev", 0))
    n_test = int(sample_sizes.get("test", 0))
    need_train_total = n_train + n_dev

    train_files = sorted(glob.glob(train_glob))
    if not train_files:
        raise FileNotFoundError(f"C4NEWS train_glob matched 0 files: {train_glob}")
    if not (validation_path and os.path.exists(validation_path)):
        raise FileNotFoundError(f"C4NEWS validation file not found: {validation_path}")

    # shuffle file order for sampling without scanning all files
    import numpy as np
    rng = np.random.default_rng(int(seed))
    file_idxs = list(range(len(train_files)))
    rng.shuffle(file_idxs)

    collected_train_dev: List[Dict[str, Any]] = []

    def _accept_text(t: str) -> bool:
        if t is None:
            return False
        t = str(t)
        L = len(t)
        return (L >= int(char_len_min)) and (L <= int(char_len_max))

    # 1) scan train shards until we have enough for train+dev
    for fi in file_idxs:
        if len(collected_train_dev) >= need_train_total:
            break
        fp = train_files[fi]
        m = _C4_TRAIN_FILE_RE.search(fp)
        if not m:
            # try basename
            m = _C4_TRAIN_FILE_RE.search(os.path.basename(fp))
        if not m:
            # if pattern doesn't match, still allow but file_id becomes index
            file_id = int(fi)
        else:
            file_id = int(m.group(1))

        line_id = 0
        for obj in _iter_jsonl_gz(fp):
            if len(collected_train_dev) >= need_train_total:
                break
            txt = obj.get("text", "")
            if not _accept_text(txt):
                line_id += 1
                continue

            tid = _c4_make_text_id(file_id=file_id, line_id=line_id)
            collected_train_dev.append(
                {
                    "dataset": "C4NEWS",
                    "split": "__TMP__",
                    "author_id": "",          # ✅ no author
                    "text_id": str(tid),
                    "title": "",
                    "rating": None,
                    "text": str(txt),
                }
            )
            line_id += 1

    if need_train_total > 0 and len(collected_train_dev) < need_train_total:
        raise RuntimeError(
            f"C4NEWS: insufficient train texts after char filter. "
            f"need(train+dev)={need_train_total}, got={len(collected_train_dev)}. "
            f"Try widening char_len range or scanning more shards."
        )

    # shuffle and split into train/dev
    rng.shuffle(collected_train_dev)
    train_part = collected_train_dev[:n_train]
    dev_part = collected_train_dev[n_train:n_train + n_dev]

    for r in train_part:
        r["split"] = "train"
    for r in dev_part:
        r["split"] = "dev"

    # 2) sample test from validation file
    collected_test: List[Dict[str, Any]] = []
    file_id_val = 999  # fixed id for validation shard; will not collide with 000~511
    line_id = 0
    for obj in _iter_jsonl_gz(validation_path):
        if len(collected_test) >= n_test:
            break
        txt = obj.get("text", "")
        if not _accept_text(txt):
            line_id += 1
            continue

        tid = _c4_make_text_id(file_id=file_id_val, line_id=line_id)
        collected_test.append(
            {
                "dataset": "C4NEWS",
                "split": "test",
                "author_id": "",          # ✅ no author
                "text_id": str(tid),
                "title": "",
                "rating": None,
                "text": str(txt),
            }
        )
        line_id += 1

    if n_test > 0 and len(collected_test) < n_test:
        raise RuntimeError(
            f"C4NEWS: insufficient validation texts after char filter. "
            f"need(test)={n_test}, got={len(collected_test)}."
        )

    df = pd.DataFrame(train_part + dev_part + collected_test)
    return df[["dataset", "split", "author_id", "text_id", "title", "rating", "text"]]


# -----------------------------
# cleaning / pool (unchanged for CCAT50)
# -----------------------------

def clean_df(df: pd.DataFrame, min_tokens: int) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["token_len"] = df["text"].apply(token_len).astype(int)

    # 过滤太短
    df = df[df["token_len"] >= int(min_tokens)].reset_index(drop=True)

    # split 规范化
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["split"] = df["split"].replace({"val": "dev", "valid": "dev", "validation": "dev"})
    return df


def _ccat50_make_dev_from_test_if_missing(df: pd.DataFrame, seed: int, ratio: float = 0.5) -> pd.DataFrame:
    """
    CCAT50 特殊处理：
    - 如果 split 中没有 dev，但存在 test，则从 test 中划出一部分作为 dev
    - 默认按作者内拆分（每个作者 test 内取一部分划到 dev），保证 dev/test 不重叠
    - 拆分在 clean_df 之后执行
    """
    if df is None or len(df) == 0:
        return df

    df = df.copy()
    df["split"] = df["split"].astype(str)

    splits = set(df["split"].unique().tolist())
    if "dev" in splits:
        return df
    if "test" not in splits:
        return df

    test_df = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        return df

    set_seed(int(seed))

    dev_indices: List[int] = []
    for author_id, grp in test_df.groupby("author_id", sort=True):
        g = grp.sample(frac=1.0, random_state=int(seed)).copy()
        n = len(g)
        if n <= 1:
            continue

        dev_n = int(n * float(ratio))
        if dev_n <= 0:
            dev_n = 1
        if dev_n >= n:
            dev_n = n - 1

        dev_indices.extend(g.index[:dev_n].tolist())

    if not dev_indices:
        # 极端兜底：全局拆
        g = test_df.sample(frac=1.0, random_state=int(seed)).copy()
        n = len(g)
        if n >= 2:
            dev_n = int(n * float(ratio))
            if dev_n <= 0:
                dev_n = 1
            if dev_n >= n:
                dev_n = n - 1
            dev_indices = g.index[:dev_n].tolist()

    if dev_indices:
        df.loc[dev_indices, "split"] = "dev"

    df = df[df["split"].isin(["train", "dev", "test"])].reset_index(drop=True)
    return df


def build_pool(df: pd.DataFrame) -> Dict[str, Dict[str, List[dict]]]:
    """
    Pool[split][author_id] = list of rows(dict)
    row dict must include: text_id, text, title, rating, dataset, split, token_len
    """
    pool: Dict[str, Dict[str, List[dict]]] = {}
    for _, r in df.iterrows():
        split = str(r["split"])
        author = str(r["author_id"])
        pool.setdefault(split, {}).setdefault(author, []).append(
            {
                "dataset": str(r["dataset"]),
                "split": split,
                "author_id": author,
                "text_id": str(r["text_id"]),
                "title": "" if pd.isna(r.get("title", "")) else str(r.get("title", "")),
                "rating": None if pd.isna(r.get("rating", None)) else r.get("rating", None),
                "text": str(r["text"]),
                "token_len": int(r["token_len"]),
            }
        )
    return pool


def load_and_clean_all(
    input_csvs: Dict[str, Dict[str, Any]],
    min_tokens: int,
    selected_datasets: Optional[List[str]] = None,
    ccat50_dev_from_test: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    input_csvs:
      { dataset: { split: path } }

    特殊支持：
      CCAT50: {"all": "./CCAT50.csv"} 或 {split: path} -> 使用文件自带 split（或覆盖）
              ✅ 若 CCAT50 不含 dev，则从 test 中取一部分作为 dev（作者内拆分，保持分离）

      C4NEWS:
        {
          "train_glob": ".../c4-train.*-of-00512.json.gz",
          "validation": ".../c4-validation.00000-of-00001.json.gz",
          "sample_sizes": {"train":1000,"dev":500,"test":500},
          "char_len": {"min":1000,"max":5000}
        }
        ✅ dev 使用 train shards 的文本构造；test 使用 validation 构造

    selected_datasets:
      若提供，则只处理列表中的 dataset，其余跳过

    ccat50_dev_from_test:
      {"seed": int, "ratio": float}
    """
    all_frames = []
    pool_all: Dict[str, Dict[str, Dict[str, List[dict]]]] = {}

    ccat50_dev_from_test = ccat50_dev_from_test or {}
    ccat_seed = int(ccat50_dev_from_test.get("seed", 20251218))
    ccat_ratio = float(ccat50_dev_from_test.get("ratio", 0.5))

    for dataset, split_map in input_csvs.items():
        if selected_datasets and dataset not in selected_datasets:
            continue

        if dataset == "CCAT50":
            frames = []
            if "all" in split_map:
                frames.append(load_ccat50_csv(split_map["all"], split_override=None))
            else:
                for split, path in split_map.items():
                    frames.append(load_ccat50_csv(path, split_override=split))
            if not frames:
                continue

            df_dataset = pd.concat(frames, axis=0, ignore_index=True)
            df_dataset = clean_df(df_dataset, min_tokens=min_tokens)

            # ✅ CCAT50 若无 dev：从 test 中拆分 dev
            df_dataset = _ccat50_make_dev_from_test_if_missing(df_dataset, seed=ccat_seed, ratio=ccat_ratio)

        elif dataset in ("C4NEWS", "C4"):
            train_glob = str(split_map.get("train_glob", "") or "")
            validation_path = str(split_map.get("validation", "") or "")
            sample_sizes = split_map.get("sample_sizes", {}) or {}
            char_len = split_map.get("char_len", {}) or {}
            char_min = int(char_len.get("min", 1000))
            char_max = int(char_len.get("max", 5000))

            df_dataset = load_c4_realnewslike(
                train_glob=train_glob,
                validation_path=validation_path,
                sample_sizes={
                    "train": int(sample_sizes.get("train", 1000)),
                    "dev": int(sample_sizes.get("dev", 500)),
                    "test": int(sample_sizes.get("test", 500)),
                },
                char_len_min=char_min,
                char_len_max=char_max,
                seed=ccat_seed,  # reuse seed source for determinism
            )
            df_dataset = clean_df(df_dataset, min_tokens=min_tokens)

        else:
            # 通用：已经是标准 schema 的数据
            frames = []
            for split, path in split_map.items():
                df = load_csv(path)
                df["dataset"] = dataset
                df["split"] = split
                df = clean_df(df, min_tokens=min_tokens)
                frames.append(df)
            if not frames:
                continue
            df_dataset = pd.concat(frames, axis=0, ignore_index=True)

        all_frames.append(df_dataset)
        pool_all[dataset] = build_pool(df_dataset)

    if not all_frames:
        raise ValueError("No dataset loaded. Please check configs.input_csvs.")
    df_all = pd.concat(all_frames, axis=0, ignore_index=True)
    return df_all, pool_all
