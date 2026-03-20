# -*- coding: utf-8 -*-
"""
Goal:
- Reduce size of *_generated_raw.csv / *_pairwise_dataset.csv by:
  1) Keeping compact sample-level columns
  2) Moving long/repeated text fields into a deduplicated text table
  3) Replacing long texts with stable text_uid hashes in the samples table

- Output:
  - samples.csv
  - texts_dedup.csv        # ✅ MIN: only style_ref_text / query_text mappings (as requested)
  - texts_dedup_all.csv    # ✅ FULL: mappings for all LONG_TEXT_COLS (keeps original functionality)
  - samples_use.csv        # ✅ downstream-only fields (must include group ids etc.)
  - (optional) gz compressed versions
  - meta.json with stats + schema notes

Notes:
- This script is schema-tolerant: it will keep only columns that exist in your CSV.
- If you add more columns later, just put them into KEEP_COLS / LONG_TEXT_COLS / USE_ONLY_COLS.
"""

import argparse
import gzip
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import pandas as pd


# -----------------------------
# Config: edit if needed
# -----------------------------

# Long text columns to deduplicate into texts table (will become *_uid columns in samples output)
# ✅ aligned with new pipeline:
# - style_ref_text / semantic_ref_text are always meaningful
# - wrong_style_ref_text exists for NEG-wrong_ref_s rows
# - query_text exists in final dataset and in generated_raw
# - final_prompt may be large, keep it deduped for debugging
# - generated_text may exist; often equals query_text but we keep it schema-tolerantly
LONG_TEXT_COLS = [
    "style_ref_text",
    "semantic_ref_text",
    "wrong_style_ref_text",
    "query_text",
    "final_prompt",
    "generated_text",
]

# Compact "sample-level" columns to keep in samples output (only those that exist will be kept)
# ✅ updated for new sampling spec + still schema-tolerant for old/extra fields
KEEP_COLS = [
    # identity / split / labels
    "dataset",
    "split",
    "neg_type",
    "label",

    # plan/sample ids
    "plan_id",
    "sample_id",       # optional legacy
    "sample_uid",      # if already exists
    "run_id",          # optional legacy
    "timestamp",       # optional legacy

    # ✅ NEW: grouping / structure
    "s_id",
    "group_id",
    "is_mismatch_s",
    "pos_plan_id",     # link to POS for mismatch-s negatives

    # ✅ NEW: (s,a) pool ids (pair input)
    "style_author_id",
    "style_text_ids",
    "semantic_author_id",
    "semantic_text_ids",

    # wrong style info (for NEG-wrong_ref_s generation; for mismatch-s it's usually empty)
    "wrong_style_author_id",
    "wrong_style_text_ids",

    # ✅ NEW: ground-truth source (what actually produced the POS)
    "true_style_author_id",
    "true_style_text_ids",
    "true_semantic_author_id",
    "true_semantic_text_ids",

    # generation info (only for rows that were generated)
    "generator_model",
    "prompt_version",
    "random_seed",

    # query provenance (for generated rows and anchor rows)
    "query_real_author_id",
    "query_real_text_id",

    # qc (only exists in finalized/qc output)
    "qc_pass",
    "qc_reason",
]

USE_ONLY_COLS = [
    "sample_uid",
    "dataset",
    "split",
    "neg_type",
    "label",

    # ✅ structural ids for grouping/analysis
    "s_id",
    "group_id",
    "is_mismatch_s",

    # minimal texts by uid
    "style_ref_text_uid",
    "query_text_uid",
]

COLLAPSE_GENERATED_TO_QUERY_IF_EQUAL = True

NORMALIZE_COLLAPSE_WHITESPACE = True
NORMALIZE_STRIP = True

SAMPLE_UID_COL = "sample_uid"


# -----------------------------
# Helpers
# -----------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha1_uid(text: str) -> str:
    h = hashlib.sha1()
    h.update((text or "").encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    if NORMALIZE_STRIP:
        s = s.strip()
    if NORMALIZE_COLLAPSE_WHITESPACE:
        s = " ".join(s.split())
    return s


def _safe_filesize(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _write_csv(df: pd.DataFrame, path: str, gzip_output: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    if gzip_output:
        gz_path = path + ".gz"
        with open(path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            f_out.write(f_in.read())


def _ensure_sample_uid(df: pd.DataFrame) -> None:
    """Ensure each row has a unique, joinable sample uid.

    Priority:
    1) plan_id (best: stable across pipeline)
    2) sample_id (legacy)
    3) fallback to row-based uid (U000000000 style)
    """
    if SAMPLE_UID_COL in df.columns:
        base = df[SAMPLE_UID_COL].astype(str).str.strip()
    else:
        base = pd.Series([""] * len(df), index=df.index)

    def _pick(src_col: str, cur: pd.Series) -> pd.Series:
        if src_col not in df.columns:
            return cur
        src = df[src_col].astype(str).str.strip()
        m = cur.astype(str).str.strip().eq("")
        cur = cur.copy()
        cur[m] = src[m]
        return cur

    base = _pick("plan_id", base)
    base = _pick("sample_id", base)

    # final fallback: row-based uid
    m2 = base.astype(str).str.strip().eq("")
    if m2.any():
        base = base.copy()
        base[m2] = [f"U{int(i):09d}" for i in df.index[m2].tolist()]

    # ensure uniqueness (stable, deterministic)
    s = base.astype(str).str.strip()
    dup_mask = s.duplicated(keep=False)
    if dup_mask.any():
        tmp = s.copy()
        suffix = df.loc[dup_mask].groupby(tmp[dup_mask]).cumcount().astype(str)
        s.loc[dup_mask] = tmp.loc[dup_mask] + "_" + suffix

    df[SAMPLE_UID_COL] = s


# -----------------------------
# Main processing
# -----------------------------

def build_minimized(
    input_csv: str,
    out_dir: str,
    encoding: str = "utf-8",
    gzip_outputs: bool = False,
) -> Tuple[str, str, str, str]:
    """
    Build:
      - samples.csv (compact sample table)
      - texts_dedup.csv (MIN: only style_ref_text/query_text mappings)   ✅ per request
      - texts_dedup_all.csv (FULL: mappings for all LONG_TEXT_COLS)      ✅ keep original behavior
      - samples_use.csv (downstream-only table, must include group ids etc.)
      - meta.json
    Returns: (samples_path, texts_min_path, use_path, meta_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv, encoding=encoding, dtype=str, keep_default_na=False)

    # Ensure a unique sample uid for joining samples_use -> samples
    _ensure_sample_uid(df)

    # Optional collapse check (kept as no-op / debug hook)
    if COLLAPSE_GENERATED_TO_QUERY_IF_EQUAL:
        if "generated_text" in df.columns and "query_text" in df.columns:
            _ = (df["generated_text"].astype(str) == df["query_text"].astype(str))

    # Determine which columns exist
    keep_cols =s = [c for c in KEEP_COLS if c in df.columns]
    long_text_cols = [c for c in LONG_TEXT_COLS if c in df.columns]

    # Build texts_dedup table (FULL)
    texts_rows: List[Dict[str, str]] = []
    text_uid_map: Dict[Tuple[str, str], str] = {}  # (col_name, normalized_text) -> uid

    def _get_uid(col: str, text: Any) -> str:
        norm = _norm_text(text)
        key = (col, norm)
        if key in text_uid_map:
            return text_uid_map[key]
        uid = _sha1_uid(col + "||" + norm)
        text_uid_map[key] = uid
        texts_rows.append({"text_uid": uid, "text_col": col, "text": norm})
        return uid

    # Create uid columns in samples table for each long text col
    samples = df[keep_cols].copy() if keep_cols else pd.DataFrame({SAMPLE_UID_COL: df[SAMPLE_UID_COL].copy()})

    # Always ensure sample_uid in samples
    if SAMPLE_UID_COL not in samples.columns and SAMPLE_UID_COL in df.columns:
        samples[SAMPLE_UID_COL] = df[SAMPLE_UID_COL]

    for col in long_text_cols:
        uid_col = f"{col}_uid"
        samples[uid_col] = df[col].apply(lambda x: _get_uid(col, x))

    # Drop raw long text columns from samples if they were also kept
    for col in long_text_cols:
        if col in samples.columns:
            samples = samples.drop(columns=[col])

    # Build FULL texts df
    texts_df_all = pd.DataFrame(texts_rows)
    if not texts_df_all.empty:
        texts_df_all = texts_df_all.sort_values(["text_col", "text_uid"]).reset_index(drop=True)

    # Build MIN texts df (only style_ref_text_uid + query_text_uid mappings)
    # i.e. only keep rows where text_col in {"style_ref_text", "query_text"} if those cols exist
    min_text_cols = [c for c in ["style_ref_text", "query_text"] if c in long_text_cols]
    if min_text_cols and not texts_df_all.empty:
        texts_df_min = texts_df_all[texts_df_all["text_col"].isin(min_text_cols)].reset_index(drop=True)
    else:
        texts_df_min = pd.DataFrame(columns=["text_uid", "text_col", "text"])

    # samples_use: only keep selected cols that exist
    use_cols = [c for c in USE_ONLY_COLS if c in samples.columns]
    # Guarantee sample_uid presence in samples_use (for join)
    if SAMPLE_UID_COL in samples.columns and SAMPLE_UID_COL not in use_cols:
        use_cols = [SAMPLE_UID_COL] + use_cols
        # de-dup
        seen = set()
        use_cols = [c for c in use_cols if not (c in seen or seen.add(c))]

    samples_use = samples[use_cols].copy() if use_cols else samples[[SAMPLE_UID_COL]].copy()

    # Write outputs
    samples_path = os.path.join(out_dir, "samples.csv")
    texts_min_path = os.path.join(out_dir, "texts_dedup.csv")          # ✅ requested minimal file
    texts_all_path = os.path.join(out_dir, "texts_dedup_all.csv")      # ✅ keep full mapping
    use_path = os.path.join(out_dir, "samples_use.csv")
    meta_path = os.path.join(out_dir, "meta.json")

    _write_csv(samples, samples_path, gzip_output=gzip_outputs)
    _write_csv(texts_df_min, texts_min_path, gzip_output=gzip_outputs)
    _write_csv(texts_df_all, texts_all_path, gzip_output=gzip_outputs)
    _write_csv(samples_use, use_path, gzip_output=gzip_outputs)

    meta = {
        "created_utc": _now_utc_iso(),
        "input_csv": input_csv,
        "input_bytes": _safe_filesize(input_csv),
        "out_dir": out_dir,
        "gzip_outputs": bool(gzip_outputs),
        "rows": int(len(df)),
        "samples_rows": int(len(samples)),
        "texts_dedup_rows": int(len(texts_df_min)),          # MIN
        "texts_dedup_all_rows": int(len(texts_df_all)),      # FULL
        "samples_use_rows": int(len(samples_use)),
        "kept_columns": keep_cols,
        "long_text_columns": long_text_cols,
        "schema_notes": {
            "samples": {
                "desc": "Compact sample-level table. Long texts replaced by *_uid columns.",
                "columns": list(samples.columns),
            },
            "texts_dedup": {
                "desc": "MIN deduplicated text table: ONLY style_ref_text + query_text mappings.",
                "columns": list(texts_df_min.columns) if not texts_df_min.empty else ["text_uid", "text_col", "text"],
                "included_text_cols": min_text_cols,
            },
            "texts_dedup_all": {
                "desc": "FULL deduplicated text table for all LONG_TEXT_COLS (debug/backfill).",
                "columns": list(texts_df_all.columns) if not texts_df_all.empty else ["text_uid", "text_col", "text"],
            },
            "samples_use": {
                "desc": "Downstream-only table. Includes group identifiers (s_id/group_id/is_mismatch_s/pos_plan_id) "
                        "and minimal text uids. Join back to samples via sample_uid.",
                "columns": list(samples_use.columns),
            },
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return samples_path, texts_min_path, use_path, meta_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_csv",
        type=str,
        default="pairwise_dataset_qc_pass.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="filters_v2/qcpass",
    )
    ap.add_argument("--encoding", type=str, default="utf-8")
    ap.add_argument("--gzip_outputs", action="store_true")
    args = ap.parse_args()

    samples_path, texts_min_path, use_path, meta_path = build_minimized(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        encoding=args.encoding,
        gzip_outputs=args.gzip_outputs,
    )

    texts_all_path = os.path.join(args.out_dir, "texts_dedup_all.csv")

    print("[OK] Wrote:")
    print(" -", samples_path)
    print(" -", texts_min_path)   # MIN (style_ref_text/query_text only)
    print(" -", texts_all_path)   # FULL (all long text cols)
    print(" -", use_path)
    print(" -", meta_path)


if __name__ == "__main__":
    main()
