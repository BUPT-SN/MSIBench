#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ---------- Rules / Patterns ----------
PATTERN_NARRATIVE_OR_STYLE = re.compile(r"\b(Narrative stance|Style Profile)\b", re.IGNORECASE)
PATTERN_BEGIN_END = re.compile(r"(<<<BEGIN>>>|<<<END>>>)")
PATTERN_CHINESE = re.compile(r"[\u4e00-\u9fff]")  # common Chinese chars
PATTERN_START_BAD = re.compile(r"^\s*(1\.|#|\{)")  # starts with "1." or "#" or "{", allowing leading spaces

# NEW: abnormal text like "(" or "A" (too short / meaningless)
# - len<=1 after strip => abnormal (covers "(" and "A")
# - you can tighten/loosen later if needed
def is_abnormal_text(text: str) -> bool:
    s = (text or "").strip()
    return len(s) <= 1


def load_texts(texts_path: Path) -> pd.DataFrame:
    df = pd.read_csv(texts_path, dtype={"text_uid": str, "text_col": str, "text": str})
    need = {"text_uid", "text_col", "text"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"texts_dedup.csv missing required columns: {sorted(missing)}")
    df["text_uid"] = df["text_uid"].astype(str)
    df["text_col"] = df["text_col"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_uid_to_text(df_texts: pd.DataFrame, text_col_name: str) -> dict:
    sub = df_texts[df_texts["text_col"] == text_col_name][["text_uid", "text"]].drop_duplicates("text_uid")
    return dict(zip(sub["text_uid"], sub["text"]))


def should_clean_query_text(text: str) -> (bool, str):
    # NEW rule: abnormal format/too short
    if is_abnormal_text(text):
        return True, "abnormal query_text format (too short like '(' or 'A')"

    # Previous rules
    if PATTERN_NARRATIVE_OR_STYLE.search(text):
        return True, "contains Narrative stance / Style Profile"
    if PATTERN_BEGIN_END.search(text):
        return True, "contains <<<BEGIN>>> or <<<END>>>"
    if PATTERN_CHINESE.search(text):
        return True, "contains Chinese characters"
    if PATTERN_START_BAD.search(text):
        return True, "starts with 1. or # or {"
    return False, ""


def contiguous_group_block_indices(df: pd.DataFrame, pos_idx: int, group_col: str = "group_id"):
    """
    Given a POS row position (integer position in current df order), find the contiguous block
    of adjacent rows that share the same group_id. This enforces "must be adjacent".
    Returns (start_pos, end_pos) inclusive positions.
    """
    if group_col not in df.columns:
        # Fallback if group_id absent: remove 7-row window from POS row (still adjacent)
        start = pos_idx
        end = min(pos_idx + 6, len(df) - 1)
        return start, end

    gid = df.iloc[pos_idx][group_col]

    start = pos_idx
    while start - 1 >= 0 and df.iloc[start - 1][group_col] == gid:
        start -= 1

    end = pos_idx
    while end + 1 < len(df) and df.iloc[end + 1][group_col] == gid:
        end += 1

    return start, end


def main():
    ap = argparse.ArgumentParser(
        description="Clean sample_use.csv by rules on query_text, POS triggers contiguous group deletion, and output cleaned texts_dedup."
    )
    ap.add_argument("--sample", default="./CCAT50/samples_use.csv", help="Path to sample_use.csv")
    ap.add_argument("--texts", default="./CCAT50/texts_dedup.csv", help="Path to texts_dedup.csv")

    ap.add_argument("--out_sample", default="./CCAT50/samples_use.cleaned.csv", help="Output cleaned sample_use path")
    ap.add_argument("--out_texts", default="./CCAT50/texts_dedup.cleaned.csv", help="Output cleaned texts_dedup path")

    ap.add_argument("--removed_samples", default="./CCAT50/removed_samples.csv", help="Audit: removed sample rows")
    ap.add_argument("--removed_texts", default="./CCAT50/removed_texts.csv", help="Audit: removed texts rows (orphaned)")

    args = ap.parse_args()

    sample_path = Path(args.sample)
    texts_path = Path(args.texts)

    if not sample_path.exists():
        print(f"[ERROR] sample file not found: {sample_path}", file=sys.stderr)
        sys.exit(1)
    if not texts_path.exists():
        print(f"[ERROR] texts file not found: {texts_path}", file=sys.stderr)
        sys.exit(1)

    # Keep original order as adjacency reference
    df_s = pd.read_csv(sample_path, dtype=str)
    df_t = load_texts(texts_path)

    required_cols = {"sample_uid", "neg_type", "style_ref_text_uid", "query_text_uid"}
    missing = required_cols - set(df_s.columns)
    if missing:
        raise ValueError(f"sample_use.csv missing required columns: {sorted(missing)}")

    # Map query_text_uid -> query_text content
    uid2query = build_uid_to_text(df_t, "query_text")
    df_s["__query_text__"] = df_s["query_text_uid"].astype(str).map(uid2query).fillna("")

    # Direct rule matches (based on query_text content)
    marks = []
    reasons = []
    for txt in df_s["__query_text__"].tolist():
        flag, reason = should_clean_query_text(txt)
        marks.append(flag)
        reasons.append(reason)
    df_s["__to_remove__"] = marks
    df_s["__reason__"] = reasons

    # Seed removals: all direct hits
    remove_uids = set(df_s.loc[df_s["__to_remove__"], "sample_uid"].tolist())
    reason_map = {
        r["sample_uid"]: (r["__reason__"] or "direct rule match")
        for _, r in df_s[df_s["__to_remove__"]].iterrows()
    }

    # POS-triggered contiguous group deletion (must be adjacent)
    pos_rows = df_s[df_s["__to_remove__"] & (df_s["neg_type"] == "POS")]

    for _, pos_row in pos_rows.iterrows():
        pos_uid = pos_row["sample_uid"]
        pos_pos = df_s.index.get_loc(pos_row.name)  # integer position in df_s
        start, end = contiguous_group_block_indices(df_s, pos_pos, group_col="group_id")

        block = df_s.iloc[start:end + 1]
        block_uids = block["sample_uid"].tolist()

        for suid in block_uids:
            if suid not in remove_uids:
                reason_map[suid] = f"POS contiguous group deletion triggered by {pos_uid}"
            remove_uids.add(suid)

        if len(block_uids) != 7:
            print(
                f"[WARN] POS-triggered contiguous block for {pos_uid} has {len(block_uids)} rows (expected 7). "
                f"Block span (pos): {start}-{end}. group_id={pos_row.get('group_id', '')}",
                file=sys.stderr,
            )

    # Build removed/kept sample_use
    final_remove_mask = df_s["sample_uid"].isin(remove_uids)
    df_removed = df_s[final_remove_mask].copy()
    df_kept = df_s[~final_remove_mask].copy()

    df_removed["remove_reason"] = df_removed["sample_uid"].map(reason_map).fillna("removed")
    df_removed["query_text_preview"] = df_removed["__query_text__"].str.slice(0, 120)

    # Drop internal cols
    drop_cols = ["__query_text__", "__to_remove__", "__reason__"]
    df_kept.drop(columns=[c for c in drop_cols if c in df_kept.columns], inplace=True, errors="ignore")
    df_removed.drop(columns=[c for c in drop_cols if c in df_removed.columns], inplace=True, errors="ignore")

    # Write sample outputs
    Path(args.out_sample).parent.mkdir(parents=True, exist_ok=True)
    df_kept.to_csv(args.out_sample, index=False)
    df_removed.to_csv(args.removed_samples, index=False)

    # ---------- Clean texts_dedup ----------
    # Keep only text_uids still referenced by kept sample rows
    keep_text_uids = set(df_kept["style_ref_text_uid"].astype(str).tolist()) | set(df_kept["query_text_uid"].astype(str).tolist())

    df_texts_kept = df_t[df_t["text_uid"].astype(str).isin(keep_text_uids)].copy()
    df_texts_removed = df_t[~df_t["text_uid"].astype(str).isin(keep_text_uids)].copy()

    df_texts_kept.to_csv(args.out_texts, index=False)
    df_texts_removed.to_csv(args.removed_texts, index=False)

    # Summary
    print("=== Cleaning Summary ===")
    print(f"Input sample rows:   {len(df_s)}")
    print(f"Removed sample rows: {len(df_removed)}")
    print(f"Kept sample rows:    {len(df_kept)}")
    print(f"Output sample:       {args.out_sample}")
    print(f"Removed samples log: {args.removed_samples}")
    print("")
    print(f"Input texts rows:    {len(df_t)}")
    print(f"Kept texts rows:     {len(df_texts_kept)}")
    print(f"Removed texts rows:  {len(df_texts_removed)}")
    print(f"Output texts:        {args.out_texts}")
    print(f"Removed texts log:   {args.removed_texts}")


if __name__ == "__main__":
    main()
