from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import dumps_list_str


def _parse_list_str(s: Any) -> List[str]:
    if s is None:
        return []
    if isinstance(s, list):
        return [str(x) for x in s]
    ss = str(s).strip()
    if not ss:
        return []
    try:
        v = json.loads(ss)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return [x.strip() for x in ss.split(",") if x.strip()]


def _safe_token_len(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    return len(t.split())


def _contains_markdown(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False

    if "```" in s:
        return True

    lines = s.splitlines()
    for ln in lines:
        t = ln.lstrip()
        if not t:
            continue
        if t.startswith("#"):
            return True
        if re.match(r"^[-*+]\s+\S", t):
            return True
        if re.match(r"^\d+\.\s+\S", t):
            return True
        if re.match(r"^>\s+\S", t):
            return True
    return False

def finalize_dataset(
    plan_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    pool_all: Dict[str, Any],
    out_path: str,
    qc_cfg_path: Optional[str] = None,
    qc_out_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge plan + generated rows into final pairwise dataset under the NEW sampling spec.

    QC:
    - ✅ Remove old qc_cfg.json-based logic.
    - ✅ Fully adopt clean_samples.py QC logic (adapted):
        1) Direct rule matches on query_text
        2) POS-triggered contiguous group deletion by group_id adjacency in the output df order
    """
    # --- local qc logic copied/adapted from clean_samples.py ---
    PATTERN_NARRATIVE_OR_STYLE = re.compile(r"\b(Narrative stance|Style Profile)\b", re.IGNORECASE)
    PATTERN_BEGIN_END = re.compile(r"(<<<BEGIN>>>|<<<END>>>)")
    PATTERN_CHINESE = re.compile(r"[\u4e00-\u9fff]")
    PATTERN_START_BAD = re.compile(r"^\s*(1\.|#|\{)")

    def is_abnormal_text(text: str) -> bool:
        s = (text or "").strip()
        return len(s) <= 1

    def should_clean_query_text(text: str) -> (bool, str):
        if is_abnormal_text(text):
            return True, "abnormal query_text format (too short like '(' or 'A')"
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
        Same idea as clean_samples.py:
        Given a POS row position (integer position in current df order), find the contiguous block
        of adjacent rows that share the same group_id.
        Returns (start_pos, end_pos) inclusive positions.
        """
        if group_col not in df.columns:
            start = pos_idx
            end = pos_idx
            return start, end

        gid = df.iloc[pos_idx][group_col]
        start = pos_idx
        while start - 1 >= 0 and df.iloc[start - 1][group_col] == gid:
            start -= 1
        end = pos_idx
        while end + 1 < len(df) and df.iloc[end + 1][group_col] == gid:
            end += 1
        return start, end

    plan = plan_df.copy()
    gen = generated_df.copy()

    plan["plan_id"] = plan["plan_id"].astype(str)
    gen["plan_id"] = gen["plan_id"].astype(str)

    gen_map: Dict[str, Dict[str, Any]] = {}
    for _, rr in gen.iterrows():
        pid = str(rr.get("plan_id", "")).strip()
        if not pid:
            continue
        gen_map[pid] = dict(rr)

    # POS success per style_text_ids (for C4NEWS filter + mismatch reuse)
    pos_plan_rows = plan[plan["neg_type"].astype(str) == "POS"].copy()
    pos_by_style_text_ids: Dict[str, List[str]] = {}
    all_pos_gen_q: List[str] = []

    for _, rr in pos_plan_rows.iterrows():
        pid = str(rr.get("plan_id", "")).strip()
        if not pid:
            continue

        style_key = str(rr.get("style_text_ids", "") or "")
        if pid not in gen_map:
            continue
        qt = str(gen_map[pid].get("query_text", "") or "").strip()
        if not qt:
            continue

        pos_by_style_text_ids.setdefault(style_key, []).append(qt)
        all_pos_gen_q.append(qt)

    # C4NEWS style filter: only keep styles with at least one successful POS generation
    c4_good_style_keys: Optional[set] = None
    try:
        c4_pos = plan[
            (plan["dataset"].astype(str).isin(["C4NEWS", "C4"]))
            & (plan["neg_type"].astype(str) == "POS")
        ].copy()
        good = set()
        for _, rr in c4_pos.iterrows():
            pid = str(rr.get("plan_id", "")).strip()
            if not pid:
                continue
            if pid in gen_map:
                qt = str(gen_map[pid].get("query_text", "") or "").strip()
                if qt:
                    good.add(str(rr.get("style_text_ids", "")))
        c4_good_style_keys = good
    except Exception:
        c4_good_style_keys = None

    out_rows: List[Dict[str, Any]] = []
    for _, r in plan.iterrows():
        plan_id = str(r.get("plan_id", "")).strip()
        if not plan_id:
            continue

        dataset = str(r.get("dataset", "") or "")
        neg_type = str(r.get("neg_type", "") or "")

        # Apply C4 style filter
        if c4_good_style_keys is not None and dataset in ("C4NEWS", "C4"):
            if neg_type == "NEG-Mismatch-S":
                pos_pid0 = str(r.get("pos_plan_id", "") or "").strip()
                pos_style_text_ids = ""
                try:
                    if pos_pid0:
                        pos_style_text_ids = str(
                            plan.loc[plan["plan_id"].astype(str) == pos_pid0, "true_style_text_ids"].head(1).tolist()[0]
                        )
                except Exception:
                    pos_style_text_ids = ""
                if pos_style_text_ids and pos_style_text_ids not in c4_good_style_keys:
                    continue
            else:
                s_key = str(r.get("style_text_ids", "") or "")
                if s_key not in c4_good_style_keys:
                    continue

        base: Dict[str, Any] = dict(r)

        # normalize list-like columns to json string list
        for k in [
            "style_text_ids",
            "semantic_text_ids",
            "wrong_style_text_ids",
            "true_style_text_ids",
            "true_semantic_text_ids",
        ]:
            if k in base:
                base[k] = dumps_list_str(_parse_list_str(base[k]))

        # Non-generation rows
        if neg_type == "NEG-Anchor":
            q = str(base.get("semantic_ref_text", "") or "")
            base["query_text"] = q
            base["generated_text"] = ""
            base["query_real_author_id"] = str(base.get("semantic_author_id", "") or "")
            try:
                ids = _parse_list_str(base.get("semantic_text_ids", ""))
                base["query_real_text_id"] = ids[0] if ids else ""
            except Exception:
                base["query_real_text_id"] = ""
            base["random_seed"] = None

        elif neg_type == "NEG-Mismatch-S":
            pos_pid = str(base.get("pos_plan_id", "") or "").strip()
            pos_gen = gen_map.get(pos_pid, None)
            if pos_gen is None:
                continue
            q = str(pos_gen.get("query_text", "") or "").strip()
            if not q:
                continue

            base["query_text"] = q
            base["generated_text"] = q
            base["query_real_author_id"] = str(pos_gen.get("query_real_author_id", "") or "")
            base["query_real_text_id"] = str(pos_gen.get("query_real_text_id", "") or "")
            base["random_seed"] = None

        else:
            # Generation rows: POS / NEG-neutral / NEG-wrong_ref_s
            if plan_id not in gen_map:
                continue
            g = gen_map[plan_id]
            base["query_text"] = str(g.get("query_text", "") or "")
            base["generated_text"] = str(g.get("generated_text", "") or "")
            base["random_seed"] = g.get("random_seed", None)
            base["generator_model"] = g.get("generator_model", base.get("generator_model", ""))
            base["prompt_version"] = g.get("prompt_version", base.get("prompt_version", ""))
            base["query_real_author_id"] = str(g.get("query_real_author_id", "") or base.get("query_real_author_id", ""))
            base["query_real_text_id"] = str(g.get("query_real_text_id", "") or base.get("query_real_text_id", ""))

        out_rows.append(base)

    out_df = pd.DataFrame(out_rows)

    for c in ["query_text", "generated_text", "style_ref_text", "semantic_ref_text"]:
        if c not in out_df.columns:
            out_df[c] = ""

    # -----------------------------
    # ✅ QC: adopt clean_samples QC logic
    # -----------------------------
    qc_df = out_df.copy()
    qc_df["qc_pass"] = 1
    qc_df["qc_reason"] = ""

    # 1) direct rule matches
    direct_flags: List[bool] = []
    direct_reasons: List[str] = []
    for txt in qc_df["query_text"].fillna("").astype(str).tolist():
        flag, reason = should_clean_query_text(txt)
        direct_flags.append(bool(flag))
        direct_reasons.append(str(reason or ""))

    m_direct = pd.Series(direct_flags, index=qc_df.index)
    qc_df.loc[m_direct, "qc_pass"] = 0
    qc_df.loc[m_direct, "qc_reason"] = pd.Series(direct_reasons, index=qc_df.index).where(m_direct, "")

    # 2) POS-triggered contiguous group deletion (adjacent block with same group_id)
    if "neg_type" in qc_df.columns:
        pos_bad_idx = qc_df.index[(qc_df["neg_type"].astype(str) == "POS") & (qc_df["qc_pass"].astype(int) == 0)].tolist()
        # iterate in output order positions
        for bad_row_idx in pos_bad_idx:
            try:
                pos_pos = qc_df.index.get_loc(bad_row_idx)  # integer position
            except Exception:
                continue
            start, end = contiguous_group_block_indices(qc_df, pos_pos, group_col="group_id")
            block = qc_df.iloc[start:end + 1]
            trigger_pid = str(qc_df.loc[bad_row_idx].get("plan_id", "") or "").strip()

            for j in block.index.tolist():
                if int(qc_df.loc[j, "qc_pass"]) == 1:
                    qc_df.loc[j, "qc_pass"] = 0
                    qc_df.loc[j, "qc_reason"] = f"POS contiguous group deletion triggered by {trigger_pid}"

    # write qc fields back
    out_df["qc_pass"] = qc_df["qc_pass"].astype(int)
    out_df["qc_reason"] = qc_df["qc_reason"].astype(str)

    out_df.to_csv(out_path, index=False)

    if qc_out_path:
        qc_df.to_csv(qc_out_path, index=False)

    return out_df
