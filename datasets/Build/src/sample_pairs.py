from __future__ import annotations

from typing import Any, Dict, List, Optional
import json

import numpy as np
import pandas as pd

from prompts import get_prompt_template, PROMPT_TABLE
from .utils import new_sample_id


def _flatten_split_pool(pool_all: Dict[str, Any], dataset: str, split: str) -> List[dict]:
    """
    pool_all[dataset][split][author_id] -> list[rows]
    Flatten into a single list, ignoring author dimension.
    """
    out: List[dict] = []
    ds_map = pool_all.get(dataset, {}) if isinstance(pool_all, dict) else {}
    sp_map = ds_map.get(split, {}) if isinstance(ds_map, dict) else {}
    if not isinstance(sp_map, dict):
        return out

    for _, rows in sp_map.items():
        if not rows:
            continue
        out.extend(list(rows))
    return out


def _truncate(s: str, max_chars: int) -> str:
    s = "" if s is None else str(s)
    if max_chars and max_chars > 0 and len(s) > int(max_chars):
        return s[: int(max_chars)]
    return s


def _prompt_versions_by_family(dataset: str, family: str) -> List[str]:
    """
    family in {"imitate","neutral"}.
    Uses PROMPT_TABLE[dataset] keys.
    """
    table = PROMPT_TABLE.get(str(dataset), {})
    if not isinstance(table, dict):
        return []
    fam = str(family).strip().lower()
    ks: List[str] = []
    for k in table.keys():
        kk = str(k).strip()
        if not kk:
            continue
        if kk.lower().startswith(fam + "_"):
            ks.append(kk)
    return ks


def _safe_choice(rng: np.random.Generator, xs: List[Any]) -> Any:
    if not xs:
        return None
    return xs[int(rng.integers(0, len(xs)))]


def _choose_distinct_rows(
    rng: np.random.Generator,
    rows: List[dict],
    exclude_text_ids: Optional[set] = None,
    k: int = 1,
) -> List[dict]:
    if not rows or k <= 0:
        return []
    exclude_text_ids = exclude_text_ids or set()
    cands = [r for r in rows if str(r.get("text_id", "")) not in exclude_text_ids]
    if not cands:
        return []
    if k >= len(cands):
        rng.shuffle(cands)
        return cands
    idxs = rng.choice(np.arange(len(cands)), size=k, replace=False).tolist()
    return [cands[i] for i in idxs]

def sample_plan_all(
    pool_all: Dict[str, Any],
    sampling_cfg: Dict[str, Any],
    selected_datasets: Optional[List[str]] = None,
    prompt_dir: Optional[str] = None,  # ✅ compat with data_processor.py
) -> pd.DataFrame:
    """
    Build sampling plan under the NEW spec:

    1) sample n style refs s (per dataset+split)
    2) for each s, build m groups:
        - each group samples 1 anchor a
        - outputs:
            POS (LLM, conditioned_on_s = s_true)
            NEG-Anchor (q = a)
            NEG-neutral (LLM, conditioned_on_s = none)
            NEG-wrong_ref_s (LLM, conditioned_on_s = wrong_s, but input pair uses s_true)
            NEG-Mismatch-S x K (no generation): replace input s with s_wrong, keep q_pos

    Returns a DataFrame that will be saved to plan.csv
    """
    sampling_cfg = sampling_cfg or {}
    seed = int(sampling_cfg.get("seed", 20251218))
    rng = np.random.default_rng(seed)

    # per-split n
    n_style_refs_cfg = sampling_cfg.get("n_style_refs", {})
    if isinstance(n_style_refs_cfg, dict):
        n_style_refs_by_split = {str(k): int(v) for k, v in n_style_refs_cfg.items()}
    else:
        # fallback single int
        n0 = int(n_style_refs_cfg or 200)
        n_style_refs_by_split = {"train": n0, "dev": max(1, n0 // 2), "test": max(1, n0 // 2)}

    m_groups_per_s = int(sampling_cfg.get("m_groups_per_s", 3))
    k_mismatch_s = int(sampling_cfg.get("k_mismatch_s", 2))
    allow_anchor_same_as_style = bool(sampling_cfg.get("allow_anchor_same_as_style", False))
    prompt_truncate_chars = int(sampling_cfg.get("prompt_truncate_chars", 6000))
    generator_keys = sampling_cfg.get("generators", []) or []

    prompt_family_by_type = sampling_cfg.get("prompt_family_by_type", {}) or {}

    def _family_for(sample_type: str) -> str:
        return str(prompt_family_by_type.get(sample_type, "imitate")).strip().lower()

    datasets = list(pool_all.keys()) if isinstance(pool_all, dict) else []
    if selected_datasets:
        datasets = [d for d in datasets if d in selected_datasets]

    plan_rows: List[Dict[str, Any]] = []
    s_id_counter = 0
    group_id_counter = 0
    plan_id_counter = 0

    def _new_pid() -> str:
        nonlocal plan_id_counter
        plan_id_counter += 1
        return new_sample_id(plan_id_counter)

    for dataset in datasets:
        ds = str(dataset)

        imitate_versions = _prompt_versions_by_family(ds, "imitate")
        neutral_versions = _prompt_versions_by_family(ds, "neutral")

        ds_splits = []
        try:
            ds_splits = sorted(list(pool_all.get(ds, {}).keys()))
        except Exception:
            ds_splits = []

        for split in ds_splits:
            sp = str(split)
            texts = _flatten_split_pool(pool_all, ds, sp)
            if len(texts) < 2:
                continue

            n_s = int(n_style_refs_by_split.get(sp, n_style_refs_by_split.get("train", 200)))
            n_s = min(n_s, len(texts))
            if n_s <= 0:
                continue

            style_idxs = rng.choice(np.arange(len(texts)), size=n_s, replace=False).tolist()

            for style_idx in style_idxs:
                s_row = texts[style_idx]
                s_text_id = str(s_row.get("text_id", ""))
                s_text = _truncate(str(s_row.get("text", "") or ""), prompt_truncate_chars)
                s_author = str(s_row.get("author_id", "") or "")

                s_id = s_id_counter
                s_id_counter += 1

                # candidate anchors
                if allow_anchor_same_as_style:
                    anchor_candidates = texts
                else:
                    anchor_candidates = [x for x in texts if str(x.get("text_id", "")) != s_text_id]

                # ✅ NEW: CCAT50 约束：anchor 必须来自“其他作者”
                if ds == "CCAT50":
                    anchor_candidates = [
                        x for x in anchor_candidates
                        if str(x.get("author_id", "") or "") != s_author
                    ]

                if not anchor_candidates:
                    continue

                # style candidates for wrong_s
                wrong_style_candidates = [x for x in texts if str(x.get("text_id", "")) != s_text_id]
                if not wrong_style_candidates:
                    continue

                for _ in range(m_groups_per_s):
                    group_id = group_id_counter
                    group_id_counter += 1

                    # anchor a
                    a_row = _safe_choice(rng, anchor_candidates)
                    if a_row is None:
                        continue
                    a_text_id = str(a_row.get("text_id", ""))
                    a_text = _truncate(str(a_row.get("text", "") or ""), prompt_truncate_chars)
                    a_author = str(a_row.get("author_id", "") or "")

                    # one wrong_s for NEG-wrong_ref_s
                    wrong_s_row = _safe_choice(rng, wrong_style_candidates)
                    if wrong_s_row is None:
                        continue
                    wrong_s_text_id = str(wrong_s_row.get("text_id", ""))
                    wrong_s_text = _truncate(str(wrong_s_row.get("text", "") or ""), prompt_truncate_chars)
                    wrong_s_author = str(wrong_s_row.get("author_id", "") or "")

                    # ---------- POS (generate, imitate)
                    pos_pid = _new_pid()
                    pos_prompt_ver = _safe_choice(rng, imitate_versions) or (imitate_versions[0] if imitate_versions else "")
                    pos_gen = _safe_choice(rng, list(generator_keys)) or (generator_keys[0] if generator_keys else "mock")
                    pos_template = get_prompt_template(dataset=ds, prompt_version=pos_prompt_ver) if pos_prompt_ver else ""
                    pos_prompt = ""
                    if pos_template:
                        pos_prompt = pos_template.format(
                            style_ref_text=s_text,
                            semantic_ref_text=a_text,
                        )

                    plan_rows.append(
                        {
                            "plan_id": pos_pid,
                            "dataset": ds,
                            "split": sp,
                            "neg_type": "POS",
                            "label": 1,

                            "s_id": s_id,
                            "group_id": group_id,
                            "is_mismatch_s": 0,

                            "style_ref_text": s_text,
                            "semantic_ref_text": a_text,

                            "style_author_id": s_author,
                            "style_text_ids": json.dumps([s_text_id]),
                            "semantic_author_id": a_author,
                            "semantic_text_ids": json.dumps([a_text_id]),

                            "true_style_author_id": s_author,
                            "true_style_text_ids": json.dumps([s_text_id]),
                            "true_semantic_author_id": a_author,
                            "true_semantic_text_ids": json.dumps([a_text_id]),

                            "generator_model": pos_gen,
                            "prompt_version": pos_prompt_ver,
                            "final_prompt": pos_prompt,

                            "wrong_style_author_id": "",
                            "wrong_style_text_ids": json.dumps([]),
                            "wrong_style_ref_text": "",
                            "pos_plan_id": "",
                        }
                    )

                    # ---------- NEG-Anchor (no gen, q=a)
                    neg_anchor_pid = _new_pid()
                    plan_rows.append(
                        {
                            "plan_id": neg_anchor_pid,
                            "dataset": ds,
                            "split": sp,
                            "neg_type": "NEG-Anchor",
                            "label": 0,

                            "s_id": s_id,
                            "group_id": group_id,
                            "is_mismatch_s": 0,

                            "style_ref_text": s_text,
                            "semantic_ref_text": a_text,

                            "style_author_id": s_author,
                            "style_text_ids": json.dumps([s_text_id]),
                            "semantic_author_id": a_author,
                            "semantic_text_ids": json.dumps([a_text_id]),

                            "true_style_author_id": s_author,
                            "true_style_text_ids": json.dumps([s_text_id]),
                            "true_semantic_author_id": a_author,
                            "true_semantic_text_ids": json.dumps([a_text_id]),

                            "generator_model": "",
                            "prompt_version": "",
                            "final_prompt": "",

                            "wrong_style_author_id": "",
                            "wrong_style_text_ids": json.dumps([]),
                            "wrong_style_ref_text": "",
                            "pos_plan_id": "",
                        }
                    )

                    # ---------- NEG-neutral (generate, neutral)
                    neg_neu_pid = _new_pid()
                    neu_prompt_ver = _safe_choice(rng, neutral_versions) or (neutral_versions[0] if neutral_versions else "")
                    neu_gen = _safe_choice(rng, list(generator_keys)) or (generator_keys[0] if generator_keys else "mock")
                    neu_template = get_prompt_template(dataset=ds, prompt_version=neu_prompt_ver) if neu_prompt_ver else ""
                    neu_prompt = ""
                    if neu_template:
                        neu_prompt = neu_template.format(
                            semantic_ref_text=a_text,
                        )

                    plan_rows.append(
                        {
                            "plan_id": neg_neu_pid,
                            "dataset": ds,
                            "split": sp,
                            "neg_type": "NEG-neutral",
                            "label": 0,

                            "s_id": s_id,
                            "group_id": group_id,
                            "is_mismatch_s": 0,

                            # pair input uses s_true
                            "style_ref_text": s_text,
                            "semantic_ref_text": a_text,

                            "style_author_id": s_author,
                            "style_text_ids": json.dumps([s_text_id]),
                            "semantic_author_id": a_author,
                            "semantic_text_ids": json.dumps([a_text_id]),

                            "true_style_author_id": s_author,
                            "true_style_text_ids": json.dumps([s_text_id]),
                            "true_semantic_author_id": a_author,
                            "true_semantic_text_ids": json.dumps([a_text_id]),

                            "generator_model": neu_gen,
                            "prompt_version": neu_prompt_ver,
                            "final_prompt": neu_prompt,

                            "wrong_style_author_id": "",
                            "wrong_style_text_ids": json.dumps([]),
                            "wrong_style_ref_text": "",
                            "pos_plan_id": "",
                        }
                    )

                    # ---------- NEG-wrong_ref_s (generate imitate using wrong_s, but input pair uses s_true)
                    neg_ws_pid = _new_pid()
                    ws_prompt_ver = _safe_choice(rng, imitate_versions) or (imitate_versions[0] if imitate_versions else "")
                    ws_gen = _safe_choice(rng, list(generator_keys)) or (generator_keys[0] if generator_keys else "mock")
                    ws_template = get_prompt_template(dataset=ds, prompt_version=ws_prompt_ver) if ws_prompt_ver else ""
                    ws_prompt = ""
                    if ws_template:
                        ws_prompt = ws_template.format(
                            style_ref_text=wrong_s_text,
                            semantic_ref_text=a_text,
                        )

                    plan_rows.append(
                        {
                            "plan_id": neg_ws_pid,
                            "dataset": ds,
                            "split": sp,
                            "neg_type": "NEG-wrong_ref_s",
                            "label": 0,

                            "s_id": s_id,
                            "group_id": group_id,
                            "is_mismatch_s": 0,

                            # input pair uses s_true
                            "style_ref_text": s_text,
                            "semantic_ref_text": a_text,

                            "style_author_id": s_author,
                            "style_text_ids": json.dumps([s_text_id]),
                            "semantic_author_id": a_author,
                            "semantic_text_ids": json.dumps([a_text_id]),

                            "true_style_author_id": s_author,
                            "true_style_text_ids": json.dumps([s_text_id]),
                            "true_semantic_author_id": a_author,
                            "true_semantic_text_ids": json.dumps([a_text_id]),

                            "generator_model": ws_gen,
                            "prompt_version": ws_prompt_ver,
                            "final_prompt": ws_prompt,

                            "wrong_style_author_id": wrong_s_author,
                            "wrong_style_text_ids": json.dumps([wrong_s_text_id]),
                            "wrong_style_ref_text": wrong_s_text,
                            "pos_plan_id": "",
                        }
                    )

                    # ---------- NEG-Mismatch-S x K (no gen): replace input s with s_wrong, keep q_pos
                    mismatch_rows = _choose_distinct_rows(
                        rng,
                        wrong_style_candidates,
                        exclude_text_ids=set([s_text_id]),
                        k=k_mismatch_s,
                    )

                    for sw in mismatch_rows:
                        sw_text_id = str(sw.get("text_id", ""))
                        sw_text = _truncate(str(sw.get("text", "") or ""), prompt_truncate_chars)
                        sw_author = str(sw.get("author_id", "") or "")

                        mm_pid = _new_pid()
                        plan_rows.append(
                            {
                                "plan_id": mm_pid,
                                "dataset": ds,
                                "split": sp,
                                "neg_type": "NEG-Mismatch-S",
                                "label": 0,

                                "s_id": s_id,
                                "group_id": group_id,
                                "is_mismatch_s": 1,

                                # input pair uses s_wrong
                                "style_ref_text": sw_text,
                                "semantic_ref_text": a_text,

                                "style_author_id": sw_author,
                                "style_text_ids": json.dumps([sw_text_id]),
                                "semantic_author_id": a_author,
                                "semantic_text_ids": json.dumps([a_text_id]),

                                # true s/a (that produced POS)
                                "true_style_author_id": s_author,
                                "true_style_text_ids": json.dumps([s_text_id]),
                                "true_semantic_author_id": a_author,
                                "true_semantic_text_ids": json.dumps([a_text_id]),

                                "generator_model": "",
                                "prompt_version": "",
                                "final_prompt": "",

                                "wrong_style_author_id": "",
                                "wrong_style_text_ids": json.dumps([]),
                                "wrong_style_ref_text": "",

                                "pos_plan_id": pos_pid,
                            }
                        )

    df = pd.DataFrame(plan_rows)

    required_cols = [
        "plan_id", "dataset", "split", "neg_type", "label",
        "s_id", "group_id", "is_mismatch_s", "pos_plan_id",
        "style_ref_text", "semantic_ref_text",
        "style_author_id", "style_text_ids",
        "semantic_author_id", "semantic_text_ids",
        "wrong_style_author_id", "wrong_style_text_ids", "wrong_style_ref_text",
        "true_style_author_id", "true_style_text_ids",
        "true_semantic_author_id", "true_semantic_text_ids",
        "generator_model", "prompt_version", "final_prompt",
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    df = df[required_cols].copy()
    return df
