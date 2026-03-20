# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Any, List, Union, Optional

from prompts.news import PROMPTS_NEWS

# PromptGrid = List[List[str]]  (defined in prompts/news.py)
PromptValue = Union[str, List[List[str]]]

# dataset -> {prompt_version -> template(str or PromptGrid)}
PROMPT_TABLE: Dict[str, Dict[str, PromptValue]] = {
    # ✅ 统一新闻类 prompts：CCAT50 / C4NEWS 共用
    "CCAT50": PROMPTS_NEWS,
    "C4NEWS": PROMPTS_NEWS,
}


def _grid_to_text(grid: List[List[str]]) -> str:
    """
    Convert PromptGrid (List[List[str]]) into a multi-line string template.
    Each inner list is a list of characters for one line.
    """
    if not grid:
        return ""
    lines: List[str] = []
    for row in grid:
        if row is None:
            lines.append("")
        else:
            lines.append("".join([str(ch) for ch in row]))
    return "\n".join(lines)


def get_prompt_template(dataset: str, prompt_version: str) -> str:
    ds = str(dataset or "").strip()
    pv = str(prompt_version or "").strip()

    table = PROMPT_TABLE.get(ds, None)
    if not table:
        # fallback: try any table
        for _, t in PROMPT_TABLE.items():
            if pv in t:
                v = t[pv]
                if isinstance(v, str):
                    return v
                return _grid_to_text(v)
        raise KeyError(f"Unknown dataset={ds} and prompt_version={pv}")

    if pv not in table:
        raise KeyError(f"Prompt version not found for dataset={ds}: {pv}")

    v = table[pv]
    if isinstance(v, str):
        return v
    return _grid_to_text(v)
