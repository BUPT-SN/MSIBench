from __future__ import annotations

import hashlib
import json
import os
import random
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def simple_tokenize(text: str) -> List[str]:
    # 轻量 token：按空白与标点切分
    text = text.strip()
    if not text:
        return []
    return [t for t in re.split(r"\s+|(?u)[^\w]+", text) if t]


def token_len(text: str) -> int:
    return len(simple_tokenize(text))


def join_texts(texts: List[str]) -> str:
    return "\n---\n".join([t.strip() for t in texts if t and t.strip()])


def dumps_list_str(xs: List[str]) -> str:
    return json.dumps([str(x) for x in xs], ensure_ascii=False)


def stable_gen_id(text: str) -> str:
    h = hashlib.sha1()
    h.update((text or "").encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def new_sample_id(i: int) -> str:
    return f"S{int(i):09d}"


def has_markdown(text: str) -> bool:
    """
    ✅ 修正点 5：检测常见 Markdown 形式（要求输出不能包含 markdown）
    这里采用“宁可误杀一点，也不放过明显 markdown”的策略。
    """
    s = (text or "")
    if not s.strip():
        return False

    # code fence / inline code
    if "```" in s or "`" in s:
        return True

    # markdown link / image
    if re.search(r"!\[[^\]]*\]\([^)]+\)", s):
        return True
    if re.search(r"\[[^\]]+\]\([^)]+\)", s):
        return True

    # heading
    if re.search(r"(?m)^\s{0,3}#{1,6}\s+\S+", s):
        return True

    # list bullets / numbered list
    if re.search(r"(?m)^\s*[-*+]\s+\S+", s):
        return True
    if re.search(r"(?m)^\s*\d+\.\s+\S+", s):
        return True

    # blockquote
    if re.search(r"(?m)^\s*>\s+\S+", s):
        return True

    # table pipes (rough)
    if re.search(r"(?m)^\s*\|.*\|\s*$", s):
        return True

    return False
