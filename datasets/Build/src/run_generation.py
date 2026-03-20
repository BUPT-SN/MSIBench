from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from prompts import PROMPT_TABLE, get_prompt_template
from .utils import ensure_dir, load_json, stable_gen_id


BEGIN_MARK = "[BEGIN]"
END_MARK = "[END]"

FORMAT_CONTROL_SUFFIX = f"""
You MUST reply with the requested output ONLY between the following markers.

{BEGIN_MARK}
(put your new article here, Do NOT include anything outside this markers.)
{END_MARK}

""".strip()


def _already_has_control_suffix(prompt: str) -> bool:
    p = prompt or ""
    return (BEGIN_MARK in p) and (END_MARK in p)


def _append_format_control(prompt: str) -> str:
    p = (prompt or "").rstrip()
    if not p:
        return FORMAT_CONTROL_SUFFIX
    if _already_has_control_suffix(p):
        return p
    return p + "\n\n" + FORMAT_CONTROL_SUFFIX + "\n"


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    if s.startswith("```") and s.endswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()
def extract_generated_text(raw: str) -> str:
    """
    从文本末尾向前找 END_MARK，再向前找 BEGIN_MARK，避免将思考过程中的标志识别为输出边界。

    新规则（按你的要求）：
    1) 找不到“成对 MARK” => 视为生成失败（抛异常），进入重试逻辑（不再退化为整段文本）
    2) 若只在开头/末尾残留 MARK，则剥离边界 MARK
    3) 剥离后若提取文本仍包含 BEGIN_MARK / END_MARK（出现在正文任意位置）=> 失败（抛异常）
    """
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty_raw_generation")

    # ---- 1) 优先：从末尾向前找“成对 MARK”
    end_pos = s.rfind(END_MARK)
    if end_pos != -1:
        begin_pos = s.rfind(BEGIN_MARK, 0, end_pos)
        if begin_pos != -1:
            inner = s[begin_pos + len(BEGIN_MARK): end_pos]
            inner = _strip_code_fences(inner).strip()

            # 边界处残留 MARK 则剥离（仅剥离开头/末尾）
            if inner.startswith(BEGIN_MARK):
                inner = inner[len(BEGIN_MARK):].lstrip()
            if inner.endswith(END_MARK):
                inner = inner[: -len(END_MARK)].rstrip()

            inner = inner.strip()
            if not inner:
                raise ValueError("empty_extracted_between_marks")

            # 剥离后正文仍含 MARK（任意位置）=> 失败
            if (BEGIN_MARK in inner) or (END_MARK in inner):
                raise ValueError("markers_still_present_after_strip")

            return inner

    # ---- 2) 没找到成对 MARK：只做“边界剥离”，但最终仍判失败以触发重试
    tmp = s
    stripped_any = False

    if tmp.startswith(BEGIN_MARK):
        tmp = tmp[len(BEGIN_MARK):].lstrip()
        stripped_any = True
    if tmp.endswith(END_MARK):
        tmp = tmp[: -len(END_MARK)].rstrip()
        stripped_any = True

    # 如果剥离后仍含 MARK，说明 MARK 混入正文/中间 => 更明确失败原因
    if (BEGIN_MARK in tmp) or (END_MARK in tmp):
        raise ValueError("unpaired_or_misaligned_markers_present")

    # 即便只是在边界残留且被剥离了，但因为“没成对 MARK”，仍视为失败触发重试
    raise ValueError("missing_paired_marks")


@dataclass
class SkipSample(Exception):
    code: str
    message: str


class BaseGenerator:
    def generate(self, prompt: str, seed: Optional[int] = None) -> str:
        raise NotImplementedError


class MockGenerator(BaseGenerator):
    def generate(self, prompt: str, seed: Optional[int] = None) -> str:
        rnd = random.Random(seed or 0)
        out = f"mock_{rnd.randint(1, 999999)}"
        return f"{BEGIN_MARK}\n{out}\n{END_MARK}"


class OpenAICompatibleChatGenerator(BaseGenerator):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: int = 90,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.extra_headers = extra_headers or {}

    def generate(self, prompt: str, seed: Optional[int] = None) -> str:
        url = self.base_url + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        if seed is not None:
            payload["seed"] = int(seed)

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data, ensure_ascii=False)


def _build_generator(generators_cfg: Dict[str, Any], generator_key: str, use_mock: bool = False) -> BaseGenerator:
    if use_mock or generator_key == "mock":
        return MockGenerator()

    if generator_key not in generators_cfg:
        raise KeyError(f"Generator key not found in generators_cfg: {generator_key}")

    cfg = generators_cfg[generator_key]
    typ = str(cfg.get("type", "openai_compatible")).lower().strip()
    if typ in ("openai_compatible", "openai-compatible", "openai"):
        api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        if not api_key:
            raise RuntimeError(f"Missing API key env: {cfg.get('api_key_env', 'OPENAI_API_KEY')}")
        base_url = str(cfg.get("base_url", "https://api.openai.com/v1"))
        model = str(cfg.get("model", "gpt-4o-mini"))
        timeout_s = int(cfg.get("timeout_s", 90))
        extra_headers = cfg.get("extra_headers", None)
        return OpenAICompatibleChatGenerator(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            extra_headers=extra_headers,
        )

    raise ValueError(f"Unsupported generator type={typ} for key={generator_key}")


def _collect_prompt_versions_for_dataset(dataset: str) -> List[str]:
    ds = str(dataset or "").strip()
    table = PROMPT_TABLE.get(ds, {})
    if isinstance(table, dict):
        return [str(k) for k in table.keys() if str(k).strip()]
    return []


def _build_pool_by_author(pool_all: Any) -> Dict[str, Dict[str, List[dict]]]:
    out: Dict[str, Dict[str, List[dict]]] = {}
    if not isinstance(pool_all, dict):
        return out

    for dataset, splits_map in pool_all.items():
        ds = str(dataset)
        if not isinstance(splits_map, dict):
            continue
        for _, authors_map in splits_map.items():
            if not isinstance(authors_map, dict):
                continue
            for author_id, rows in authors_map.items():
                aid = str(author_id)
                if not rows:
                    continue
                out.setdefault(ds, {}).setdefault(aid, []).extend(list(rows))
    return out


def run_generation(
    plan_df: pd.DataFrame,
    pool_all: Any,
    generators_cfg_path: str,
    out_path: str,
    generation_cfg: Dict[str, Any],
) -> pd.DataFrame:
    ensure_dir(os.path.dirname(out_path))

    use_mock = bool(generation_cfg.get("use_mock", False))
    max_workers = int(generation_cfg.get("max_workers", 1))
    max_retries = int(generation_cfg.get("per_sample_retries", generation_cfg.get("max_retries", 2)))
    retry_backoff_s = float(generation_cfg.get("retry_backoff_s", 1.0))
    base_seed = int(generation_cfg.get("seed", 12345))

    generators_cfg = load_json(generators_cfg_path)

    available_models = list(plan_df.get("generator_model", pd.Series([], dtype=str)).astype(str).unique())
    available_models = [m for m in available_models if m and m != "nan"]
    seen = set()
    available_models_uniq = []
    for m in available_models:
        if m not in seen:
            seen.add(m)
            available_models_uniq.append(m)

    done_plan_ids = set()
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            old = pd.read_csv(out_path, dtype=str, keep_default_na=False)
            for _, rr in old.iterrows():
                pid = str(rr.get("plan_id", "")).strip()
                if pid:
                    done_plan_ids.add(pid)
        except Exception:
            pass

    todo = plan_df.reset_index(drop=False).rename(columns={"index": "plan_row_index"}).copy()
    if done_plan_ids:
        todo = todo[~todo["plan_id"].astype(str).isin(done_plan_ids)].reset_index(drop=True)

    header_written = os.path.exists(out_path) and os.path.getsize(out_path) > 0

    import threading
    lock = threading.Lock()

    def safe_append(row: Dict[str, Any]) -> None:
        nonlocal header_written
        with lock:
            df1 = pd.DataFrame([row])
            df1.to_csv(out_path, mode="a", index=False, header=not header_written)
            header_written = True

    def _log_skip(idx: int, plan_id: str, code: str, msg: str) -> None:
        if str(generation_cfg.get("verbose", "0")).lower() in ("1", "true", "yes", "y"):
            print(f"[SKIP] idx={idx} plan_id={plan_id} code={code} msg={msg}")

    pool_by_author = _build_pool_by_author(pool_all)

    def _pick_semantic_retry_text(dataset: str, author_id: Any, exclude_text_ids: Any, seed_i: int) -> Optional[dict]:
        ds = str(dataset or "")
        aid = str(author_id or "")
        if not ds or not aid:
            return None

        texts = pool_by_author.get(ds, {}).get(aid, [])
        if not texts:
            return None

        exclude_set = set()
        try:
            if exclude_text_ids is None:
                exclude_set = set()
            elif isinstance(exclude_text_ids, (list, tuple, set)):
                exclude_set = set([str(x) for x in exclude_text_ids])
            else:
                ss = str(exclude_text_ids).strip()
                if ss:
                    try:
                        v = json.loads(ss)
                        if isinstance(v, list):
                            exclude_set = set([str(x) for x in v])
                        else:
                            exclude_set = set([ss])
                    except Exception:
                        exclude_set = set([x.strip() for x in ss.split(",") if x.strip()])
        except Exception:
            exclude_set = set()

        cands = [t for t in texts if str(t.get("text_id", "")) not in exclude_set]
        if not cands:
            cands = list(texts)
        if not cands:
            return None

        rnd = random.Random(int(seed_i))
        return cands[int(rnd.randrange(0, len(cands)))]

    def process_one(idx: int, r: Any) -> Optional[Dict[str, Any]]:
        plan_id = str(r.get("plan_id", "") or "").strip()
        dataset = str(r.get("dataset", ""))
        split = str(r.get("split", ""))
        neg_type = str(r.get("neg_type", ""))
        label = int(r.get("label", 0))

        # ✅ New design: skip non-generation types
        if neg_type in ("NEG-Anchor", "NEG-Mismatch-S"):
            return None

        generator_key0 = str(r.get("generator_model", "mock") or "mock")
        prompt_version0 = str(r.get("prompt_version", "") or "").strip()
        prompt0 = str(r.get("final_prompt", "") or "")

        style_ref_text0 = str(r.get("style_ref_text", "") or "")
        semantic_ref_text0 = str(r.get("semantic_ref_text", "") or "")
        wrong_style_ref_text0 = r.get("wrong_style_ref_text", None)

        semantic_text_ids0 = r.get("semantic_text_ids", None)
        semantic_author_id0 = r.get("semantic_author_id", None)

        def _attempt(
            generator_key: str,
            prompt_version: str,
            prompt_text: str,
            seed_i: int,
            semantic_ref_text: str = semantic_ref_text0,
            semantic_text_ids: Any = semantic_text_ids0,
            semantic_author_id: Any = semantic_author_id0,
        ) -> Dict[str, Any]:
            gen = _build_generator(generators_cfg, generator_key, use_mock=use_mock)
            prompt_text = _append_format_control(prompt_text)

            raw = gen.generate(prompt_text, seed=seed_i)
            query_text = extract_generated_text(raw)
            query_text = (query_text or "").strip()
            if not query_text:
                raise ValueError("empty_query_text_after_extraction")

            query_text_id = stable_gen_id(query_text)

            out_row = {
                "plan_id": plan_id,

                "dataset": dataset,
                "split": split,
                "neg_type": neg_type,
                "label": label,

                "style_author_id": r.get("style_author_id", None),
                "style_text_ids": r.get("style_text_ids", None),
                "semantic_author_id": semantic_author_id,
                "semantic_text_ids": semantic_text_ids,
                "wrong_style_author_id": r.get("wrong_style_author_id", None),
                "wrong_style_text_ids": r.get("wrong_style_text_ids", None),

                "style_ref_text": r.get("style_ref_text", None),
                "semantic_ref_text": semantic_ref_text,
                "wrong_style_ref_text": r.get("wrong_style_ref_text", None),

                "final_prompt": prompt_text,

                "query_real_author_id": f"LLM::{generator_key}",
                "query_real_text_id": query_text_id,
                "generated_text": query_text,
                "query_text": query_text,

                "generator_model": generator_key,
                "prompt_version": prompt_version,

                "random_seed": int(seed_i),
            }

            safe_append(out_row)
            return out_row

        plan_row_index = int(r.get("plan_row_index", idx) or idx)
        seed0 = base_seed + plan_row_index * 10007

        # 0) base attempt
        try:
            return _attempt(
                generator_key=generator_key0,
                prompt_version=prompt_version0,
                prompt_text=prompt0,
                seed_i=seed0,
            )
        except SkipSample as e:
            _log_skip(idx, plan_id, e.code, e.message)
            return None
        except Exception as e:
            last_err = e

        # 1) retry: same model + prompt fallback
        available_prompts = _collect_prompt_versions_for_dataset(dataset)

        for t in range(max_retries):
            try:
                time.sleep(retry_backoff_s * (t + 1))

                if prompt_version0 and prompt_version0 in set(available_prompts):
                    prompt_version = prompt_version0
                else:
                    prompt_version = random.choice(available_prompts) if available_prompts else (prompt_version0 or "")

                if prompt_version and prompt_version in set(available_prompts):
                    # ✅ bugfix: correct kw is prompt_version (NOT version)
                    template = get_prompt_template(dataset=dataset, prompt_version=prompt_version)
                    try:
                        prompt_text = template.format(
                            style_ref_text=style_ref_text0,
                            semantic_ref_text=semantic_ref_text0,
                            wrong_style_ref_text=wrong_style_ref_text0,
                            dataset=dataset,
                            split=split,
                            neg_type=neg_type,
                            label=label,
                        )
                    except Exception:
                        prompt_text = prompt0 or template
                else:
                    prompt_text = prompt0

                return _attempt(
                    generator_key=generator_key0,
                    prompt_version=prompt_version,
                    prompt_text=prompt_text,
                    seed_i=seed0 + (t + 1),
                )
            except SkipSample as e:
                _log_skip(idx, plan_id, e.code, e.message)
                return None
            except Exception as e:
                last_err = e

        # 2) retry: switch model
        other_models = [generator_key0] + [m for m in available_models_uniq if m != generator_key0]
        for m in other_models:
            for t in range(max_retries):
                try:
                    time.sleep(retry_backoff_s * (t + 1))
                    return _attempt(
                        generator_key=m,
                        prompt_version=prompt_version0 or "",
                        prompt_text=prompt0,
                        seed_i=seed0 + 1000 + (t + 1),
                    )
                except SkipSample as e:
                    _log_skip(idx, plan_id, e.code, e.message)
                    return None
                except Exception as e:
                    last_err = e

        # 3) semantic retry: swap semantic ref within same author (legacy behavior)
        if semantic_author_id0 is not None:
            cand = _pick_semantic_retry_text(
                dataset=dataset,
                author_id=semantic_author_id0,
                exclude_text_ids=semantic_text_ids0,
                seed_i=seed0,
            )
            if cand is not None:
                try:
                    sem_text = str(cand.get("text", "") or "")
                    sem_id = cand.get("text_id", None)
                    return _attempt(
                        generator_key=generator_key0,
                        prompt_version=prompt_version0 or "",
                        prompt_text=prompt0,
                        seed_i=seed0 + 2000,
                        semantic_ref_text=sem_text,
                        semantic_text_ids=sem_id,
                        semantic_author_id=semantic_author_id0,
                    )
                except SkipSample as e:
                    _log_skip(idx, plan_id, e.code, e.message)
                    return None
                except Exception as e:
                    last_err = e

        _log_skip(idx, plan_id, "generation_failed", f"{type(last_err).__name__}: {last_err}")
        return None

    if max_workers <= 1:
        for i in tqdm(range(len(todo)), desc="gen"):
            r = todo.iloc[i]
            _ = process_one(i, r)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {}
            for i in range(len(todo)):
                r = todo.iloc[i]
                futs[ex.submit(process_one, i, r)] = i

            for fut in tqdm(as_completed(list(futs.keys())), total=len(futs), desc="gen"):
                try:
                    _ = fut.result()
                except Exception:
                    pass

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            return pd.read_csv(out_path, dtype=str, keep_default_na=False)
        except Exception:
            return pd.DataFrame([])
    return pd.DataFrame([])
