# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional

from src.run_generation import _build_generator
from src.utils import load_json


def _safe_preview(text: str, limit: int = 280) -> str:
    text = (text or "").strip().replace("\r", "")
    if len(text) <= limit:
        return text
    return text[:limit] + " ... [truncated]"


def _detect_models_in_cfg(gen_cfgs: Dict[str, Any]) -> Dict[str, str]:
    """用于打印更友好信息：key -> model"""
    out = {}
    for k, v in gen_cfgs.items():
        try:
            out[k] = str(v.get("model", ""))
        except Exception:
            out[k] = ""
    return out


def test_all_generators(
    generators_cfg_path: str = "generations.json",
    timeout_hint_s: Optional[int] = None,
) -> Dict[str, Any]:
    """
    遍历 generations.json 的所有 generator key，逐个调用一次 generate。
    返回一个汇总 dict（也会打印日志）。
    """
    gen_cfgs = load_json(generators_cfg_path)
    model_map = _detect_models_in_cfg(gen_cfgs)

    # 一条尽量兼容、且不会触发太大开销的 prompt
    prompt = (
        "You are a connectivity test. Reply with a single short sentence that includes:\n"
        "1) the word OK\n"
        "2) the model name if you know it\n"
        "3) a random-looking token: 7QZ\n"
        "No markdown."
    )

    # ✅ 新接口：只保留 seed
    seed = 123

    results: Dict[str, Any] = {
        "ok": [],
        "fail": [],
        "details": {},
    }

    print(f"\n[TEST] Loading generator configs from: {generators_cfg_path}")
    print(f"[TEST] Found {len(gen_cfgs)} generators: {list(gen_cfgs.keys())}\n")

    for key in gen_cfgs.keys():
        cfg = gen_cfgs[key]
        provider = cfg.get("provider")
        base_url = cfg.get("base_url")
        model = model_map.get(key, "")

        print("=" * 88)
        print(f"[TEST] Generator: {key}")
        print(f"  provider = {provider}")
        print(f"  model    = {model}")
        print(f"  base_url = {base_url}")

        # 1) 检查 api_key_env 是否存在
        api_key_env = cfg.get("api_key_env")
        api_key_present = bool(os.environ.get(str(api_key_env), "").strip()) if api_key_env else False
        print(f"  api_key_env = {api_key_env}  (present={api_key_present})")

        # 2) 构建 generator 并调用
        try:
            gen = _build_generator(gen_cfgs, key)

            # 可选：如果你想临时覆盖 timeout，可以在 OpenAICompatibleChatGenerator 上加一个可配字段
            if timeout_hint_s is not None and hasattr(gen, "timeout_s"):
                try:
                    setattr(gen, "timeout_s", int(timeout_hint_s))
                    print(f"  timeout_s overridden -> {getattr(gen, 'timeout_s')}")
                except Exception:
                    pass

            text = gen.generate(
                prompt=prompt,
                seed=seed,
            )

            preview = _safe_preview(text, limit=400)
            print("\n  [RESULT] SUCCESS")
            print("  [OUTPUT] " + preview.replace("\n", "\\n"))
            results["ok"].append(key)
            results["details"][key] = {
                "status": "success",
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "output_preview": preview,
            }

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            print("\n  [RESULT] FAILED")
            print("  [ERROR ] " + err_msg)

            tb = traceback.format_exc()
            print("  [TRACE ]\n" + tb)

            results["fail"].append(key)
            results["details"][key] = {
                "status": "failed",
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "error": err_msg,
                "traceback": tb,
            }

    print("\n" + "=" * 88)
    print("[TEST] SUMMARY")
    print(f"  success: {len(results['ok'])} -> {results['ok']}")
    print(f"  failed : {len(results['fail'])} -> {results['fail']}")

    return results


if __name__ == "__main__":
    cfg_path = os.environ.get("GENERATORS_CFG", "../configs/generators.json")
    timeout_hint = int(os.environ.get("GEN_TEST_TIMEOUT_S", "90"))

    summary = test_all_generators(
        generators_cfg_path=cfg_path,
        timeout_hint_s=timeout_hint,
    )

    if summary["fail"]:
        raise SystemExit(1)
