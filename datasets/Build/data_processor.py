from __future__ import annotations

import argparse
import os
import pandas as pd

from src.load_and_clean import load_and_clean_all
from src.sample_pairs import sample_plan_all
from src.run_generation import run_generation
from src.qc_and_finalize import finalize_dataset
from src.test_generations_main import test_all_generators
from src.utils import ensure_dir, load_json

# 直接复用 filter.py 的 build_minimized（集成 filter.py 的工作流）
from src.filter import build_minimized


def _confirm_yes(msg: str) -> None:
    """Ask user to type y/yes to continue, otherwise abort."""
    try:
        ans = input(msg)
    except (EOFError, KeyboardInterrupt):
        print("\n[ABORT] No confirmation received. Exit.")
        raise SystemExit(1)
    if str(ans).strip().lower() not in ("y", "yes"):
        print("[ABORT] User declined. Exit.")
        raise SystemExit(1)


def _confirm_output_dir(out_dir: str) -> None:
    """
    如果 output_dir 已存在，要求用户确认后才继续执行。
    """
    if not os.path.exists(out_dir):
        return
    msg = (
        f"[WARN] output_dir already exists: {os.path.abspath(out_dir)}\n"
        f"Proceeding may overwrite/append files under this directory.\n"
        f"Type 'y' or 'yes' to continue: "
    )
    _confirm_yes(msg)


def _build_filter_version(input_csv: str, out_dir: str, tag: str, gzip_outputs: bool = False) -> None:
    """
    生成一版 filter 产物到 out_dir/filters/tag 子目录：
    - samples.csv
    - texts_dedup.csv
    - samples_use.csv
    - meta.json
    """
    vdir = os.path.join(out_dir, "filters", tag)
    ensure_dir(vdir)
    build_minimized(
        input_csv=input_csv,
        out_dir=vdir,
        gzip_outputs=gzip_outputs,
    )


def _print_preflight_report(report: dict) -> None:
    required = report.get("required", [])
    ok = set(report.get("ok", []))
    failed = report.get("failed", {})

    print("\n[Preflight] Generator endpoint test results:")
    print(f"  tested_count  = {len(report.get('tested', []))}")
    print(f"  ok_count      = {len(ok)}")
    print(f"  fail_count    = {len(failed)}")

    if required:
        print("\n  Required by current run:")
        for k in required:
            status = "OK" if k in ok else f"FAIL ({failed.get(k, 'not tested')})"
            print(f"   - {k}: {status}")

    if ok:
        print("\n  Usable generators:")
        print("   - " + ", ".join(sorted(list(ok))))

    if failed:
        print("\n  Failed generators (showing up to 20):")
        for i, (k, err) in enumerate(list(failed.items())[:20]):
            print(f"   - {k}: {err}")
        if len(failed) > 20:
            print(f"   ... ({len(failed) - 20} more)")


def main(cfg_path: str, datasets: list[str] | None = None) -> None:
    cfg = load_json(cfg_path)

    root_out_dir = cfg["output_dir"]  # e.g. "./outputs/20260109"
    _confirm_output_dir(root_out_dir)
    ensure_dir(root_out_dir)

    selected = None
    if datasets:
        selected = [d.strip() for d in datasets if d and d.strip()]
        if selected:
            unknown = [d for d in selected if d not in cfg["input_csvs"]]
            if unknown:
                raise ValueError(f"Datasets not found in config.input_csvs: {unknown}")

    datasets_to_run = selected if selected else list(cfg["input_csvs"].keys())
    if not datasets_to_run:
        raise ValueError("No datasets_to_run resolved. Check config.input_csvs.")

    # 只在整个程序开始时做一次 preflight（避免多数据集时反复卡住等待确认）
    gen_list = cfg.get("sampling", {}).get("generators", []) or []
    stub_plan_df = pd.DataFrame({"generator_model": [str(x) for x in gen_list if str(x).strip()]})

    report = test_all_generators(
        generators_cfg_path=cfg.get("generators_cfg_path", "configs/generators.json"),
        timeout_hint_s=90,
    )
    _print_preflight_report(report)
    _confirm_yes("\nType 'y' or 'yes' to continue with generation: ")

    for ds in datasets_to_run:
        out_dir = os.path.join(root_out_dir, str(ds))
        ensure_dir(out_dir)

        # 1) load & clean（只处理当前数据集）
        df_all, pool_all = load_and_clean_all(
            input_csvs=cfg["input_csvs"],
            min_tokens=int(cfg["cleaning"]["min_tokens"]),
            selected_datasets=[str(ds)],
            ccat50_dev_from_test=cfg.get(
                "ccat50_dev_from_test",
                {"seed": cfg.get("sampling", {}).get("seed", 20251218), "ratio": 0.5},
            ),
        )
        df_all.to_csv(os.path.join(out_dir, "cleaned_all.csv"), index=False)

        # 2) sample plan：一次性生成 POS + N1.N5 + MismatchS_*
        plan_path = os.path.join(out_dir, "plan.csv")
        plan_df = sample_plan_all(pool_all=pool_all, sampling_cfg=cfg["sampling"], prompt_dir="prompts")
        plan_df.to_csv(plan_path, index=False)

        # 3) run generation：只生成需要 LLM 的行（POS/N3/N4/N5）
        gen_out_path = os.path.join(out_dir, "generated_raw.csv")

        generated_df = run_generation(
            plan_df=plan_df,
            pool_all=pool_all,  # 用于“换语义参考文本”重试
            generators_cfg_path=cfg.get("generators_cfg_path", "configs/generators.json"),
            generation_cfg=cfg["generation"],
            out_path=gen_out_path,
        )

        # 4) qc & finalize：产出最终 pairwise_dataset.csv
        final_out_path = os.path.join(out_dir, "pairwise_dataset.csv")
        final_df = finalize_dataset(
            plan_df=plan_df,
            generated_df=generated_df,
            pool_all=pool_all,
            qc_cfg_path=cfg["qc"]["qc_cfg_path"],
            out_path=final_out_path,
        )

        # 5) filter v1：generated_raw
        _build_filter_version(
            input_csv=gen_out_path,
            out_dir=out_dir,
            tag="v1_generated_raw",
            gzip_outputs=bool(cfg.get("filter", {}).get("gzip_outputs", False)),
        )

        # 6) filter v2：pairwise_dataset
        _build_filter_version(
            input_csv=final_out_path,
            out_dir=out_dir,
            tag="v2_pairwise_dataset",
            gzip_outputs=bool(cfg.get("filter", {}).get("gzip_outputs", False)),
        )

        # 7) qc_pass 导出 + filter
        qc_pass_path = os.path.join(out_dir, "pairwise_dataset_qc_pass.csv")
        qc_pass_df = final_df[final_df["qc_pass"].astype(int) == 1].copy()
        qc_pass_df.to_csv(qc_pass_path, index=False)

        _build_filter_version(
            input_csv=qc_pass_path,
            out_dir=out_dir,
            tag="v2_qc_pass",
            gzip_outputs=bool(cfg.get("filter", {}).get("gzip_outputs", False)),
        )

        print(f"[OK] Dataset={ds} done. Outputs in: {out_dir}")
        print(" - cleaned_all.csv")
        print(" - plan.csv")
        print(" - generated_raw.csv")
        print(" - pairwise_dataset.csv")
        print(" - pairwise_dataset_qc_pass.csv")
        print(" - filters/v1_generated_raw/ (samples.csv, texts_dedup.csv, samples_use.csv, meta.json)")
        print(" - filters/v2_pairwise_dataset/ (samples.csv, texts_dedup.csv, samples_use.csv, meta.json)")
        print(" - filters/v2_qc_pass/ (samples.csv, texts_dedup.csv, samples_use.csv, meta.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to configs/dataset_cfg.json (or dataset_cfg_large.json)")
    ap.add_argument("--datasets", type=str, nargs="*", default=None, help="Optional subset of datasets to run")
    args = ap.parse_args()
    main(cfg_path=args.cfg, datasets=args.datasets)
