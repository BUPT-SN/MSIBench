# scripts/main.py
from __future__ import annotations

import json
import os

from src.trainer import train_and_eval
from src.evaluator import evaluate_checkpoint
from utils.config import parse_args, resolve_config_for_train, resolve_config_for_test
from utils.utils import make_run_group_name


def _resolve_ckpt(args) -> str:
    if args.checkpoint:
        return args.checkpoint
    if args.run_dir:
        cand = os.path.join(args.run_dir, "best")
        if os.path.isdir(cand):
            return cand
        raise FileNotFoundError(f"--run_dir provided but best dir not found: {cand}")
    raise ValueError("For --mode test, you must provide either --checkpoint or --run_dir.")


def main():
    args = parse_args()
    cfg = resolve_config_for_train(args)
    run_name = make_run_group_name(cfg)
    output_root = str(cfg.get("output_dir", "outputs"))
    run_dir = os.path.join(output_root, run_name)

    if args.mode == "train":

        summary = train_and_eval(cfg)
        print("\n=== TRAIN SUMMARY ===")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        # 保存 train summary 到文件

        with open(os.path.join(run_dir, "train_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return

    # mode == test
    ckpt_dir = _resolve_ckpt(args)
    split = args.split or str((cfg.get("data", {}) or {}).get("test_split", "test"))
    report = evaluate_checkpoint(cfg, ckpt_dir, split=split)
    os.makedirs(run_dir, exist_ok=True)
    print("\n=== TEST REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # 保存 test report 到文件
    with open(os.path.join(run_dir, "test_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
