import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader

# Ensure project root is in path
sys.path.append(os.getcwd())

from utils.checkpoint import load_model_checkpoint
from utils.dataloader import load_split_dataset, make_pair_collator


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine_distance(a, b):
    """
    Compute Cosine Distance: 1 - CosineSimilarity
    Range: [0, 2] (0 is identical, 1 is orthogonal, 2 is opposite)
    """
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0  # Max distance if zero vector
    sim = np.dot(a, b) / (norm_a * norm_b)
    return 1.0 - sim


def extract_distance_data(cfg, checkpoint_dir, split="test", device="cuda", max_samples=5000):
    """
    提取成对的距离数据，返回 DataFrame 便于绘图。
    Columns: [Distance, Category]
    """
    print(f"Loading model from {checkpoint_dir}...")
    wrapper = load_model_checkpoint(cfg, checkpoint_dir, device=device)
    wrapper.eval()

    print(f"Loading dataset split: {split}...")
    ds = load_split_dataset(cfg, split)

    max_len = getattr(wrapper, "max_length", 256)
    collate_fn = make_pair_collator("bi_encoder", wrapper.tokenizer, max_len, cfg=cfg)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    records = []

    print("Computing embeddings & distances...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            if max_samples and batch_idx * 64 > max_samples:
                break

            # Encode
            s_emb = wrapper.model.encode_text(
                batch["style_input_ids"].to(device),
                batch["style_attention_mask"].to(device)
            ).cpu().numpy()
            q_emb = wrapper.model.encode_text(
                batch["query_input_ids"].to(device),
                batch["query_attention_mask"].to(device)
            ).cpu().numpy()

            labels = batch["labels"].cpu().numpy()
            neg_types = batch.get("neg_type", [""] * len(labels))

            for i in range(len(labels)):
                dist = cosine_distance(s_emb[i], q_emb[i])
                label = int(labels[i])
                nt = str(neg_types[i])

                # 定义类别名称（用于内部分组）
                category = "Other"
                if label == 1:
                    category = "Positive Imitation\n($q_{pos}$)"
                elif "Anchor" in nt:
                    category = "Semantic Anchor\n($q_{anchor}$)"
                elif "neutral" in nt:
                    category = "Neutral Gen\n($q_{sem}$)"
                elif "wrong_ref" in nt:
                    category = "Wrong Reference"
                elif "Mismatch" in nt:
                    category = "Style Mismatch\n($q_{wrong}$)"

                # 只保留我们关心的几类
                if category != "Other":
                    records.append({"Distance": dist, "Category": category})

    df = pd.DataFrame(records)
    return df


def plot_boxplot(df, output_dir):
    # 3) 全图宽度缩短为原本的 2/3 左右：原(8,6) -> 约(5.3,6)
    plt.figure(figsize=(5.3, 6))

    # 设置风格
    sns.set_theme(style="whitegrid")

    desired_order = [
        "Positive Imitation\n($q_{pos}$)",
        "Semantic Anchor\n($q_{anchor}$)",
        "Neutral Gen\n($q_{sem}$)",
        "Style Mismatch\n($q_{wrong}$)"
    ]
    existing_cats = set(df["Category"].unique())
    final_order = [cat for cat in desired_order if cat in existing_cats]

    ax = sns.boxplot(
        data=df,
        x="Category",
        y="Distance",
        order=final_order,
        palette="Set2",
        width=0.5,
        linewidth=1.5,
        showfliers=False
    )

    # 标题与轴标签（保留你原来的语义）
    plt.title("C4: Bi_encoder + BiGR", fontsize=18, fontweight="bold")
    plt.ylabel("Cosine Distance to Style Reference ($s$)\n(Lower is Better)", fontsize=18)
    plt.xlabel("")
    plt.ylim(0, 2.0)

    # 1) x轴只显示 q_x：把刻度文本替换成只保留数学符号部分
    tick_label_map = {
        "Positive Imitation\n($q_{pos}$)": r"$q_{pos}$",
        "Semantic Anchor\n($q_{anchor}$)": r"$q_{anchor}$",
        "Neutral Gen\n($q_{sem}$)": r"$q_{sem}$",
        "Style Mismatch\n($q_{wrong}$)": r"$q_{wrong}$",
    }
    new_xticklabels = [tick_label_map.get(cat, cat) for cat in final_order]
    ax.set_xticklabels(new_xticklabels)

    # 2) x、y轴标尺字体大小加大加粗
    tick_fontsize = 16
    for tick in ax.get_xticklabels():
        tick.set_fontsize(tick_fontsize)
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontsize(tick_fontsize)
        tick.set_fontweight("bold")

    # 4) 适当标注数值：在每个 box 上方标注中位数（median）
    #    位置：该类的 x 坐标上，y=median + 小偏移；避免压到箱体
    medians = df.groupby("Category")["Distance"].median()
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_offset = 0.03 * y_range

    for x_pos, cat in enumerate(final_order):
        if cat not in medians.index:
            continue
        m = float(medians.loc[cat])
        ax.text(
            x_pos,
            m + y_offset,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "distance_boxplot_c4_group.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="visual_results_box")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    set_seed(42)
    cfg = load_config(args.config)
    if "model" not in cfg:
        cfg["model"] = {}
    cfg["model"]["method"] = "bi_encoder"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = extract_distance_data(cfg, args.checkpoint, args.split, device, args.max_samples)

    if df.empty:
        print("No data extracted! Check split name or dataset path.")
        return

    print(f"Extracted {len(df)} samples.")
    print(df.groupby("Category")["Distance"].describe())  # 打印统计数据，方便写论文

    plot_boxplot(df, args.output_dir)


if __name__ == "__main__":
    main()
