# MSIBench

![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-green.svg)

MSIBench is a benchmark for Machine Style Imitation Verification (MSIV), a task that evaluate whether a query text is generated under stylistic conditioning on a given reference.

---

## Features

* **Flexible training modes**

  * Pairwise (contrastive)
  * Group-wise ranking

* **Evaluation**

  * Binary classification metrics
  * Ranking & attribution metrics
  * Neg-type breakdown

* **Dataset pipeline**

  * Cleaning → Sampling → Generation → QC → Filtering
---

## Installation

```bash
git clone <your-repo-url>
cd MSIBench

pip install -r requirements.txt
```

For dataset building:

```bash
cd datasets/Build
pip install -r requirements.txt
```

---

## Quick Start

### 1. Train

```bash
python scripts/main.py --mode train --config configs/default.yaml
```

### 2. Test

```bash
python scripts/main.py --mode test --run_dir outputs/<your_run>
```

or:

```bash
python scripts/main.py --mode test --checkpoint outputs/<your_run>/best
```

---

## Dataset

### Default Data Location

```
datasets/
  ├── C4NEWS/
  ├── CCAT50/
```

Key files:

* `samples_use.cleaned.csv` → pairwise samples
* `texts_dedup.cleaned.csv` → text pool

---

## Dataset Construction

Dataset pipeline is located in:

```
datasets/Build/
```

Run:

```bash
python data_processor.py --cfg configs/dataset_cfg.json
```

---

## Model Configuration

Main config:

```
configs/default.yaml
```

Key options:

* `model.method`: `bi_encoder` | `cross_encoder`
* `train_mode`: `pair` | `group`
* `pretrained_name`: HuggingFace model

---

## Project Structure

```
MSIBench/
├── configs/          # training configs
├── datasets/         # data + build pipeline
├── models/           # model definitions
├── scripts/          # entry point
├── src/              # training / evaluation logic
├── utils/            # utilities
```

---

## Notes

* LLM generation requires API configuration:

```bash
export OPENAI_API_KEY=your_key
```

* Default generator config:

```
datasets/Build/configs/generators.json
```

---

## License

MIT License
