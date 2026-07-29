import os
import json
from datasets import load_dataset

# --- 1. Format Dataset in CL_Benchmark/SC/financial_phrasebank ---
data_dir = "CL_Benchmark/SC/financial_phrasebank"
os.makedirs(data_dir, exist_ok=True)

print("📥 Processing Financial PhraseBank into CL_Benchmark/SC/...")
phrasebank = load_dataset("takala/financial_phrasebank", "sentences_allagree")["train"]
label_map = {0: "negative", 1: "neutral", 2: "positive"}

formatted_data = []
for item in phrasebank:
    formatted_data.append({
        "sentence": item["sentence"],
        "label": label_map[item["label"]]
    })

# 80% Train / 10% Dev / 10% Test split
total = len(formatted_data)
train_end = int(total * 0.8)
dev_end = int(total * 0.9)

train_data = formatted_data[:train_end]
dev_data = formatted_data[train_end:dev_end]
test_data = formatted_data[dev_end:]

with open(f"{data_dir}/train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open(f"{data_dir}/dev.json", "w") as f:
    json.dump(dev_data, f, indent=2)

with open(f"{data_dir}/test.json", "w") as f:
    json.dump(test_data, f, indent=2)

# Create labels.json
labels = ["negative", "neutral", "positive"]
with open(f"{data_dir}/labels.json", "w") as f:
    json.dump(labels, f, indent=2)

print(f"✓ Saved {data_dir}/ (train: {len(train_data)}, dev: {len(dev_data)}, test: {len(test_data)}, labels.json)")

# --- 2. Create Task Configs in configs/order_minimal_configs/ ---
empty_dev_tasks = {
    "SC": [], "TC": [], "NLI": [], "QQP": [],
    "WiC": [], "MultiRC": [], "COPA": [], "BoolQA": []
}

config_base = "configs/order_minimal_configs"
os.makedirs(f"{config_base}/task1_dbpedia", exist_ok=True)
os.makedirs(f"{config_base}/task2_fpb", exist_ok=True)

# Task 1 (DBpedia) Configs
with open(f"{config_base}/task1_dbpedia/train_tasks.json", "w") as f:
    json.dump({"TC": [{"sampling strategy": "full", "dataset name": "dbpedia"}]}, f, indent=2)
with open(f"{config_base}/task1_dbpedia/dev_tasks.json", "w") as f:
    json.dump(empty_dev_tasks, f, indent=2)
with open(f"{config_base}/task1_dbpedia/test_tasks.json", "w") as f:
    json.dump({"TC": [{"sampling strategy": "full", "dataset name": "dbpedia"}]}, f, indent=2)

# Task 2 (FPB) Configs
with open(f"{config_base}/task2_fpb/train_tasks.json", "w") as f:
    json.dump({"SC": [{"sampling strategy": "full", "dataset name": "financial_phrasebank"}]}, f, indent=2)
with open(f"{config_base}/task2_fpb/dev_tasks.json", "w") as f:
    json.dump(empty_dev_tasks, f, indent=2)

# Task 2 Test Config (Evaluates BOTH DBpedia and FPB to measure forgetting)
with open(f"{config_base}/task2_fpb/test_tasks.json", "w") as f:
    json.dump({
        "TC": [{"sampling strategy": "full", "dataset name": "dbpedia"}],
        "SC": [{"sampling strategy": "full", "dataset name": "financial_phrasebank"}]
    }, f, indent=2)

print(f"✓ Task configuration files generated under {config_base}/")