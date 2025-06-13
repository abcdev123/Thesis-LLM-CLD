#!/usr/bin/env python3
import os
import re
import time
import pandas as pd
from openai import OpenAI, RateLimitError
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from dotenv import load_dotenv
from tqdm.auto import tqdm

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

MODEL          = "deepseek-reasoner"
DATA_PATH      = "Dataset_Gijs_prompts.xlsx"
OUTPUT_DIR     = "Evaluation_results_DeepSeek-Reasoner"
SAMPLE_N       = None  # set to int for smoke test, None for full run
MAX_TOKENS     = 10000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── PARSING FUNCTION ────────────────────────────────────────────────────────────
def parse_relationship_from_output(text: str) -> str:
    m = re.search(r'[Rr]elationship"\s*:\s*"([^"]+)"', text)
    return m.group(1).strip().lower() if m else ""

# ─── EVALUATION FUNCTION ─────────────────────────────────────────────────────────
def evaluate_deepseek(test_ds: Dataset, debug: bool = False):
    trues, preds, records = [], [], []

    for idx, ex in enumerate(tqdm(test_ds, desc="Evaluating DeepSeek")):
        prompt = ex["prompt"]
        true_completion = ex["completion"]
        true_rel = parse_relationship_from_output(true_completion)

        # ─── BUILD PROMPT FOR DEEPSEEK API ──────────────────────────────────────
        inline_prompt = f"<s>[INST] {prompt} [/INST]"

        # ─── API CALL ───────────────────────────────────────────────────────────
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": inline_prompt}],
                max_tokens=MAX_TOKENS,
            )
        except RateLimitError:
            time.sleep(10)
            continue

        raw_output = resp.choices[0].message.content.strip()
        pred_rel = parse_relationship_from_output(raw_output)

        if debug and idx < 5:
            print(f"[DEBUG] #{idx} raw_output={raw_output!r} → pred_rel={pred_rel!r}")

        trues.append(true_rel)
        preds.append(pred_rel)
        records.append({
            "prompt": prompt,
            "true": true_rel,
            "output": raw_output,
            "pred_rel": pred_rel,
            "correct": (pred_rel == true_rel),
        })

    return trues, preds, records

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    # Load & split dataset
    df = pd.read_excel(DATA_PATH)
    ds = Dataset.from_pandas(df[["prompt", "completion"]]).shuffle(seed=42)
    test_ds = ds.train_test_split(test_size=0.2, seed=42)["test"]

    if SAMPLE_N is not None:
        test_ds = test_ds.select(range(SAMPLE_N))

    # Evaluate DeepSeek
    trues, preds, records = evaluate_deepseek(test_ds, debug=True)

    # Compute metrics
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, average="weighted", zero_division=0)
    rec = recall_score(trues, preds, average="weighted", zero_division=0)
    f1 = f1_score(trues, preds, average="weighted", zero_division=0)
    report_dict = classification_report(
        trues, preds, digits=4, zero_division=0, output_dict=True
    )
    labels = sorted(l for l in ["positive", "negative", "none"] if l in trues)
    cm = confusion_matrix(trues, preds, labels=labels)

    # Save results
    df_rec = pd.DataFrame.from_records(records)
    df_met = pd.DataFrame([{
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_samples": len(trues),
    }])
    df_rep = pd.DataFrame(report_dict).transpose()
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    out_path = os.path.join(OUTPUT_DIR, "deepseek_results.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        df_rec.to_excel(writer, sheet_name="per_example", index=False)
        df_met.to_excel(writer, sheet_name="metrics", index=False)
        df_rep.to_excel(writer, sheet_name="report")
        df_cm.to_excel(writer, sheet_name="confusion")
    print("✅ Results saved to", out_path)

if __name__ == "__main__":
    main()
