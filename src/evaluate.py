#!/usr/bin/env python3
import os
import json
import torch
import warnings
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    logging as hf_logging,
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ─── QUIET WARNINGS ──────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*right-padding was detected.*")
hf_logging.set_verbosity_error()

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
BASE_MODEL      = "mistralai/Mistral-7B-Instruct-v0.2"
FINETUNED_MODEL = "src/Mistral_LLM_7B_Instruct-v0.2_lora_finetuned/merged_fp16"
DATA_PATH       = "src/Dataset_Gijs_prompts.xlsx"
OUTPUT_DIR      = "Evaluation_results_26-05-2025_lora"
MAX_NEW_TOKENS  = 10
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Running on device: {DEVICE}")

# ─── HELPERS ─────────────────────────────────────────────────────────────────────
def parse_relationship(json_str: str) -> str:
    """
    Extracts the 'Relationship' (case-insensitive) from the JSON string,
    or falls back to the raw string if parsing fails.
    """
    try:
        obj = json.loads(json_str)
        # case-insensitive key lookup
        return (obj.get("Relationship") or obj.get("relationship") or "").strip().lower()
    except:
        return json_str.strip().lower()

# ─── EVALUATION FUNCTION ─────────────────────────────────────────────────────────
def evaluate_model(model_name: str, tokenizer, test_ds: Dataset, debug: bool = False):
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    # ensure pad_token is set on model
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    preds, trues, records = [], [], []
    for idx, ex in enumerate(test_ds):
        prompt          = ex["prompt"]
        true_completion = ex["completion"]
        true_rel        = parse_relationship(true_completion)

        # tokenize only the prompt (no padding)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=False
        ).to(DEVICE)

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )[0]

        # slice off exactly the prompt length
        prompt_len = enc["input_ids"].shape[-1]
        new_tokens = gen_ids[prompt_len:].tolist()
        raw_out    = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred_rel   = parse_relationship(raw_out)

        # debug first few LoRA outputs
        if debug and idx < 5:
            print(f"[DEBUG] example {idx} raw model output: {raw_out!r}")

        preds.append(pred_rel)
        trues.append(true_rel)
        records.append({
            "prompt":    prompt,
            "true_json": true_completion,
            "pred_json": raw_out,
            "true_rel":  true_rel,
            "pred_rel":  pred_rel,
            "correct":   (pred_rel == true_rel),
        })

    return trues, preds, records

# ─── MAIN ─────────────────────────────────────────────────────────────────────────
def main():
    # 1) load raw data & reproduce the 80/20 split
    df = pd.read_excel(DATA_PATH)
    ds = Dataset.from_pandas(df[["prompt", "completion"]]).shuffle(seed=42)
    test_ds = ds.train_test_split(test_size=0.2, seed=42)["test"]

    # 2) tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) evaluate both models
    for label, model_name in [("base", BASE_MODEL), ("lora", FINETUNED_MODEL)]:
        print(f"\n>> Evaluating {label} model")
        # enable debug printing for LoRA
        trues, preds, records = evaluate_model(model_name, tokenizer,
                                               test_ds,
                                               debug=(label=="lora"))

        # ─── METRICS ───────────────────────────────────────────────────────────
        acc  = accuracy_score(trues, preds)
        prec = precision_score(trues, preds, average="weighted", zero_division=0)
        rec  = recall_score(trues, preds, average="weighted", zero_division=0)
        f1   = f1_score(trues, preds, average="weighted", zero_division=0)
        report_dict = classification_report(
            trues, preds, digits=4, zero_division=0, output_dict=True
        )
        labels = sorted(set(trues))
        cm = confusion_matrix(trues, preds, labels=labels)

        # ─── BUILD DATAFRAMES & WRITE EXCEL ──────────────────────────────────
        df_records = pd.DataFrame.from_records(records)
        df_metrics = pd.DataFrame([{
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "n_samples": len(trues),
        }])
        df_report = pd.DataFrame(report_dict).transpose()
        df_cm     = pd.DataFrame(cm, index=labels, columns=labels)

        out_path = os.path.join(OUTPUT_DIR, f"{label}_results.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            df_records.to_excel(writer, sheet_name="per_example", index=False)
            df_metrics.to_excel(writer, sheet_name="metrics", index=False)
            df_report.to_excel(writer, sheet_name="report")
            df_cm.to_excel(writer, sheet_name="confusion")
        print(f"  → Results saved to: {out_path}")

if __name__ == "__main__":
    main()




