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

# ─── SILENCE WARNINGS ─────────────────────────────────────────────────────────────
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
    try:
        return json.loads(json_str)["Relationship"].strip().lower()
    except:
        return json_str.strip().lower()

# ─── EVALUATION FUNCTION ─────────────────────────────────────────────────────────
def evaluate_model(model_name: str, tokenizer, test_ds: Dataset):
    # load model and ensure it knows the pad token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    preds, trues, records = [], [], []
    for ex in test_ds:
        prompt          = ex["prompt"]
        true_completion = ex["completion"]
        true_rel        = parse_relationship(true_completion)

        # tokenize just the prompt (needs pad_token defined)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )[0]

        # grab only the newly generated tokens
        appended = gen_ids[enc["input_ids"].shape[-1] :].tolist()
        raw_out  = tokenizer.decode(appended, skip_special_tokens=True).strip()
        pred_rel = parse_relationship(raw_out)

        preds.append(pred_rel)
        trues.append(true_rel)
        records.append({
            "prompt":    prompt,
            "true_json": true_completion,
            "pred_json": raw_out,
            "true_rel":  true_rel,
            "pred_rel":  pred_rel,
            "correct":   pred_rel == true_rel,
        })

    return trues, preds, records

# ─── MAIN ─────────────────────────────────────────────────────────────────────────
def main():
    # 1) load raw data & make the same test split
    df = pd.read_excel(DATA_PATH)
    ds = Dataset.from_pandas(df[["prompt", "completion"]]).shuffle(seed=42)
    test_ds = ds.train_test_split(test_size=0.2, seed=42)["test"]

    # 2) prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) evaluate both models
    for label, model_name in [("base", BASE_MODEL), ("lora", FINETUNED_MODEL)]:
        print(f"\n>> Evaluating {label} ({model_name})")
        trues, preds, records = evaluate_model(model_name, tokenizer, test_ds)

        # compute metrics
        acc    = accuracy_score(trues, preds)
        prec   = precision_score(trues, preds, average="weighted", zero_division=0)
        rec    = recall_score(trues, preds, average="weighted", zero_division=0)
        f1     = f1_score(trues, preds, average="weighted", zero_division=0)
        report_dict = classification_report(
            trues, preds, digits=4, zero_division=0, output_dict=True
        )
        present = sorted(set(trues))
        cm = confusion_matrix(trues, preds, labels=present)

        # build DataFrames
        df_records = pd.DataFrame.from_records(records)
        df_metrics = pd.DataFrame([{
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "n_samples": len(trues),
        }])
        df_report = pd.DataFrame(report_dict).transpose()
        df_cm     = pd.DataFrame(cm, index=present, columns=present)

        # write all sheets to one Excel file
        out_path = os.path.join(OUTPUT_DIR, f"{label}_results.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            df_records.to_excel(writer, sheet_name="per_example", index=False)
            df_metrics.to_excel(writer, sheet_name="metrics", index=False)
            df_report.to_excel(writer, sheet_name="report")
            df_cm.to_excel(writer, sheet_name="confusion")
        print("  → Results saved to:", out_path)

if __name__ == "__main__":
    main()



