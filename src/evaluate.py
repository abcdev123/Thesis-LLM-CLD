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

# ─── Silence unwanted warnings ──────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*right-padding was detected.*")
hf_logging.set_verbosity_error()

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_MODEL      = "mistralai/Mistral-7B-Instruct-v0.2"
FINETUNED_MODEL = "src/Mistral_LLM_7B_Instruct-v0.2_lora_finetuned/merged_fp16"
DATA_PATH       = "src/Dataset_Gijs_prompts.xlsx"
OUTPUT_DIR      = "Evaluation_results_26-05-2025_lora"
MAX_NEW_TOKENS  = 10
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Running on device: {DEVICE}")

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def parse_relationship(json_str: str) -> str:
    """
    Given a string like 
      {"Relationship":"positive","Explanation":"…"}
    return the lowercased Relationship value, or the raw fallback.
    """
    try:
        return json.loads(json_str)["Relationship"].strip().lower()
    except:
        return json_str.strip().lower()

# ─── EVALUATION ─────────────────────────────────────────────────────────────────
def evaluate_model(model_name, tokenizer, test_ds):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    preds, trues, records = [], [], []
    for ex in test_ds:
        prompt = ex["prompt"]
        true_completion = ex["completion"]
        true_rel = parse_relationship(true_completion)

        # tokenize only the prompt for generation
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
                pad_token_id=tokenizer.eos_token_id,
            )[0]

        # decode only what the model appended
        appended = gen_ids[ enc["input_ids"].shape[-1] : ].tolist()
        raw_out = tokenizer.decode(appended, skip_special_tokens=True).strip()
        pred_rel = parse_relationship(raw_out)

        preds.append(pred_rel)
        trues.append(true_rel)
        records.append({
            "prompt":     prompt,
            "true_json":  true_completion,
            "pred_json":  raw_out,
            "true_rel":   true_rel,
            "pred_rel":   pred_rel,
            "correct":    pred_rel == true_rel,
        })

    return trues, preds, records

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    # 1) load raw data & make same test split
    df = pd.read_excel(DATA_PATH)
    ds = Dataset.from_pandas(df[["prompt","completion"]]).shuffle(seed=42)
    split = ds.train_test_split(test_size=0.2, seed=42)
    test_ds = split["test"]

    # 2) prepare shared tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.padding_side = "left"

    # 3) evaluate both models
    for label, model_name in [("base", BASE_MODEL), ("lora", FINETUNED_MODEL)]:
        print(f"\n>> Evaluating {label} ({model_name})")
        trues, preds, records = evaluate_model(model_name, tokenizer, test_ds)

        # metrics
        acc    = accuracy_score(trues, preds)
        prec   = precision_score(trues, preds, average="weighted", zero_division=0)
        rec    = recall_score(trues, preds, average="weighted", zero_division=0)
        f1     = f1_score(trues, preds, average="weighted", zero_division=0)
        report = classification_report(trues, preds, digits=4, zero_division=0)

        present = sorted(set(trues))
        cm = confusion_matrix(trues, preds, labels=present)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1       : {f1:.4f}")
        print("Classification Report:\n", report)
        print("Confusion Matrix (labels =", present, ")\n", cm)

        # dump per‐example sheet
        out_path = os.path.join(OUTPUT_DIR, f"{label}_results.xlsx")
        pd.DataFrame(records).to_excel(out_path, index=False)
        print("Results →", out_path)

if __name__ == "__main__":
    main()


