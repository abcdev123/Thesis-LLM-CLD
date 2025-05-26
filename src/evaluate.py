#!/usr/bin/env python3
import os
import torch
import json
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    logging as hf_logging,
)
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ─── SILENCE TRANSFORMERS PADDING WARNINGS ───────────────────────────────────────
hf_logging.set_verbosity_error()

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
BASE_MODEL        = "mistralai/Mistral-7B-Instruct-v0.2" 
FINETUNED_MODEL   = "src/Mistral_LLM_7B_Instruct-v0.2_lora_finetuned/merged_fp16" # path to your merged_fp16 folder
# DATA_PATH         = "src/Dataset_Gijs_prompts.xlsx"
OUTPUT_DIR        = "Evaluation_results_26-05-2025_lora"         # where to write all results
# TOKENIZED_TEST_DIR = os.path.join(OUTPUT_DIR, "tokenized_test")
TOKENIZED_TEST_DIR = "src/Mistral_LLM_7B_Instruct-v0.2_lora_finetuned/tokenized_test" # path to the tokenized test dataset
MAX_NEW_TOKENS    = 10                                      # how many tokens to generate
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Running inference on device: {DEVICE}")
print(f"Base model:    {BASE_MODEL}")
print(f"Fine-tuned:    {FINETUNED_MODEL}")
print(f"Test tokens:   {TOKENIZED_TEST_DIR}")
print(f"Results in:    {OUTPUT_DIR}")
print(f"Max new tokens per example: {MAX_NEW_TOKENS}")

# ─── EVALUATION FUNCTION ────────────────────────────────────────────────────────
def evaluate_model(model_dir, tokenizer, test_dataset):
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    preds, trues, records = [], [], []
    for ex in test_dataset:
        input_ids      = torch.tensor(ex["input_ids"]).unsqueeze(0).to(DEVICE)
        attention_mask = torch.tensor(ex["attention_mask"]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
            )

        # decode the entire completion
        raw_pred = tokenizer.decode(
            out[0][ input_ids.shape[-1]: ].tolist(),
            skip_special_tokens=True
        ).strip()

        # extract just the Relationship field from JSON
        try:
            rel = json.loads(raw_pred).get("Relationship", "")
        except json.JSONDecodeError:
            rel = raw_pred  # fallback if not valid JSON

        pred_relation = rel.strip().lower()

        # decode the true label, dropping any -100 entries and special tokens
        label_ids = ex.get("labels", [])
        valid_label_ids = [
            t for t in label_ids
            if t >= 0 and t not in tokenizer.all_special_ids
        ]
        true_text = tokenizer.decode(
            valid_label_ids,
            skip_special_tokens=True
        ).strip().lower()

        preds.append(pred_relation)
        trues.append(true_text)
        records.append({
            "prompt":  ex.get("prompt", ""),
            "true":    true_text,
            "pred":    pred_relation,
            "correct": pred_relation == true_text
        })

    return trues, preds, records

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.padding_side='left'  # ensure correct padding for decoder-only models
    test_ds = load_from_disk(TOKENIZED_TEST_DIR)

    for name, model_path in [("base", BASE_MODEL), ("lora", FINETUNED_MODEL)]:
        print(f"\n>> Evaluating `{name}` model ({model_path})")
        trues, preds, records = evaluate_model(model_path, tokenizer, test_ds)

        # compute metrics on just the Relationship label
        acc    = accuracy_score(trues, preds)
        prec   = precision_score(trues, preds, average="weighted", zero_division=0)
        rec    = recall_score(trues, preds, average="weighted", zero_division=0)
        f1     = f1_score(trues, preds, average="weighted", zero_division=0)
        report = classification_report(trues, preds, digits=4, zero_division=0)
        # cm     = confusion_matrix(trues, preds, labels=["positive", "negative", "none"])

        # only keep labels that actually appear in the true set
        # all_labels    = ["positive", "negative", "none"]
        # present_labels = [lbl for lbl in all_labels if lbl in trues]
        # cm = confusion_matrix(trues, preds, labels=present_labels)

        # print summary
        print(f"  Accuracy : {acc*100:6.2f}%")
        print(f"  Precision: {prec*100:6.2f}% (weighted)")
        print(f"  Recall   : {rec*100:6.2f}% (weighted)")
        print(f"  F1-score : {f1*100:6.2f}% (weighted)\n")
        print("  Classification report:")
        print(report)
        print("  Confusion matrix (rows=true, cols=pred | order: positive, negative, none):")
        # print(cm)

        # write detailed Excel
        excel_path = os.path.join(OUTPUT_DIR, f"{name}_results.xlsx")
        pd.DataFrame.from_records(records).to_excel(excel_path, index=False)
        print(f"  → Detailed per-example results saved to: {excel_path}")

if __name__ == "__main__":
    main()

