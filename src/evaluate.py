#!/usr/bin/env python3
import os
import re
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

# ─── QUIET WARNINGS ─────────────────────────────────────────────────────────────
# warnings.filterwarnings("ignore", message=".*right-padding was detected.*")
# hf_logging.set_verbosity_error()

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
# BASE_MODEL    = "mistralai/Mistral-7B-Instruct-v0.2"
BASE_MODEL      = "Qwen/Qwen2.5-14B-Instruct-1M"
# BASE_MODEL      = "mistralai/Mistral-7B-v0.1"  # Base Mistral
# FINETUNED_MODEL = "src/Mistral_LLM_7B_v0.1_Base_lora_finetuned/merged_fp16_7Bv0.1"  # Path to your fine-tuned model
FINETUNED_MODEL = "src/Qwen2.5-14B-Instruct_lora_finetuned_w_wrapping/merged_fp16"
DATA_PATH       = "src/Dataset_Gijs_prompts.xlsx"
OUTPUT_DIR      = "Evaluation_results_Qwen2.5-14B-Instruct-02-06-2025"
SEQ_LEN         = 1300
MAX_NEW_TOKENS  = 150
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Running inference on {DEVICE}")

# ─── HELPERS ─────────────────────────────────────────────────────────────────────
def parse_relationship_from_output(text: str) -> str:
    """
    Regex out exactly the value behind "relationship":"...".
    Returns 'positive', 'negative', or 'none' (lowercased), or '' if no match.
    """
    m = re.search(r'[Rr]elationship"\s*:\s*"([^"]+)"', text)
    return m.group(1).strip().lower() if m else ""

# ─── EVALUATION FUNCTION ─────────────────────────────────────────────────────────
def evaluate_model(model_name: str, tokenizer, test_ds: Dataset, debug: bool = False):
    print(f"\nLoading model from {model_name}…")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    trues, preds, records = [], [], []
    for idx, ex in enumerate(test_ds):
        prompt          = ex["prompt"]
        true_completion = ex["completion"]
        true_rel        = parse_relationship_from_output(true_completion)

        # ─── wrap in [INST] tags ────────────────────────────────────────────
        wrapped = f"<s>[INST] {prompt} [/INST]"

        # wrapper = (
        #     "### Instruction:\n"
        #     f"{prompt}\n"
        #     "### Response:\n"
        # )

        encodings = tokenizer(
            wrapped,
            truncation=True,
            max_length=SEQ_LEN,
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(DEVICE)

        # ─── generate exactly MAX_NEW_TOKENS (disable early EOS) ────────────
        with torch.no_grad():
            ids = model.generate(
                **encodings,
                max_new_tokens=MAX_NEW_TOKENS,
                # eos_token_id=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )[0]

        # ─── slice off prompt tokens to get only the model’s output ────────
        prompt_len  = encodings["input_ids"].shape[-1]
        new_tokens  = ids[prompt_len:].tolist()
        raw_output  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred_rel    = parse_relationship_from_output(raw_output)

        if debug and idx < 5:
            print(f"[DEBUG] #{idx} raw_output={raw_output!r} → pred_rel={pred_rel!r}")

        trues.append(true_rel)
        preds.append(pred_rel)
        records.append({
            "prompt":   prompt,
            "true":     true_rel,
            "output":   raw_output,
            "pred_rel": pred_rel,
            "correct":  (pred_rel == true_rel),
        })

    return trues, preds, records

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    # load & split
    df    = pd.read_excel(DATA_PATH)
    ds    = Dataset.from_pandas(df[["prompt","completion"]]).shuffle(seed=42)
    test_dataset  = ds.train_test_split(test_size=0.2, seed=42)["test"]

    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL, use_fast=True)
    # tokenizer.padding_side = "left"
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # run eval for both
    for label, model_name in [("base", BASE_MODEL), ("lora", FINETUNED_MODEL)]:
        print(f"\n>> Evaluating {label} model")
        trues, preds, records = evaluate_model(
            model_name, tokenizer, test_dataset, debug=(label=="lora")
        )

        # compute metrics
        acc         = accuracy_score(trues, preds)
        prec        = precision_score(trues, preds, average="weighted", zero_division=0)
        rec         = recall_score(trues, preds, average="weighted", zero_division=0)
        f1          = f1_score(trues, preds, average="weighted", zero_division=0)
        report_dict = classification_report(
            trues, preds, digits=4, zero_division=0, output_dict=True
        )
        labels      = sorted(l for l in ["positive","negative","none"] if l in trues)
        cm          = confusion_matrix(trues, preds, labels=labels)

        # build & write Excel
        df_rec   = pd.DataFrame.from_records(records)
        df_met   = pd.DataFrame([{
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "n_samples": len(trues),
        }])
        df_rep   = pd.DataFrame(report_dict).transpose()
        df_cm    = pd.DataFrame(cm, index=labels, columns=labels)

        out_path = os.path.join(OUTPUT_DIR, f"{label}_results.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            df_rec.to_excel(writer, sheet_name="per_example", index=False)
            df_met.to_excel(writer, sheet_name="metrics",      index=False)
            df_rep.to_excel(writer, sheet_name="report")
            df_cm.to_excel(writer, sheet_name="confusion")
        print("  → Saved results to", out_path)

if __name__ == "__main__":
    main()






