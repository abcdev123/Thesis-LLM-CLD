#!/usr/bin/env python3
import os
import re
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
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

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
BASE_MODEL      = "mistralai/Mistral-7B-Instruct-v0.2"
# BASE_MODEL      = "rombodawg/Rombos-LLM-V2.5-Qwen-32b"
# BASE_MODEL      = "Qwen/Qwen2.5-14B-Instruct-1M"
# FINETUNED_MODEL = "src/Mistral_LLM_7B_Instruct-v0.2_Qlora_finetuned/merged_fp16"
FINETUNED_MODEL = "src/Mistral-7B-Instruct-v0.2_Qlora_finetuned_w_wrapping-03-06-2025/merged_fp16" 
DATA_PATH       = "src/Dataset_Gijs_prompts.xlsx"
# OUTPUT_DIR      = "Evaluation_results_Rombos-LLM-V2.5-Qwen-32b_Qlora__w_wrapping_31-05-2025"
OUTPUT_DIR      = "Evaluation_results_Mistral-7B-Instruct-v0.2_Qlora_finetuned_w_wrapping_03-06-2025"
SEQ_LEN         = 1300
MAX_NEW_TOKENS  = 150
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE           = torch.float16 if DEVICE == "cuda" else torch.float32

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Running inference on {DEVICE} with dtype={DTYPE}")

# ─── HELPERS ─────────────────────────────────────────────────────────────────────
def parse_relationship_from_output(text: str) -> str:
    """
    Extracts the value behind "Relationship":"...".
    Returns 'positive', 'negative', 'none', or '' if no match.
    """
    m = re.search(r'[Rr]elationship"\s*:\s*"([^"]+)"', text)
    return m.group(1).strip().lower() if m else ""

# ─── EVALUATION FUNCTION ─────────────────────────────────────────────────────────
def evaluate_model(model_name: str, tokenizer, test_ds: Dataset, debug: bool = False):
    print(f"\nLoading model from {model_name} with 4-bit quantization…")
    # configure 4-bit inference
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    # ensure pad token is set
    if model.config.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    trues, preds, records = [], [], []
    for idx, ex in enumerate(test_ds):
        prompt          = ex["prompt"]
        true_completion = ex["completion"]
        true_rel        = parse_relationship_from_output(true_completion)

        # wrap in Instruct chat tags
        wrapped = f"<s>[INST] {prompt} [/INST]"

        # tokenize
        enc = tokenizer(
            wrapped,
            return_tensors="pt",
            truncation=True,
            max_length=SEQ_LEN,
            padding=False,
            add_special_tokens=True,
        ).to(DEVICE)

        # generate
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=None,  # force full-length
                pad_token_id=tokenizer.pad_token_id,
            )[0]

        # decode only new tokens
        prompt_len = enc["input_ids"].shape[-1]
        gen_ids    = out_ids[prompt_len:].tolist()
        raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred_rel   = parse_relationship_from_output(raw_output)

        if debug and idx < 5:
            print(f"[DEBUG] #{idx} raw={raw_output!r} → pred={pred_rel!r}")

        trues.append(true_rel)
        preds.append(pred_rel)
        records.append({
            "prompt": prompt,
            "true":   true_rel,
            "output": raw_output,
            "pred":   pred_rel,
            "correct": pred_rel == true_rel,
        })

    return trues, preds, records

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    # load & split dataset
    df = pd.read_excel(DATA_PATH).dropna(subset=["prompt","completion"])
    ds = Dataset.from_pandas(df[["prompt","completion"]]).shuffle(seed=42)
    test_ds = ds.train_test_split(test_size=0.2, seed=42)["test"]

    # load tokenizer (shared for both base & fine-tuned)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # evaluate both models
    for label, model_name in [("base_qlora", BASE_MODEL), ("fine_tuned", FINETUNED_MODEL)]:
        print(f"\n>> Evaluating {label}")
        trues, preds, records = evaluate_model(model_name, tokenizer, test_ds, debug=(label=="fine_tuned"))

        # compute metrics
        acc  = accuracy_score(trues, preds)
        prec = precision_score(trues, preds, average="weighted", zero_division=0)
        rec  = recall_score(trues, preds, average="weighted", zero_division=0)
        f1   = f1_score(trues, preds, average="weighted", zero_division=0)
        report_dict = classification_report(trues, preds, digits=4, zero_division=0, output_dict=True)
        labels     = sorted(l for l in ["positive","negative","none"] if l in trues)
        cm         = confusion_matrix(trues, preds, labels=labels)

        # write results to Excel
        df_rec = pd.DataFrame.from_records(records)
        df_met = pd.DataFrame([{
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "n_samples": len(trues),
        }])
        df_rep = pd.DataFrame(report_dict).transpose()
        df_cm  = pd.DataFrame(cm, index=labels, columns=labels)

        out_path = os.path.join(OUTPUT_DIR, f"{label}_results.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            df_rec.to_excel(writer, sheet_name="per_example", index=False)
            df_met.to_excel(writer, sheet_name="metrics",      index=False)
            df_rep.to_excel(writer, sheet_name="report")
            df_cm.to_excel(writer, sheet_name="confusion")
        print(f"  → Saved results to {out_path}")

if __name__ == "__main__":
    main()

