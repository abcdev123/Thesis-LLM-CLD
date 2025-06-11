#!/usr/bin/env python3
import os
import time
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# =====================================
# Configuration
# =====================================
MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_ID   = "Qwen/Qwen2.5-14B-Instruct-1M"
# MODEL_ID   = "rombodawg/Rombos-LLM-V2.5-Qwen-32b"
DATA_PATH  = "src/Dataset_Gijs_prompts.xlsx"
# DATA_PATH  = "src/Dataset_Gijs_prompts_with_reasoning.xlsx"
SEQ_LEN    = 1300     # ≤ 32768 for this model
MAX_NEW_TOKENS = 200  # ≤ 32768 for this model
OUTPUT_DIR = "Mistral-7B-Instruct-v0.2_lora_finetuned_w_wrapping_array_job"
# OUTPUT_DIR = "Qwen2.5-14B-Instruct-1M_lora_finetuned_w_wrapping_second_run_for_comparison_with_first_run"

torch.backends.cuda.matmul.allow_tf32 = True

# ─── PLOTTING ────────────────────────────────────────────────────────────────────
def plot_training(trainer, output_dir):
    logs = trainer.state.log_history
    steps      = [l["step"]           for l in logs if "step" in l]
    train_loss = [l["loss"]           for l in logs if "loss" in l]
    eval_loss  = [l["eval_loss"]      for l in logs if "eval_loss" in l]
    # train_acc  = [l["accuracy"]       for l in logs if "accuracy" in l]
    # eval_acc   = [l["eval_accuracy"]  for l in logs if "eval_accuracy" in l]
    lr         = [l["learning_rate"]  for l in logs if "learning_rate" in l]

    # Loss plot
    plt.figure()
    if train_loss:
        plt.plot(steps[:len(train_loss)], train_loss, label="train_loss")
    if eval_loss:
        eval_steps = steps[1:1+len(eval_loss)]
        plt.plot(eval_steps, eval_loss, label="eval_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Eval Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))

    # Learning rate plot
    if lr:
        plt.figure()
        plt.plot(steps[:len(lr)], lr)
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("LR Schedule")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lr_plot.png"))

# ─── TOKENIZATION ─────────────────────────────────────────────────────────────────
def tokenize_instruct(examples, tokenizer, pad_id):
    texts, labels = [], []
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        text = f"<s>[INST] {prompt} [/INST] {completion}"
        texts.append(text)
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
        add_special_tokens=False,
    )
    # mask prompt tokens in labels
    for i, prompt in enumerate(examples["prompt"]):
        proto = f"<s>[INST] {prompt} [/INST]"
        tok_proto = tokenizer(proto, add_special_tokens=False).input_ids
        prompt_len = len(tok_proto)
        input_ids = enc["input_ids"][i]
        lbl = [-100] * prompt_len + input_ids[prompt_len:]
        lbl = lbl[:SEQ_LEN] + [-100] * max(0, SEQ_LEN - len(lbl))
        labels.append(lbl)
    enc["labels"] = labels
    enc["attention_mask"] = [
        [0 if tok == pad_id else 1 for tok in seq]
        for seq in enc["input_ids"]
    ]
    return enc


def parse_relationship_from_output(text: str) -> str:
    """
    Regex out exactly the value behind "relationship":"...".
    Returns 'positive', 'negative', or 'none' (lowercased), or '' if no match.
    """
    m = re.search(r'[Rr]elationship"\s*:\s*"([^"]+)"', text)
    return m.group(1).strip().lower() if m else ""


def evaluate_model(model_name: str, tokenizer, test_ds: Dataset, debug: bool = False):
    print(f"\nLoading model from {model_name}…")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    trues, preds, records = [], [], []
    for idx, ex in enumerate(test_ds):
        prompt          = ex["prompt"]
        true_completion = ex["completion"]
        true_rel        = parse_relationship_from_output(true_completion)

        wrapped = f"<s>[INST] {prompt} [/INST]"
        encodings = tokenizer(
            wrapped,
            truncation=True,
            max_length=SEQ_LEN,
            padding=False,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            ids = model.generate(
                **encodings,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )[0]

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
def main(job_id):
    print(f"Starting job {job_id}...")

    base_path = f"data/{time.strftime('/%Y/%m/%d')}/{job_id}/"
    os.makedirs(base_path, exist_ok=True)
    full_output_dir = os.path.join(base_path, OUTPUT_DIR)

    # Bepaal SLURM array-task en totaal aantal
    task_id    = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    print(f"Task {task_id+1}/{task_count} gestart")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)

    # 1) Load and split data
    df = pd.read_excel(DATA_PATH).dropna(subset=["prompt","completion"])
    ds = Dataset.from_pandas(df[["prompt","completion"]]).shuffle()
    split = ds.train_test_split(test_size=0.2)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # 3) Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 5) LoRA setup
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 6) Tokenize
    tok_ds = split.map(
        lambda ex: tokenize_instruct(ex, tokenizer, pad_id),
        batched=True,
        remove_columns=split["train"].column_names,
        num_proc=4,
        desc="Tokenising"
    )

    # 7) Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 8) Training args
    training_args = TrainingArguments(
        output_dir=full_output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
        learning_rate=2e-4,
        optim="adamw_torch",
        weight_decay=0.01,
    )

    # 9) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
  
    # 10) Train
    trainer.train()

    # 11) Plots & save metrics
    os.makedirs(full_output_dir, exist_ok=True)
    plot_training(trainer, full_output_dir)

    # 12) Conditionally save modellen
    if task_id == task_count - 1:
        print("Laatste taak: saven modellen.")
        # alleen de laatste taak schrijft weg
        model.save_pretrained(f"{full_output_dir}/lora_adapter")
        tokenizer.save_pretrained(f"{full_output_dir}/lora_adapter")
        merged = model.merge_and_unload()
        merged.save_pretrained(f"{full_output_dir}/merged_fp16")
        tokenizer.save_pretrained(f"{full_output_dir}/merged_fp16")
    else:
        print(f"Taak {task_id}: save overgeslagen.")

    # ─── EVALUATION ──────────────────────────────────────────────────────────────────
    test_dataset  = split["test"]
    FINETUNED_MODEL = f"{full_output_dir}/merged_fp16"

    evaltokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL, use_fast=True)
    evaltokenizer.padding_side = "right"
    if evaltokenizer.pad_token is None: 
        evaltokenizer.pad_token = evaltokenizer.eos_token

    for label, model_name in [("base", MODEL_ID), ("lora", FINETUNED_MODEL)]:
        print(f"\n>> Evaluating {label} model")
        trues, preds, records = evaluate_model(
            model_name, evaltokenizer, test_dataset, debug=(label=="lora")
        )

        acc         = accuracy_score(trues, preds)
        prec        = precision_score(trues, preds, average="weighted", zero_division=0)
        rec         = recall_score(trues, preds, average="weighted", zero_division=0)
        f1          = f1_score(trues, preds, average="weighted", zero_division=0)
        report_dict = classification_report(
            trues, preds, digits=4, zero_division=0, output_dict=True
        )
        labels      = sorted(l for l in ["positive","negative","none"] if l in trues)
        cm          = confusion_matrix(trues, preds, labels=labels)

        df_rec   = pd.DataFrame.from_records(records)
        df_met   = pd.DataFrame([{\
            "accuracy":  acc,\
            "precision": prec,\
            "recall":    rec,\
            "f1":        f1,\
            "n_samples": len(trues),\
            "job_id":    job_id,\
            "label":     label,\
        }])
        df_rep   = pd.DataFrame(report_dict).transpose()
        df_cm    = pd.DataFrame(cm, index=labels, columns=labels)

        out_path = os.path.join(full_output_dir, f"{label}_results.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            df_rec.to_excel(writer, sheet_name="per_example", index=False)
            df_met.to_excel(writer, sheet_name="metrics",      index=False)
            df_rep.to_excel(writer, sheet_name="report")
            df_cm.to_excel(writer, sheet_name="confusion")
        print("  → Saved results to", out_path)

if __name__ == "__main__":
    import sys
    import uuid

    job_id = sys.argv[1] if len(sys.argv) > 1 else uuid.uuid4()
    print(f"Running job with ID: {job_id}")

    main(job_id)
