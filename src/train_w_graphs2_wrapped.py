#!/usr/bin/env python3
import os
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
from sklearn.metrics import accuracy_score

# =====================================
# Configuration
# =====================================
# MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
# MODEL_ID   = "Qwen/Qwen2.5-14B-Instruct-1M"
MODEL_ID   = "rombodawg/Rombos-LLM-V2.5-Qwen-32b"
# DATA_PATH  = "src/Dataset_Gijs_prompts.xlsx"
DATA_PATH  = "src/Dataset_Gijs_prompts_with_reasoning_and_mediators_deepseekR1_distilled.xlsx"
SEQ_LEN    = 1500     # ≤ 32768 for this model
# OUTPUT_DIR = "Mistral-7B-Instruct-v0.2_lora_finetuned_w_wrapping_and_reasoning_traces_and_mediators-15-06-2025"
# OUTPUT_DIR = "Qwen2.5-14B-Instruct-1M_lora_finetuned_w_wrapping_and_reasoning_traces_and_mediators-16-06-2025"
OUTPUT_DIR = "Rombos-32B-LLMV2.5-Qwen-32b_lora_finetuned_w_wrapping_and_reasoning_traces_and_mediators-17-06-2025"

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

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)

    # 1) Load and split data
    df = pd.read_excel(DATA_PATH).dropna(subset=["prompt","completion"])
    ds = Dataset.from_pandas(df[["prompt","completion"]]).shuffle(seed=42)
    split = ds.train_test_split(test_size=0.2, seed=42)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # 3) Load & quantize instruct model
    # bnb_cfg = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    # )

    # 4) Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # model = prepare_model_for_kbit_training(model)

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
        output_dir=OUTPUT_DIR,
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
        # compute_metrics=compute_metrics,
    )

    # 10) Train
    trainer.train()

    # 11) Plots & save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_training(trainer, OUTPUT_DIR)

    # 12) Save adapters & merged
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(f"{OUTPUT_DIR}/merged_fp16")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_fp16")

if __name__ == "__main__":
    main()
