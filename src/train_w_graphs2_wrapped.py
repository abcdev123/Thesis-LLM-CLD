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

# =====================================
# Configuration
# =====================================
MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_PATH  = "src/Dataset_Gijs_prompts.xlsx"
SEQ_LEN    = 2048     # keep <= block size you want (model max is 32768)
EOS_ID     = None     # initialized after tokenizer is loaded
PAD_ID     = None     # initialized after tokenizer is loaded
OUTPUT_DIR = "Mistral_LLM_7B_Instruct-v0.2_lora_finetuned_w_wrapping"

# Speed-up flags
torch.backends.cuda.matmul.allow_tf32 = True

def plot_training(trainer, output_dir):
    logs = trainer.state.log_history
    steps = [l["step"] for l in logs if "step" in l]
    train_loss = [l["loss"] for l in logs if "loss" in l]
    eval_loss  = [l["eval_loss"] for l in logs if "eval_loss" in l]
    lr         = [l["learning_rate"] for l in logs if "learning_rate" in l]

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

    if lr:
        plt.figure()
        plt.plot(steps[:len(lr)], lr)
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lr_plot.png"))


def tokenize_instruct(examples, tokenizer):
    # wrap prompt and completion in [INST] tags
    texts, labels = [], []
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        # input string: <s>[INST] prompt [/INST] completion</s>
        text = f"<s>[INST] {prompt} [/INST] {completion}"
        texts.append(text)
    # tokenize
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
        add_special_tokens=False,  # we've added <s> and [/INST] manually
    )
    # build labels to ignore prompt portion
    for i, (input_ids, prompt) in enumerate(zip(enc["input_ids"], examples["prompt"])):
        # count tokens in prompt wrapper: <s>[INST] prompt [/INST]
        proto = f"<s>[INST] {prompt} [/INST]"
        toks = tokenizer(proto, add_special_tokens=False).input_ids
        prompt_len = len(toks)
        # mask prompt tokens
        lbl = [-100] * prompt_len + input_ids[prompt_len:]
        # pad or trim to SEQ_LEN
        lbl = lbl[:SEQ_LEN] + [-100] * max(0, SEQ_LEN - len(lbl))
        labels.append(lbl)
    enc["labels"] = labels
    # set pad and eos
    enc["attention_mask"] = [[0 if tok == PAD_ID else 1 for tok in seq] for seq in enc["input_ids"]]
    return enc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)

    # 1) Load data
    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=["prompt","completion"])
    ds = Dataset.from_pandas(df[["prompt","completion"]]).shuffle(seed=42)
    split = ds.train_test_split(test_size=0.2, seed=42)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    # add pad_token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    global EOS_ID, PAD_ID
    EOS_ID = tokenizer.eos_token_id
    PAD_ID = tokenizer.pad_token_id

    # 3) Load & quantize instruct model
    # bnb_cfg = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # prepare LoRA
    # model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 4) Tokenize dataset
    tok_ds = split.map(
        lambda ex: tokenize_instruct(ex, tokenizer),
        batched=True,
        remove_columns=split["train"].column_names,
        num_proc=4,
        desc="Tokenising"
    )

    # 5) Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 6) Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
        learning_rate=2e-4,
        optim="adamw_torch",
        weight_decay=0.01,
    )

    # 7) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 8) Train
    trainer.train()

    # 9) Save plots and model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_training(trainer, OUTPUT_DIR)

    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(f"{OUTPUT_DIR}/merged_fp16")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_fp16")

if __name__ == "__main__":
    main()
