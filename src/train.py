import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# MODEL_ID   = "cerebras/Cerebras-GPT-111M"
MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
DATA_PATH  = "src/Dataset_Gijs_prompts.xlsx"
SEQ_LEN    = 1024   # max sequence length
EOS_ID     = None   # to be set after tokenizer initialization
PAD_ID     = None   # to be set after tokenizer initialization
# OUTPUT_DIR = "cerebras_gpt_111m_lora_finetuned"
OUTPUT_DIR = "mistral_7b_lora_finetuned"

# Speed-up flags
torch.backends.cuda.matmul.allow_tf32 = True

def tokenize(row, tokenizer):
    """
    Tokenizes a single example into input_ids, attention_mask, and labels with fixed length.
    """
    # 1) Tokenise prompt and completion separately
    prompt_ids = tokenizer(
        row["prompt"],
        truncation=True,
        max_length=SEQ_LEN,
        add_special_tokens=False
    )["input_ids"]

    completion_ids = tokenizer(
        row["completion"],
        truncation=True,
        max_length=SEQ_LEN // 4,
        add_special_tokens=False
    )["input_ids"]

    # 2) Combine prompt, completion, and EOS
    ids = prompt_ids + completion_ids + [EOS_ID]

    # 3) Pad or truncate to fixed length
    if len(ids) > SEQ_LEN:
        ids = ids[-SEQ_LEN:]
    else:
        ids = ids + [PAD_ID] * (SEQ_LEN - len(ids))

    # 4) Create attention mask and labels
    attn = [0 if token == PAD_ID else 1 for token in ids]
    prompt_len = min(len(prompt_ids), SEQ_LEN - 1)
    labels = [-100] * prompt_len + ids[prompt_len:]

    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def main():
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)

    # Load data
    df = pd.read_excel(DATA_PATH)
    dataset = Dataset.from_pandas(df[["prompt", "completion"]])
    dataset = dataset.shuffle(seed=42)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    # Load tokenizer and set special tokens
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    global EOS_ID, PAD_ID
    EOS_ID = tokenizer.eos_token_id
    PAD_ID = tokenizer.pad_token_id

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # torch_dtype=torch.float16,
        device_map={"": device}
    )

    # Configure LoRA

    # TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]
    # lora_cfg = LoraConfig(
    #     r=32,
    #     lora_alpha=64,
    #     lora_dropout=0.05,
    #     bias="none",
    #     target_modules=TARGET_MODULES,
    #     task_type="CAUSAL_LM",
    # )

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                        "up_proj", "down_proj"],  # standard Mistral blocks
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Tokenize dataset
    tokenized = split.map(
        lambda row: tokenize(row, tokenizer),
        remove_columns=split["train"].column_names,
        num_proc=4,
        desc="Tokenising to fixed length",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        fp16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save LoRA adapters
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")

    # Merge adapters & save full model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{OUTPUT_DIR}/merged_fp16")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_fp16")


if __name__ == "__main__":
    main()
