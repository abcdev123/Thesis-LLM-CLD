# import os
# import json
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling,
#     BitsAndBytesConfig
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
# # MODEL_ID   = "Qwen/Qwen2.5-72B-Instruct"
# # MODEL_ID   = "rombodawg/Rombos-LLM-V2.5-Qwen-32b"
# DATA_PATH  = "src/Dataset_Gijs_prompts.xlsx"
# SEQ_LEN    = 1024   # max sequence length
# EOS_ID     = None   # set after tokenizer init
# PAD_ID     = None   # set after tokenizer init
# # OUTPUT_DIR = "Rombos_LLM_32B_lora_finetuned"
# OUTPUT_DIR = "Mistral_LLM_7B_Instruct-v0.2_Qlora_finetuned_wrapped"

# # Speed-up flags
# torch.backends.cuda.matmul.allow_tf32 = True

# def plot_training(trainer, output_dir):
#     logs = trainer.state.log_history
#     steps = [l["step"] for l in logs if "step" in l]
#     train_loss = [l["loss"] for l in logs if "loss" in l]
#     eval_loss  = [l["eval_loss"] for l in logs if "eval_loss" in l]
#     lr         = [l["learning_rate"] for l in logs if "learning_rate" in l]

#     # Loss & Perplexity
#     plt.figure()
#     if train_loss:
#         plt.plot(steps[:len(train_loss)], train_loss, label="train_loss")
#         # plt.plot(steps[:len(train_loss)], np.exp(train_loss), linestyle="--", label="train_ppl")
#     if eval_loss:
#         eval_steps = steps[1:1+len(eval_loss)]
#         plt.plot(eval_steps, eval_loss, label="eval_loss")
#         # plt.plot(eval_steps, np.exp(eval_loss), linestyle="--", label="eval_ppl")
#     plt.xlabel("Step")
#     plt.ylabel("Loss")
#     plt.title("Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "loss_plot.png"))

#     # Learning rate
#     if lr:
#         plt.figure()
#         plt.plot(steps[:len(lr)], lr)
#         plt.xlabel("Step")
#         plt.ylabel("Learning Rate")
#         plt.title("LR Schedule")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, "lr_plot.png"))


# def tokenize(row, tokenizer):
#     prompt_ids = tokenizer(
#         row["prompt"],
#         truncation=True,
#         max_length=SEQ_LEN,
#         add_special_tokens=False
#     )["input_ids"]
#     completion_ids = tokenizer(
#         row["completion"],
#         truncation=True,
#         max_length=SEQ_LEN // 4,
#         add_special_tokens=False
#     )["input_ids"]

#     ids = prompt_ids + completion_ids + [EOS_ID]
#     if len(ids) > SEQ_LEN:
#         ids = ids[-SEQ_LEN:]
#     else:
#         ids += [PAD_ID] * (SEQ_LEN - len(ids))

#     attn = [0 if tok == PAD_ID else 1 for tok in ids]
#     prompt_len = min(len(prompt_ids), SEQ_LEN - 1)
#     labels = [-100] * prompt_len + ids[prompt_len:]
#     return {"input_ids": ids, "attention_mask": attn, "labels": labels}

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Running on:", device)

#     # 1) Data
#     df = pd.read_excel(DATA_PATH)
#     ds = Dataset.from_pandas(df[["prompt","completion"]]).shuffle(seed=42)
#     split = ds.train_test_split(test_size=0.2, seed=42)

#     # 2) Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     global EOS_ID, PAD_ID
#     EOS_ID = tokenizer.eos_token_id
#     PAD_ID = tokenizer.pad_token_id

#     # 3) Load & quantize base model
#     # bnb_cfg = BitsAndBytesConfig(
#     #     load_in_4bit=True,
#     #     bnb_4bit_quant_type="nf4",
#     #     bnb_4bit_compute_dtype=torch.float16,
#     #     bnb_4bit_use_double_quant=True,
#     # )
#     base_model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         # quantization_config=bnb_cfg,
#         device_map="auto",
#         # torch_dtype=torch.float16,
#         low_cpu_mem_usage=True,
#     )

#     # prepare for qlora
#     # base_model = prepare_model_for_kbit_training(base_model)

#     # 4) LoRA
#     lora_cfg = LoraConfig(
#         r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
#         target_modules=[
#             "q_proj","k_proj","v_proj","o_proj",
#             "gate_proj","up_proj","down_proj"
#         ],
#         task_type="CAUSAL_LM",
#     )
#     model = get_peft_model(base_model, lora_cfg)
#     model.print_trainable_parameters()

#     # turn off the transformer cache (incompatible with checkpointing)
#     model.config.use_cache = False

#     # Enable gradient checkpointing to save memory
#     model.gradient_checkpointing_enable()
#     model.enable_input_require_grads()

#     # 5) Tokenize
#     tok_ds = split.map(
#         lambda r: tokenize(r, tokenizer),
#         remove_columns=split["train"].column_names,
#         num_proc=4, desc="Tokenising"
#     )

#     # save the tokenized datasets
#     # tok_ds["train"].save_to_disk(f"{OUTPUT_DIR}/tokenized_train")
#     # tok_ds["test"].save_to_disk(f"{OUTPUT_DIR}/tokenized_test")


#     # 6) Data collator
#     data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

#     # 7) Training args
#     training_args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         gradient_accumulation_steps=16,
#         num_train_epochs=3,
#         evaluation_strategy="epoch",    # only eval after each epoch instead of steps
#         # eval_steps=200,
#         save_strategy="no",
#         logging_steps=50,
#         bf16=True,
#         gradient_checkpointing=True,
#         remove_unused_columns=False,
#         eval_accumulation_steps=1,      # accumulate 1 batch at a time
#         learning_rate=2e-4,             
#         optim="adamw_torch",            
#         weight_decay=0.01,
#     )

#     # 8) Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tok_ds["train"],
#         eval_dataset=tok_ds["test"],
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#     )

#     # 9) Train
#     trainer.train()

#     # 10) Plots
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     plot_training(trainer, OUTPUT_DIR)

#     # 11) Save adapters & merged
#     model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
#     tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
#     merged = model.merge_and_unload()
#     merged.save_pretrained(f"{OUTPUT_DIR}/merged_fp16")
#     tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_fp16")

# if __name__ == "__main__":
#     main()












# import os
# import json
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling,
#     BitsAndBytesConfig
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
# # MODEL_ID   = "Qwen/Qwen2.5-72B-Instruct"
# # MODEL_ID   = "rombodawg/Rombos-LLM-V2.5-Qwen-32b"
# DATA_PATH  = "src/Dataset_Gijs_prompts.xlsx"
# SEQ_LEN    = 1024   # max sequence length
# EOS_ID     = None   # set after tokenizer init
# PAD_ID     = None   # set after tokenizer init
# # OUTPUT_DIR = "Rombos_LLM_32B_lora_finetuned"
# OUTPUT_DIR = "Mistral_LLM_7B_Instruct-v0.2_Qlora_finetuned"

# # Speed-up flags
# torch.backends.cuda.matmul.allow_tf32 = True

# def plot_training(trainer, output_dir):
#     logs = trainer.state.log_history
#     steps = [l["step"] for l in logs if "step" in l]
#     train_loss = [l["loss"] for l in logs if "loss" in l]
#     eval_loss  = [l["eval_loss"] for l in logs if "eval_loss" in l]
#     lr         = [l["learning_rate"] for l in logs if "learning_rate" in l]

#     # Loss & Perplexity
#     plt.figure()
#     if train_loss:
#         plt.plot(steps[:len(train_loss)], train_loss, label="train_loss")
#         plt.plot(steps[:len(train_loss)], np.exp(train_loss), linestyle="--", label="train_ppl")
#     if eval_loss:
#         eval_steps = steps[1:1+len(eval_loss)]
#         plt.plot(eval_steps, eval_loss, label="eval_loss")
#         plt.plot(eval_steps, np.exp(eval_loss), linestyle="--", label="eval_ppl")
#     plt.xlabel("Step")
#     plt.ylabel("Loss / Perplexity")
#     plt.title("Loss & Perplexity")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "loss_ppl_plot.png"))

#     # Learning rate
#     if lr:
#         plt.figure()
#         plt.plot(steps[:len(lr)], lr)
#         plt.xlabel("Step")
#         plt.ylabel("Learning Rate")
#         plt.title("LR Schedule")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, "lr_plot.png"))


# def tokenize(row, tokenizer):
#     prompt_ids = tokenizer(
#         row["prompt"],
#         truncation=True,
#         max_length=SEQ_LEN,
#         add_special_tokens=False
#     )["input_ids"]
#     completion_ids = tokenizer(
#         row["completion"],
#         truncation=True,
#         max_length=SEQ_LEN // 4,
#         add_special_tokens=False
#     )["input_ids"]

#     ids = prompt_ids + completion_ids + [EOS_ID]
#     if len(ids) > SEQ_LEN:
#         ids = ids[-SEQ_LEN:]
#     else:
#         ids += [PAD_ID] * (SEQ_LEN - len(ids))

#     attn = [0 if tok == PAD_ID else 1 for tok in ids]
#     prompt_len = min(len(prompt_ids), SEQ_LEN - 1)
#     labels = [-100] * prompt_len + ids[prompt_len:]
#     return {"input_ids": ids, "attention_mask": attn, "labels": labels}

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Running on:", device)

#     # 1) Data
#     df = pd.read_excel(DATA_PATH)
#     ds = Dataset.from_pandas(df[["prompt","completion"]]).shuffle(seed=42)
#     split = ds.train_test_split(test_size=0.2, seed=42)

#     # 2) Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     global EOS_ID, PAD_ID
#     EOS_ID = tokenizer.eos_token_id
#     PAD_ID = tokenizer.pad_token_id

#     # 3) Load & quantize base model
#     bnb_cfg = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#     )
#     base_model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         quantization_config=bnb_cfg,
#         device_map="auto",
#         # torch_dtype=torch.float16,
#         low_cpu_mem_usage=True,
#     )

#     # prepare for qlora
#     base_model = prepare_model_for_kbit_training(base_model)

#     # 4) LoRA
#     lora_cfg = LoraConfig(
#         r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
#         target_modules=[
#             "q_proj","k_proj","v_proj","o_proj",
#             "gate_proj","up_proj","down_proj"
#         ],
#         task_type="CAUSAL_LM",
#     )
#     model = get_peft_model(base_model, lora_cfg)
#     model.print_trainable_parameters()

#     # turn off the transformer cache (incompatible with checkpointing)
#     model.config.use_cache = False

#     # Enable gradient checkpointing to save memory
#     model.gradient_checkpointing_enable()
#     model.enable_input_require_grads()

#     # 5) Tokenize
#     tok_ds = split.map(
#         lambda r: tokenize(r, tokenizer),
#         remove_columns=split["train"].column_names,
#         num_proc=4, desc="Tokenising"
#     )

#     # save the tokenized datasets
#     # tok_ds["train"].save_to_disk(f"{OUTPUT_DIR}/tokenized_train")
#     # tok_ds["test"].save_to_disk(f"{OUTPUT_DIR}/tokenized_test")


#     # 6) Data collator
#     data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

#     # 7) Training args
#     training_args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         gradient_accumulation_steps=16,
#         num_train_epochs=3,
#         evaluation_strategy="epoch",    # only eval after each epoch instead of steps
#         # eval_steps=200,
#         save_strategy="no",
#         logging_steps=50,
#         bf16=True,
#         gradient_checkpointing=True,
#         remove_unused_columns=False,
#         eval_accumulation_steps=1,      # accumulate 1 batch at a time
#     )

#     # 8) Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tok_ds["train"],
#         eval_dataset=tok_ds["test"],
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#     )

#     # 9) Train
#     trainer.train()

#     # 10) Plots
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     plot_training(trainer, OUTPUT_DIR)

#     # 11) Save adapters & merged
#     model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
#     merged = model.merge_and_unload()
#     merged.save_pretrained(f"{OUTPUT_DIR}/merged_fp16")
#     tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_fp16")

# if __name__ == "__main__":
#     main()


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
MODEL_ID   = "mistralai/Mistral-7B-v0.1"  # Base Mistral
DATA_PATH  = "src/Dataset_Gijs_prompts.xlsx"
SEQ_LEN    = 1024   # <= 8192 for base model
EOS_ID     = None   # will set after tokenizer init
PAD_ID     = None   # will set after tokenizer init
OUTPUT_DIR = "Mistral_LLM_7B_v0.1_Base_lora_finetuned"

# Speed-up flags
torch.backends.cuda.matmul.allow_tf32 = True

def plot_training(trainer, output_dir):
    logs = trainer.state.log_history
    steps = [l["step"] for l in logs if "step" in l]
    train_loss = [l["loss"] for l in logs if "loss" in l]
    eval_loss  = [l["eval_loss"] for l in logs if "eval_loss" in l]
    lr         = [l["learning_rate"] for l in logs if "learning_rate" in l]

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
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lr_plot.png"))


def tokenize_base(row, tokenizer):
    # Concatenate prompt and completion, add special tokens, pad/truncate to SEQ_LEN
    enc = tokenizer(
        row["prompt"],
        row["completion"],
        truncation=True,
        max_length=SEQ_LEN,
        padding="max_length",
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    # Labels: same as input_ids, but mask padding
    labels = [l if l != PAD_ID else -100 for l in input_ids]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    global EOS_ID, PAD_ID
    EOS_ID = tokenizer.eos_token_id
    PAD_ID = tokenizer.pad_token_id

    # 3) Load & quantize base model
    # bnb_cfg = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    # )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # quantization_config=bnb_cfg,
        device_map="auto",
        # torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # prepare for qlora if desired
    # base_model = prepare_model_for_kbit_training(base_model)

    # 4) LoRA configuration
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # disable cache and enable gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 5) Tokenize dataset
    tok_ds = split.map(
        lambda r: tokenize_base(r, tokenizer),
        remove_columns=split["train"].column_names,
        num_proc=4,
        desc="Tokenising"
    )

    # 6) Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 7) Training arguments
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

    # 8) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 9) Train
    trainer.train()

    # 10) Save plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_training(trainer, OUTPUT_DIR)

    # 11) Save adapters & merged model
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter_7Bv0.1")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter_7Bv0.1")
    merged = model.merge_and_unload()
    merged.save_pretrained(f"{OUTPUT_DIR}/merged_fp16_7Bv0.1")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged_fp16_7Bv0.1")

if __name__ == "__main__":
    main()


