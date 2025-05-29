#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_DIR      = "mistralai/Mistral-7B-v0.1"            # base Mistral checkpoint
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16 if DEVICE == "cuda" else torch.float32
SEQ_LEN        = 1024
MAX_NEW_TOKENS = 50

# ─── MODEL LOADING ───────────────────────────────────────────────────────────────
def load_model():
    """
    Loads the tokenizer and model from MODEL_DIR onto the correct device.
    """
    print(f"Loading model from {MODEL_DIR} on {DEVICE} (dtype={DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=(DEVICE == "cpu"),
    ).to(DEVICE)

    # Ensure pad token is defined
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return tokenizer, model

# ─── GENERATION ─────────────────────────────────────────────────────────────────
def generate_response(prompt: str, tokenizer, model) -> str:
    """
    Wraps the raw prompt in an Instruction/Response template, then generates.
    """
    # Build the wrapper
    wrapper = (
        "### Instruction:\n"
        f"{prompt}\n"
        "### Response:\n"
    )

    # Tokenize the wrapped prompt (includes BOS/EOS)
    enc = tokenizer(
        wrapper,
        return_tensors="pt",
        truncation=True,
        max_length=SEQ_LEN,
        padding=False,
        add_special_tokens=True,
    ).to(DEVICE)

    # Generate continuation, forcing full length
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=None,                  # disable early stopping
            pad_token_id=tokenizer.pad_token_id,
        )

    # Slice off the prompt tokens and decode only the new ones
    gen_ids = output_ids[0, enc["input_ids"].shape[-1]:].tolist()
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return response.strip()

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    tokenizer, model = load_model()

    test_prompts = [
        "You are a scientific system dynamics model expert. Given 'More chickens' and 'More eggs', what is the causal relationship? Don't repeat the prompt, just return the relationship only.",
        "You are a scientific system dynamics model expert. Given 'Price drops' and 'Sales rise', what is the causal relationship? Don't repeat the prompt, just return the relationship only.",
    ]

    for prompt in test_prompts:
        print("\n→ Prompt:\n", prompt)
        response = generate_response(prompt, tokenizer, model)
        print("\n→ Response:\n", response)

if __name__ == "__main__":
    main()










