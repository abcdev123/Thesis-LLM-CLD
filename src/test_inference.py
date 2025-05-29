#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# MODEL_DIR      = "mistralai/Mistral-7B-v0.1"            # base Mistral checkpoint
MODEL_DIR      = "src/Mistral_LLM_7B_Instruct-v0.2_lora_finetuned/merged_fp16"  # Path to your fine-tuned model
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16 if DEVICE == "cuda" else torch.float32
SEQ_LEN        = 1024
MAX_NEW_TOKENS = 100

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
        """You are a scientific system dynamics model expert specializing in causal loop diagrams. """
        """Given two variables and their definitions, your task is to determine the precise causal relationship between them.\n"""
        """Approach this systematically: \n"""
        """1. First, clearly understand what each variable means and how it's measured \n"""
        """2. Consider the mechanisms by which the first variable might directly affect the second \n"""
        """3. Evaluate whether this mechanism works within the given spatial and temporal scales \n"""
        """4. Check for potential mediating variables that might explain the relationship \n"""
        """5. Distinguish between correlation and causation - focus only on true causal relationships \n"""
        """6. Determine if the relationship is positive, negative, or non-existent \n"""
        """7. Verify your reasoning by considering hypothetical increases and decreases in the first variable \n"""
        """8. Provide a concise explanation that justifies your conclusion\n"""
        """Select the most appropriate relationship type based on causal logic, not mere correlation. """
        """Respond with a dictionary: {"Relationship":"TYPE","Explanation":"A brief explanation."}. """
        """Make sure there is no additional text.\n"""
        """\n"""
        """I need to determine if there is a causal relationship between \"Management burden\" (\"\") """
        """and \"Time senior engineers spend on engineering\" (\"\") for a Causal Loop Diagram (CLD). """
        """Let me think through this step by step:\n\n"""
        """Step 1: Understand the variables\n"""
        """- Variable 1: \"Management burden\" defined as \"\"\n"""
        """- Variable 2: \"Time senior engineers spend on engineering\" defined as \"\"\n"""
        """- Scale: Spatial = \"Global\", Temporal = \"Annual\"\n\n"""
        """Step 2: Consider potential causal mechanisms\n"""
        """- Does a change in \"Management burden\" directly cause a change in \"Time senior engineers spend on engineering\" through a clear mechanism?\n"""
        """- Does this mechanism operate within the given spatial and temporal scales?\n\n"""
        """Step 3: Check for mediating variables\n"""
        """- Are there variables in the CLD that mediate this relationship: \n"""
        """- If a mediator exists, the direct relationship should not be included in the CLD\n\n"""
        """Step 4: Determine relationship type\n"""
        """If there is a causal relationship, I'll classify it as one of these:\n"""
        """1. POSITIVE – An increase in \"Management burden\" causes an increase in \"Time senior engineers spend on engineering\", or a decrease in \"Management burden\" causes a decrease in \"Time senior engineers spend on engineering\".\n"""
        """2. NEGATIVE – An increase in \"Management burden\" causes a decrease in \"Time senior engineers spend on engineering\", or a decrease in \"Management burden\" causes an increase in \"Time senior engineers spend on engineering\".\n"""
        """3. NONE – No causal relationship exists between \"Management burden\" and \"Time senior engineers spend on engineering\".\n\n"""
        """Provide a brief, maximum 4-sentence explanation in the format: {\"Relationship\":\"TYPE\",\"Explanation\":\"A brief explanation.\"}\n\n"""
        """Now, define the relationship between \"Management burden\" (\"\") and \"Time senior engineers spend on engineering\" (\"\") following this format: {\"Relationship\":\"TYPE\",\"Explanation\":\"A brief explanation.\"}, taking particular care to make sure the relationships are causal and not just correlated or associated.""",
        "You are a scientific system dynamics model expert. Given 'Price drops' and 'Sales rise', what is the causal relationship? Don't repeat the prompt, just return the relationship only.",
    ]

    for prompt in test_prompts:
        print("\n→ Prompt:\n", prompt)
        response = generate_response(prompt, tokenizer, model)
        print("\n→ Response:\n", response)

if __name__ == "__main__":
    main()











# #!/usr/bin/env python3
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # ─── CONFIG ─────────────────────────────────────────────────────────────────────
# MODEL_DIR      = "mistralai/Mistral-7B-v0.1"            # base Mistral checkpoint
# DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE          = torch.float16 if DEVICE == "cuda" else torch.float32
# SEQ_LEN        = 1024
# MAX_NEW_TOKENS = 100

# # ─── MODEL LOADING ───────────────────────────────────────────────────────────────
# def load_model():
#     """
#     Loads the tokenizer and model from MODEL_DIR onto the correct device.
#     """
#     print(f"Loading model from {MODEL_DIR} on {DEVICE} (dtype={DTYPE})...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_DIR,
#         torch_dtype=DTYPE,
#         low_cpu_mem_usage=(DEVICE == "cpu"),
#     ).to(DEVICE)

#     # Ensure pad token is defined
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         model.config.pad_token_id = tokenizer.pad_token_id

#     model.eval()
#     return tokenizer, model

# # ─── GENERATION ─────────────────────────────────────────────────────────────────
# def generate_response(prompt: str, tokenizer, model) -> str:
#     """
#     Wraps the raw prompt in an Instruction/Response template, then generates.
#     """
#     # Build the wrapper
#     wrapper = (
#         "### Instruction:\n"
#         f"{prompt}\n"
#         "### Response:\n"
#     )

#     # Tokenize the wrapped prompt (includes BOS/EOS)
#     enc = tokenizer(
#         wrapper,
#         return_tensors="pt",
#         truncation=True,
#         max_length=SEQ_LEN,
#         padding=False,
#         add_special_tokens=True,
#     ).to(DEVICE)

#     # Generate continuation, forcing full length
#     with torch.no_grad():
#         output_ids = model.generate(
#             input_ids=enc["input_ids"],
#             attention_mask=enc["attention_mask"],
#             max_new_tokens=MAX_NEW_TOKENS,
#             eos_token_id=None,                  # disable early stopping
#             pad_token_id=tokenizer.pad_token_id,
#         )

#     # Slice off the prompt tokens and decode only the new ones
#     gen_ids = output_ids[0, enc["input_ids"].shape[-1]:].tolist()
#     response = tokenizer.decode(gen_ids, skip_special_tokens=True)
#     return response.strip()

# # ─── MAIN ───────────────────────────────────────────────────────────────────────
# def main():
#     tokenizer, model = load_model()

#     test_prompts = [
#         "You are a scientific system dynamics model expert. Given 'More chickens' and 'More eggs', what is the causal relationship? Don't repeat the prompt, just return the relationship only.",
#         "You are a scientific system dynamics model expert. Given 'Price drops' and 'Sales rise', what is the causal relationship? Don't repeat the prompt, just return the relationship only.",
#     ]

#     for prompt in test_prompts:
#         print("\n→ Prompt:\n", prompt)
#         response = generate_response(prompt, tokenizer, model)
#         print("\n→ Response:\n", response)

# if __name__ == "__main__":
#     main()










