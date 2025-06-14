#!/usr/bin/env python3
"""
DeepSeek-Reasoner smoke test (first SAMPLE_N prompts).

Writes one-column completions of the form:

RT: <reasoning>

ANS: <JSON answer>
"""

import os, time, pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError          # ← new client & exception

# ─── CONFIG ────────────────────────────────────────────────────────────────────
load_dotenv()                                      # must come *before* getenv
client = OpenAI(                                   # new client instance
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com/v1",
)

MODEL        = "deepseek-reasoner"
TEMPERATURE  = 0.3
MAX_TOKENS   = 2000

DATA_IN   = "src/Dataset_Gijs_prompts_w_mediators_and_context.xlsx"
DATA_OUT  = "Dataset_Gijs_prompts_with_reasoning_and_mediators_deepseekR1_distilled.xlsx"
SAMPLE_N  = None            # None → full run

# ─── PROMPT BUILDER ────────────────────────────────────────────────────────────
def build_inline_prompt(user_prompt: str) -> str:
    return (
        "<s>[INST] You are an expert system-dynamics reasoning assistant. "
        "After thinking step-by-step, output EXACTLY:\n"
        "RT: <your reasoning>\n\n"
        "ANS: <valid JSON answer>\n"
        "Do NOT omit 'RT:' or 'ANS:'. [/INST]\n\n"
        f"{user_prompt}"
    )

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    df = (
        pd.read_excel(DATA_IN)
          .dropna(subset=["prompt"])
          .head(SAMPLE_N)
          .reset_index(drop=True)
    )

    completions = []

    for prompt in tqdm(df["prompt"], desc="DeepSeek-Reasoner"):
        inline_prompt = build_inline_prompt(prompt)

        try:
            resp = client.chat.completions.create(      # ← new call
                model       = MODEL,
                messages    = [{"role": "user", "content": inline_prompt}],
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
            )
        except RateLimitError:
            time.sleep(10)
            continue

        text = resp.choices[0].message.content

        if "ANS:" not in text:                          # safety tag
            text = f"RT: \n{text.strip()}\n\nANS: {{}}"

        completions.append(text.strip())

    pd.DataFrame({
        "prompt"    : df["prompt"],
        "completion": completions,
    }).to_excel(DATA_OUT, index=False)
    print(f"✅  Saved augmented dataset ➜  {DATA_OUT}")

# ─── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()




