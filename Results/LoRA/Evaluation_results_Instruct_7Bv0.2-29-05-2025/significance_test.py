import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# ─── CONFIG ───
BASE_FILE = "base_results.xlsx"
LORA_FILE = "lora_results.xlsx"

# these must match your sheet
TRUE_COL      = "true"
PRED_COL      = "pred_rel"
# ─────────────

# load both result tables
df_base = pd.read_excel(BASE_FILE)
df_lora = pd.read_excel(LORA_FILE)

# sanity check: same number of rows?
if len(df_base) != len(df_lora):
    raise ValueError(
       f"Row‐count mismat ch: base has {len(df_base)} rows, "
        f"LoRA has {len(df_lora)}."
    )

# extract true labels and predictions
y_true      = df_base[TRUE_COL].reset_index(drop=True)
y_base_pred = df_base[PRED_COL].reset_index(drop=True)
y_lora_pred = df_lora[PRED_COL].reset_index(drop=True)

# compute correctness booleans
base_correct = (y_base_pred == y_true)
lora_correct = (y_lora_pred == y_true)

# build the 2×2 contingency table:
# [ [ both wrong, base wrong & lora right ],
#   [ base right & lora wrong, both right ] ]
table = [
    [((~base_correct) & (~lora_correct)).sum(),
     ((~base_correct) & ( lora_correct)).sum()],
    [(( base_correct) & (~lora_correct)).sum(),
     (( base_correct) & ( lora_correct)).sum()],
]

# run McNemar’s exact test
result = mcnemar(table, exact=True)

# print results
print("Contingency table:")
print(f"  both wrong               = {table[0][0]}")
print(f"  base wrong, lora right   = {table[0][1]}")
print(f"  base right, lora wrong   = {table[1][0]}")
print(f"  both right               = {table[1][1]}\n")

print(f"McNemar’s χ² (exact) = {result.statistic:.3f}")
print(f"p-value             = {result.pvalue:.5f}")
print(
    "=> Significant difference (α=0.05)"
    if result.pvalue < 0.05 else
    "=> No significant difference (α=0.05)"
)
