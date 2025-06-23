import re
import pandas as pd

# === Configuration ===
V4_DATA_FILE   = "Dataset_Gijs_updated_v4.xlsx"
TEST_DATA_FILE = "Dataset_Gijs_prompts_w_mediators.xlsx"
OUTPUT_FILE    = "Dataset_Gijs_prompts_w_mediators_test.xlsx"

CLD_CTX_COL = "CLD context"
SRC_COL     = "I:Source"
TGT_COL     = "I:Target"
PROMPT_COL  = "prompt"
# ======================

# Regex om uit de prompt te halen: tussen "A" en "B"
PAIR_RE = re.compile(r'between\s+"([^"]+)"\s+and\s+"([^"]+)"', re.IGNORECASE)

def build_maps_from_v4(v4_df: pd.DataFrame):
    """
    Bouw drie dicts keyed by (source, target):
      - context_map[(src,tgt)]  -> cld context
      - mediators_map[(src,tgt)]-> comma-joined mediators
      - allvars_map[(src,tgt)]  -> comma-joined alle variabelen in die CLD
    """
    context_map, mediators_map, allvars_map = {}, {}, {}

    for ctx, grp in v4_df.groupby(CLD_CTX_COL):
        nodes = set(grp[SRC_COL]) | set(grp[TGT_COL])
        allvars_str = ", ".join(sorted(nodes))
        for _, row in grp.iterrows():
            key = (row[SRC_COL], row[TGT_COL])
            context_map[key] = ctx
            meds = sorted(nodes - {row[SRC_COL], row[TGT_COL]})
            mediators_map[key] = ", ".join(meds) if meds else "(none)"
            allvars_map[key]   = allvars_str

    return context_map, mediators_map, allvars_map

def inject_into_prompts(test_df: pd.DataFrame,
                        context_map: dict,
                        mediators_map: dict,
                        allvars_map: dict) -> pd.DataFrame:
    """
    Parse per prompt de (src,tgt), lookup in de maps, en append:
      CLD context: …
      Mediating variables: …
      All variables in this CLD: …
    """
    updates = {}
    for idx, row in test_df.iterrows():
        text = row[PROMPT_COL]
        m = PAIR_RE.search(text)
        if not m:
            print(f"⚠️ Kan SRC/TGT niet parsen in prompt #{idx}")
            continue

        src, tgt = m.group(1), m.group(2)
        key = (src, tgt)
        ctx      = context_map.get(key, "(UNKNOWN CLD)")
        meds     = mediators_map.get(key, "(none)")

        new_prompt = (
            f"{text}\n\n"
            f"CLD context: {ctx}\n\n"
            f"Other variables in this CLD (Potentially mediating for this link. Please check them carefully): {meds}\n"
        )
        updates[idx] = new_prompt

    test_df.loc[list(updates.keys()), PROMPT_COL] = pd.Series(updates)
    return test_df

def main():
    # 1) laad v4 en test
    v4_df   = pd.read_excel(V4_DATA_FILE)
    test_df = pd.read_excel(TEST_DATA_FILE)

    # 2) bouw lookup‐maps
    context_map, mediators_map, allvars_map = build_maps_from_v4(v4_df)

    # 3) injecteer
    updated = inject_into_prompts(test_df,
                                  context_map,
                                  mediators_map,
                                  allvars_map)

    # 4) sla op
    updated.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Updated prompts written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()





