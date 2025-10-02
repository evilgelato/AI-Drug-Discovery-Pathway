import pandas as pd
from typing import Tuple
from src.features import rdkit_descriptors, morgan_fingerprint_bits

def load_csv(path: str, smiles_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert smiles_col in df.columns and label_col in df.columns
    return df[[smiles_col, label_col]].rename(columns={smiles_col: "smiles", label_col: "label"})

def build_feature_table(df: pd.DataFrame, use_morgan=False) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        feats = rdkit_descriptors(row["smiles"])
        if feats is None:
            continue
        if use_morgan:
            fp = morgan_fingerprint_bits(row["smiles"])
            if fp is None:
                continue
            for i, bit in enumerate(fp):
                feats[f"FP_{i}"] = bit
        feats["label"] = row["label"]
        rows.append(feats)
    return pd.DataFrame(rows).dropna()
