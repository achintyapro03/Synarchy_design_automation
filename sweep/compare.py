#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances

# Define the canonical parameter space
param_space = {
    "num_cores": [1, 2, 4, 8, 16, 32],
    "cpu_clock_GHz": np.round(np.linspace(1.0, 4.0, 16), 2).tolist(),
    "l1i_kb": [16, 32, 64, 128],
    "l1d_kb": [16, 32, 64, 128],
    "l1_assoc": [1, 2, 4, 8],
    "l2_kb": [128, 256, 512, 1024, 2048],
    "l2_assoc": [2, 4, 8, 16],
    "fetchWidth": [2, 4, 8, 12],
    "decodeWidth": [2, 4, 8, 12],
    "issueWidth": [2, 4, 8, 12],
    "commitWidth": [2, 4, 8, 12],
    "numROBEntries": [32, 64, 128, 192, 256],
    "numIQEntries": [16, 32, 64, 96, 128],
    "LQEntries": [8, 16, 32, 64],
    "SQEntries": [8, 16, 32, 64],
    "branch_predictor": [
        "BiModeBP",
        "LocalBP",
        "TAGE",
        "TAGE_SC_L_64KB",
        "MultiperspectivePerceptron64KB",
        "TournamentBP"
    ]
}

def get_numeric_ranges_from_space(space):
    """Return dict of (min, max) for numeric parameters from param_space."""
    ranges = {}
    for k, vals in space.items():
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            arr = np.array(vals, dtype=float)
            ranges[k] = (float(arr.min()), float(arr.max()))
    return ranges

def build_feature_matrix(df, param_space, drop_cols=("exp_id",)):
    """
    Build normalized feature matrix:
    - numeric features scaled to [0,1] using param_space ranges
    - categorical features one-hot encoded and scaled so different category -> L1 contribution 1
    Returns (X, feature_names)
    """
    df = df.copy()
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    num_ranges = get_numeric_ranges_from_space(param_space)
    numeric_cols = [c for c in df.columns if c in num_ranges]
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    # Normalize numeric columns to [0,1]
    norm_numeric = []
    norm_names = []
    for col in numeric_cols:
        col_min, col_max = num_ranges[col]
        if col_max == col_min:
            scaled = np.zeros(len(df))
        else:
            scaled = (df[col].astype(float) - col_min) / (col_max - col_min)
            scaled = np.clip(scaled, 0.0, 1.0)
        norm_numeric.append(scaled.to_numpy().reshape(-1, 1))
        norm_names.append(col)

    # Handle categorical columns (one-hot)
    cat_feature_parts = []
    cat_names = []
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_matrix = ohe.fit_transform(df[cat_cols].astype(str))
        categories = ohe.categories_
        start = 0
        scaled_blocks = []
        for i, cats in enumerate(categories):
            k = len(cats)
            block = cat_matrix[:, start:start+k] * 0.5  # scale to make diff=1
            scaled_blocks.append(block)
            col_name = cat_cols[i]
            for cat in cats:
                cat_names.append(f"{col_name}={cat}")
            start += k
        if scaled_blocks:
            cat_feature_parts = scaled_blocks

    parts = []
    if norm_numeric:
        parts.extend(norm_numeric)
    if cat_feature_parts:
        parts.extend(cat_feature_parts)
    if not parts:
        raise ValueError("No features found after processing.")
    X = np.hstack(parts)
    feature_names = norm_names + cat_names
    return X, feature_names

def pair_similarity_counts(X, s_thresh=0.9):
    """
    Compute mean similarity per parameter between pairs (1 - normalized L1 diff).
    Return similar_pairs count and total pairs.
    """
    n = X.shape[0]
    total_pairs = n * (n - 1) // 2

    # Compute mean per-dim difference
    manhattan = pairwise_distances(X, metric="manhattan")
    mean_diff = manhattan / X.shape[1]

    # Convert to similarity
    mean_sim = 1.0 - mean_diff
    tri_idx = np.triu_indices(n, k=1)
    vals = mean_sim[tri_idx]
    similar_pairs = int(np.sum(vals >= s_thresh))
    return similar_pairs, int(total_pairs), vals

def summarize(csv_file, s_thresh=0.9):
    df = pd.read_csv(csv_file)
    X, feat_names = build_feature_matrix(df, param_space, drop_cols=("exp_id",))
    similar_pairs, total_pairs, vals = pair_similarity_counts(X, s_thresh=s_thresh)
    pct = similar_pairs / total_pairs * 100 if total_pairs > 0 else 0.0

    # Convert values and threshold to percentage scale
    vals_pct = vals * 100
    s_thresh_pct = s_thresh * 100

    print(f"File: {csv_file}")
    print(f"Datapoints: {X.shape[0]}")
    print(f"Features used: {len(feat_names)}")
    print(f"Total pairs compared: {total_pairs}")
    print(f"Pairs with mean-per-parameter similarity >= {s_thresh_pct:.1f}%: {similar_pairs}")
    print(f"Similarity percentage: {pct:.6f}%")

    print("\nPairwise mean-similarity stats (mean-per-dim, in %):")
    print(f"min: {vals_pct.min():.2f}%, 25%: {np.percentile(vals_pct,25):.2f}%, "
          f"median: {np.median(vals_pct):.2f}%, 75%: {np.percentile(vals_pct,75):.2f}%, "
          f"max: {vals_pct.max():.2f}%")

if __name__ == "__main__":
    csv_file = "arch_sweep_dataset2.csv"
    s_thresh = 0.8  # similarity threshold (e.g., 0.9 = 90%)
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    summarize(csv_file, s_thresh=s_thresh)
