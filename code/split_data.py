"""
split_data.py
-------------
ParticleSignal: Machine Learning-Based Classification of Signal and
Background Events in High-Energy Physics

Step 1 of the pipeline: split the raw Kaggle dataset into a training set
and a testing set.

Why a stratified split?
    Signal events are usually much rarer than background events. A plain
    random split could leave the test set with a slightly different
    signal/background ratio than the training set. Stratifying on the
    label keeps the class ratio identical in both sets, so the test score
    is a fair estimate of how the model behaves on unseen data.

Why a fixed random seed?
    So anyone who downloads the project can re-run this script and get
    exactly the same rows in training_data/ and testing_data/.

Usage:
    python split_data.py --input path/to/dataset.csv
    python split_data.py --input dataset.csv --test-size 0.2 --seed 42
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


# Column names the label is commonly stored under in HEP datasets.
# The first one found in the CSV is used, unless --label is given.
LABEL_CANDIDATES = ["Label", "label", "target", "Target", "class", "Class", "y"]


def find_label_column(df, user_choice=None):
    """Return the name of the column that holds signal/background labels."""
    if user_choice is not None:
        if user_choice not in df.columns:
            sys.exit(f"Error: column '{user_choice}' not found. "
                     f"Available columns: {list(df.columns)}")
        return user_choice

    for name in LABEL_CANDIDATES:
        if name in df.columns:
            return name

    sys.exit("Error: could not find a label column automatically. "
             f"Available columns: {list(df.columns)}\n"
             "Pass it explicitly, e.g. --label Label")


def describe(name, df, label_col):
    """Print row count and class balance for one split."""
    counts = df[label_col].value_counts()
    print(f"\n{name}: {len(df):,} rows")
    for cls, n in counts.items():
        print(f"    {str(cls):>10}: {n:>8,}  ({100 * n / len(df):5.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Stratified train/test split for the ParticleSignal project.")
    parser.add_argument("--input", required=True,
                        help="Path to the raw dataset CSV from Kaggle.")
    parser.add_argument("--label", default=None,
                        help="Name of the label column (auto-detected if omitted).")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of events held out for testing (default 0.2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for a reproducible split (default 42).")
    parser.add_argument("--out-dir", default="..",
                        help="Folder in which training_data/ and testing_data/ "
                             "are created (default: the project root).")
    args = parser.parse_args()

    if not 0.0 < args.test_size < 1.0:
        sys.exit("Error: --test-size must be between 0 and 1.")

    # ---- 1. Load ---------------------------------------------------------
    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} events with {df.shape[1]} columns.")

    label_col = find_label_column(df, args.label)
    print(f"Using '{label_col}' as the label column.")

    # Rows without a label cannot be used for supervised learning.
    missing = df[label_col].isna().sum()
    if missing:
        print(f"Dropping {missing:,} rows with no label.")
        df = df.dropna(subset=[label_col])

    # ---- 2. Split (stratified on the label) ------------------------------
    # The data is NOT modified here: no scaling, no imputation, no feature
    # removal. In this dataset -999.0 marks a quantity that is undefined for
    # that event (e.g. dijet mass when fewer than two jets were found); those
    # values are kept as-is and handled in the training code. All preprocessing is learned later from the training set
    # only, so no information from the test set leaks into training.
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df[label_col],
        random_state=args.seed,
        shuffle=True,
    )

    # ---- 3. Save ---------------------------------------------------------
    train_dir = os.path.join(args.out_dir, "training_data")
    test_dir = os.path.join(args.out_dir, "testing_data")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_path = os.path.join(train_dir, "train.csv")
    test_path = os.path.join(test_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # ---- 4. Report -------------------------------------------------------
    describe("Full dataset", df, label_col)
    describe("Training set", train_df, label_col)
    describe("Testing set ", test_df, label_col)

    # Sanity check: no event should appear in both sets.
    overlap = train_df.index.intersection(test_df.index)
    assert len(overlap) == 0, "Train and test sets overlap!"

    print(f"\nSaved training set -> {train_path}")
    print(f"Saved testing set  -> {test_path}")
    print(f"Split: {100 * (1 - args.test_size):.0f}% train / "
          f"{100 * args.test_size:.0f}% test, seed = {args.seed}")


if __name__ == "__main__":
    main()
