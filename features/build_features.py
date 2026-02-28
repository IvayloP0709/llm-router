"""Loads the data from the generated CSV, builds features, and saves the resulting dataset."""

import sys
import os

# Make sure we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from features.extractor import extract_features

def main():
    # load the raw data 
    df = pd.read_csv("data/queries.csv")

    # extract features for every query
    feature_rows = [] 
    for i, row in df.iterrows():
        feats = extract_features(row["query"])
        feature_rows.append(feats)
        if (i + 1) % 25 == 0:
            print(f"Processed {round((i + 1)/len(df),1)} rows...")

    features_df = pd.DataFrame(feature_rows)
    print(f"Extracted features for {len(features_df.columns)} features: {list(features_df.columns)}.")

    # merge with the target label column
    result = pd.concat([features_df, df[["optimal_model"]]], axis=1)

    # save the dataset 
    result.to_csv("data/features.csv", index=False)
    print(f"\nSaved data/features.csv ({result.shape[0]} rows, {result.shape[1]} columns).")

    # sanity check
    print("\nSample of the resulting dataset:")
    print(result.head(3).to_string())
    print("\nNull check:")
    nulls = result.isnull().sum()
    if nulls.sum() == 0:
        print("No null values found.")
    else:
        print(nulls[nulls > 0])

if __name__ == "__main__":
    main()