"""
Tile-to-WSI aggregation script for DFSPAIM.

Takes patch-level predictions (e.g. all_probs_{kernel_type}.csv produced by predict.py)
and aggregates them to whole-slide-image (WSI) or case-level predictions.

This script is additive: it does not modify train.py / predict.py / dataset.py.
"""

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate patch-level predictions to WSI/case level."
    )
    parser.add_argument(
        "--patch-csv",
        type=str,
        required=True,
        help="Path to patch prediction CSV. Expected columns: image_name, class_0, ..., class_4 (or 0,1,2,3,4).",
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        default=None,
        help="Optional CSV with columns image_name, slide_id. If provided, it overrides --slide-id-pattern.",
    )
    parser.add_argument(
        "--slide-id-pattern",
        type=str,
        default=None,
        help="Regex with a capture group to extract slide_id from image_name. Example: '^(\\d+)_RGB_'",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mean",
        choices=["mean", "max", "median", "majority"],
        help="Aggregation method for class probabilities / logits.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Output CSV path for WSI-level predictions.",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        default="./class_map.json",
        help="Path to class_map.json (used only to label the predicted class in the output).",
    )
    parser.add_argument(
        "--image-name-col",
        type=str,
        default="image_name",
        help="Column name for patch image identifiers.",
    )
    parser.add_argument(
        "--slide-id-col",
        type=str,
        default="slide_id",
        help="Column name for slide/case identifiers in mapping CSV and output.",
    )
    return parser.parse_args()


def load_class_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx_to_short = {
        int(k): v["short_code"] for k, v in data["canonical_map"].items()
    }
    return idx_to_short


def infer_class_columns(df):
    """Find the probability columns. Accepts either class_0..class_4 or 0..4."""
    candidates_a = [f"class_{i}" for i in range(5)]
    candidates_b = [str(i) for i in range(5)]
    if all(c in df.columns for c in candidates_a):
        return candidates_a
    if all(c in df.columns for c in candidates_b):
        return candidates_b
    raise ValueError(
        "Could not find class probability columns. Expected class_0..class_4 or 0..4."
    )


def extract_slide_id_default(image_name):
    """
    Default heuristic: try to extract a leading numeric case ID.
    Examples:
        '_18-03822_RGB_Extended0_256_1black.jpg' -> '18-03822'
        '18-03822_patch_001.jpg' -> '18-03822'
    Falls back to the first token split by '_'.
    """
    # Remove common image extensions
    name = image_name
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break

    # Try leading digits/dashes like 18-03822
    m = re.match(r"^_?([0-9]+(?:-[0-9]+)*)_", name)
    if m:
        return m.group(1)

    # Fallback: first underscore-separated token
    return name.split("_")[0]


def get_slide_ids(df, image_name_col, mapping_df=None, slide_id_col=None, pattern=None):
    if mapping_df is not None:
        if image_name_col not in mapping_df.columns:
            raise ValueError(f"mapping CSV must contain column '{image_name_col}'")
        if slide_id_col not in mapping_df.columns:
            raise ValueError(f"mapping CSV must contain column '{slide_id_col}'")
        merged = df.merge(mapping_df[[image_name_col, slide_id_col]], on=image_name_col, how="left")
        missing = merged[slide_id_col].isna().sum()
        if missing:
            raise ValueError(
                f"{missing} patch images could not be matched to a slide_id in the mapping CSV."
            )
        return merged[slide_id_col].values

    if pattern is not None:
        regex = re.compile(pattern)
        slide_ids = []
        for name in df[image_name_col]:
            m = regex.search(str(name))
            if not m:
                raise ValueError(f"Pattern '{pattern}' did not match image_name '{name}'")
            slide_ids.append(m.group(1) if m.groups() else m.group(0))
        return np.array(slide_ids)

    # Default heuristic
    return df[image_name_col].apply(extract_slide_id_default).values


def aggregate(df, slide_ids, class_cols, method):
    df = df.copy()
    df["__slide_id__"] = slide_ids

    probs = df[class_cols].values.astype(np.float64)
    df[class_cols] = probs

    if method in ("mean", "max", "median"):
        grouped = df.groupby("__slide_id__")[class_cols]
        if method == "mean":
            agg = grouped.mean()
        elif method == "max":
            agg = grouped.max()
        else:
            agg = grouped.median()
        counts = grouped.size().rename("n_patches")
        result = pd.concat([agg, counts], axis=1).reset_index().rename(
            columns={"__slide_id__": "slide_id"}
        )
    elif method == "majority":
        pred_per_patch = df[class_cols].values.argmax(axis=1)
        df["__pred__"] = pred_per_patch
        result = []
        for sid, sub in df.groupby("__slide_id__"):
            votes = np.bincount(sub["__pred__"].values, minlength=len(class_cols))
            majority_class = int(votes.argmax())
            probs_majority = np.zeros(len(class_cols))
            probs_majority[majority_class] = 1.0
            row = {"slide_id": sid, "n_patches": len(sub), "predicted_class": majority_class}
            for i, col in enumerate(class_cols):
                row[col] = probs_majority[i]
            result.append(row)
        result = pd.DataFrame(result)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    if "predicted_class" not in result.columns:
        result["predicted_class"] = result[class_cols].values.argmax(axis=1)

    return result


def main():
    args = parse_args()

    df = pd.read_csv(args.patch_csv)
    if args.image_name_col not in df.columns:
        raise ValueError(f"Column '{args.image_name_col}' not found in {args.patch_csv}")

    class_cols = infer_class_columns(df)

    mapping_df = None
    if args.mapping_csv:
        mapping_df = pd.read_csv(args.mapping_csv)

    slide_ids = get_slide_ids(
        df,
        args.image_name_col,
        mapping_df=mapping_df,
        slide_id_col=args.slide_id_col,
        pattern=args.slide_id_pattern,
    )

    result = aggregate(df, slide_ids, class_cols, args.method)

    # Add human-readable class labels
    try:
        idx_to_short = load_class_map(args.class_map)
        result["predicted_label"] = result["predicted_class"].map(
            lambda x: idx_to_short.get(int(x), "unknown")
        )
    except Exception:
        result["predicted_label"] = result["predicted_class"]

    # Rename slide_id column if requested
    result = result.rename(columns={"slide_id": args.slide_id_col})

    result.to_csv(args.output_csv, index=False)
    print(f"Aggregated {len(df)} patches into {len(result)} slides/cases.")
    print(f"Method: {args.method}")
    print(f"Output saved to: {args.output_csv}")
    print(result.head())


if __name__ == "__main__":
    main()
