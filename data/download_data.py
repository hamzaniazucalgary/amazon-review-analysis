"""Download Amazon Reviews Polarity dataset."""

import argparse
import os
import sys


def download_huggingface(output_dir: str, fmt: str) -> None:
    """Download dataset from HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        sys.exit(1)

    print("Downloading Amazon Polarity dataset from HuggingFace...")
    dataset = load_dataset("amazon_polarity")

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    print(f"Train size: {len(train_ds):,} rows")
    print(f"Test size:  {len(test_ds):,} rows")

    assert len(train_ds) == 3_600_000, f"Expected 3.6M train rows, got {len(train_ds):,}"
    assert len(test_ds) == 400_000, f"Expected 400K test rows, got {len(test_ds):,}"

    # Rename HuggingFace columns to match our schema:
    #   HF: label (ClassLabel 0=neg,1=pos), title, content
    #   Ours: polarity (int 1=neg,2=pos), title, text
    # Must cast ClassLabel -> int before remapping, since ClassLabel enforces num_classes=2.
    from datasets import Value
    for split_name, ds in [("train", train_ds), ("test", test_ds)]:
        ds = ds.cast_column("label", Value("int32"))
        ds = ds.rename_column("content", "text")
        ds = ds.rename_column("label", "polarity")
        ds = ds.map(lambda x: {"polarity": x["polarity"] + 1})
        if split_name == "train":
            train_ds = ds
        else:
            test_ds = ds
    print("Columns renamed: label->polarity (0/1->1/2), content->text")

    if fmt == "parquet":
        train_path = os.path.join(output_dir, "train.parquet")
        test_path = os.path.join(output_dir, "test.parquet")
        train_ds.to_parquet(train_path)
        test_ds.to_parquet(test_path)
    else:
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        train_ds.to_csv(train_path, index=False)
        test_ds.to_csv(test_path, index=False)

    print(f"Saved to {train_path} and {test_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Amazon Reviews Polarity dataset")
    parser.add_argument("--source", choices=["huggingface", "kaggle"], default="huggingface",
                        help="Data source (default: huggingface)")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                        help="Output format (default: parquet)")
    parser.add_argument("--output-dir", default="data", help="Output directory (default: data)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.source == "huggingface":
        download_huggingface(args.output_dir, args.format)
    elif args.source == "kaggle":
        print("Kaggle download not yet implemented.")
        print("Download manually from: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews")
        print(f"Place train.csv and test.csv in {args.output_dir}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
