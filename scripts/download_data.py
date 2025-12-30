#!/usr/bin/env python
"""Download and prepare datasets for credit bias audit."""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

import pandas as pd


# Dataset URLs
GERMAN_CREDIT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
GERMAN_CREDIT_NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc"

# Adult dataset from UCI
ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# Column definitions
GERMAN_COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status_sex",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
    "own_telephone", "foreign_worker", "credit-risk"
]

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from URL.

    Args:
        url: Source URL.
        dest_path: Destination file path.

    Returns:
        True if successful, False otherwise.
    """
    try:
        print(f"Downloading {url}...")
        urlretrieve(url, dest_path)
        print(f"  Saved to {dest_path}")
        return True
    except URLError as e:
        print(f"  Error downloading: {e}")
        return False


def process_german_credit(raw_path: Path, output_path: Path) -> None:
    """Process German Credit raw data into usable format.

    Args:
        raw_path: Path to raw german.data file.
        output_path: Path for processed CSV.
    """
    print("Processing German Credit dataset...")

    # Read space-separated file
    df = pd.read_csv(raw_path, sep=" ", header=None, names=GERMAN_COLUMNS)

    # Decode categorical values
    checking_status_map = {
        "A11": "<0", "A12": "0<=X<200", "A13": ">=200", "A14": "no checking"
    }
    credit_history_map = {
        "A30": "no credits/all paid", "A31": "all paid",
        "A32": "existing paid", "A33": "delayed previously",
        "A34": "critical/other existing credit"
    }
    purpose_map = {
        "A40": "new car", "A41": "used car", "A42": "furniture/equipment",
        "A43": "radio/tv", "A44": "domestic appliance", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining",
        "A49": "business", "A410": "other"
    }
    savings_status_map = {
        "A61": "<100", "A62": "100<=X<500", "A63": "500<=X<1000",
        "A64": ">=1000", "A65": "no known savings"
    }
    employment_map = {
        "A71": "unemployed", "A72": "<1", "A73": "1<=X<4",
        "A74": "4<=X<7", "A75": ">=7"
    }
    personal_status_sex_map = {
        "A91": "male:divorced/separated", "A92": "female:divorced/separated/married",
        "A93": "male:single", "A94": "male:married/widowed",
        "A95": "female:single"
    }
    other_parties_map = {
        "A101": "none", "A102": "co applicant", "A103": "guarantor"
    }
    property_magnitude_map = {
        "A121": "real estate", "A122": "life insurance",
        "A123": "car", "A124": "no known property"
    }
    other_payment_plans_map = {
        "A141": "bank", "A142": "stores", "A143": "none"
    }
    housing_map = {
        "A151": "rent", "A152": "own", "A153": "for free"
    }
    job_map = {
        "A171": "unemp/unskilled non res", "A172": "unskilled resident",
        "A173": "skilled", "A174": "high qualif/self emp/mgmt"
    }
    telephone_map = {
        "A191": "none", "A192": "yes"
    }
    foreign_worker_map = {
        "A201": "yes", "A202": "no"
    }

    # Apply mappings
    df["checking_status"] = df["checking_status"].map(checking_status_map)
    df["credit_history"] = df["credit_history"].map(credit_history_map)
    df["purpose"] = df["purpose"].map(purpose_map)
    df["savings_status"] = df["savings_status"].map(savings_status_map)
    df["employment"] = df["employment"].map(employment_map)
    df["other_parties"] = df["other_parties"].map(other_parties_map)
    df["property_magnitude"] = df["property_magnitude"].map(property_magnitude_map)
    df["other_payment_plans"] = df["other_payment_plans"].map(other_payment_plans_map)
    df["housing"] = df["housing"].map(housing_map)
    df["job"] = df["job"].map(job_map)
    df["own_telephone"] = df["own_telephone"].map(telephone_map)
    df["foreign_worker"] = df["foreign_worker"].map(foreign_worker_map)

    # Map personal_status_sex first, then extract sex
    df["personal_status_sex"] = df["personal_status_sex"].map(personal_status_sex_map)
    df["sex"] = df["personal_status_sex"].apply(
        lambda x: "male" if str(x).startswith("male") else "female"
    )

    # Extract marital status
    def extract_marital(x):
        if pd.isna(x):
            return "unknown"
        if "single" in x:
            return "single"
        elif "divorced" in x or "separated" in x:
            return "div/sep"
        elif "married" in x or "widowed" in x:
            return "mar/wid"
        return "unknown"

    df["marital_status"] = df["personal_status_sex"].apply(extract_marital)
    df = df.drop(columns=["personal_status_sex"])

    # Create age category
    df["age_cat"] = df["age"].apply(lambda x: "aged" if x >= 25 else "young")

    # Convert credit risk (1=good, 2=bad in original)
    df["credit-risk"] = df["credit-risk"].map({1: "good", 2: "bad"})

    # Save
    df.to_csv(output_path, index=False)
    print(f"  Processed {len(df)} records -> {output_path}")


def process_adult_census(train_path: Path, test_path: Path, output_path: Path) -> None:
    """Process Adult Census raw data into usable format.

    Args:
        train_path: Path to adult.data file.
        test_path: Path to adult.test file.
        output_path: Path for processed CSV.
    """
    print("Processing Adult Census dataset...")

    # Read both files
    df_train = pd.read_csv(train_path, header=None, names=ADULT_COLUMNS, na_values=" ?", skipinitialspace=True)
    df_test = pd.read_csv(test_path, header=None, names=ADULT_COLUMNS, na_values=" ?", skipinitialspace=True, skiprows=1)

    # Combine
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Clean income column (test set has trailing period)
    df["income"] = df["income"].str.strip().str.rstrip(".")

    # Drop fnlwgt (sampling weight, not useful for our analysis)
    df = df.drop(columns=["fnlwgt"])

    # Drop rows with missing values
    df = df.dropna()

    # Save
    df.to_csv(output_path, index=False)
    print(f"  Processed {len(df)} records -> {output_path}")


def create_sample_data(input_path: Path, output_path: Path, n_samples: int = 500) -> None:
    """Create a small sample dataset for testing.

    Args:
        input_path: Path to full dataset.
        output_path: Path for sample dataset.
        n_samples: Number of samples to include.
    """
    print(f"Creating sample dataset ({n_samples} rows)...")
    df = pd.read_csv(input_path)
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)
    sample.to_csv(output_path, index=False)
    print(f"  Sample saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for credit bias audit")
    parser.add_argument(
        "--dataset",
        choices=["german", "adult", "all"],
        default="german",
        help="Dataset to download (default: german)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to store data (default: data/)"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Also create a small sample for CI testing"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Size of sample dataset (default: 500)"
    )

    args = parser.parse_args()

    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.data_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    datasets_to_download = []
    if args.dataset in ["german", "all"]:
        datasets_to_download.append("german")
    if args.dataset in ["adult", "all"]:
        datasets_to_download.append("adult")

    success = True

    for dataset in datasets_to_download:
        if dataset == "german":
            raw_file = raw_dir / "german.data"
            processed_file = args.data_dir / "german_credit.csv"

            if not download_file(GERMAN_CREDIT_URL, raw_file):
                success = False
                continue

            process_german_credit(raw_file, processed_file)

            if args.create_sample:
                sample_file = args.data_dir / "sample_german_credit.csv"
                create_sample_data(processed_file, sample_file, args.sample_size)

        elif dataset == "adult":
            train_file = raw_dir / "adult.data"
            test_file = raw_dir / "adult.test"
            processed_file = args.data_dir / "adult_data.csv"

            if not download_file(ADULT_TRAIN_URL, train_file):
                success = False
                continue
            if not download_file(ADULT_TEST_URL, test_file):
                success = False
                continue

            process_adult_census(train_file, test_file, processed_file)

            if args.create_sample:
                sample_file = args.data_dir / "sample_adult_data.csv"
                create_sample_data(processed_file, sample_file, args.sample_size)

    if success:
        print("\nData download and processing complete!")
    else:
        print("\nSome downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
