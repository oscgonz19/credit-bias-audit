"""Data loading, preprocessing, and splitting utilities."""

from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset


# Protected attribute configurations
PROTECTED_ATTR_CONFIG = {
    "german": {
        "sex": {"privileged": 1, "unprivileged": 0, "map": {"male": 1, "female": 0}},
        "age": {"privileged": 1, "unprivileged": 0, "map": {"aged": 1, "young": 0}},
        "foreign_worker": {"privileged": 1, "unprivileged": 0, "map": {"no": 1, "yes": 0}},
    },
    "adult": {
        "sex": {"privileged": 1, "unprivileged": 0, "map": {"Male": 1, "Female": 0}},
        "race": {"privileged": 1, "unprivileged": 0, "map": {"White": 1, "Black": 0}},
    },
}


def load_german_credit(data_path: Path) -> pd.DataFrame:
    """Load German Credit dataset from CSV.

    Args:
        data_path: Path to the german_credit.csv file.

    Returns:
        Raw DataFrame with original column names.
    """
    df = pd.read_csv(data_path)
    return df


def load_adult_census(data_path: Path) -> pd.DataFrame:
    """Load Adult Census Income dataset from CSV.

    Args:
        data_path: Path to the adult_data.csv file.

    Returns:
        Raw DataFrame with original column names.
    """
    df = pd.read_csv(data_path)
    # Standardize column names
    df = df.rename(columns={
        "education-num": "education_num",
        "marital-status": "marital_status",
        "capital-gain": "capital_gain",
        "capital-loss": "capital_loss",
        "hours-per-week": "hours_per_week",
        "native-country": "native_country",
    })
    return df


def preprocess_german(
    df: pd.DataFrame,
    protected_attr: str = "sex",
    label_col: str = "credit-risk",
) -> Tuple[pd.DataFrame, str, str]:
    """Preprocess German Credit dataset for fairness analysis.

    Args:
        df: Raw DataFrame.
        protected_attr: Protected attribute to analyze ('sex', 'age', 'foreign_worker').
        label_col: Name of the label column.

    Returns:
        Tuple of (processed DataFrame, protected attribute column name, label column name).
    """
    df = df.copy()

    # Binarize protected attributes
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"male": 1, "female": 0})
    if "age_cat" in df.columns:
        df["age_cat"] = df["age_cat"].map({"aged": 1, "young": 0})
    elif "age" in df.columns and df["age"].dtype == object:
        # Handle if age is categorical
        df["age_cat"] = df["age"].map({"aged": 1, "young": 0})
    if "foreign_worker" in df.columns:
        df["foreign_worker"] = df["foreign_worker"].map({"no": 1, "yes": 0})

    # Binarize label
    if label_col in df.columns:
        if df[label_col].dtype == object:
            df[label_col] = df[label_col].map({"good": 1, "bad": 0})

    # Map protected_attr name to actual column
    attr_col = protected_attr if protected_attr != "age" else "age_cat"

    return df, attr_col, label_col


def preprocess_adult(
    df: pd.DataFrame,
    protected_attr: str = "sex",
    label_col: str = "income",
) -> Tuple[pd.DataFrame, str, str]:
    """Preprocess Adult Census dataset for fairness analysis.

    Args:
        df: Raw DataFrame.
        protected_attr: Protected attribute to analyze ('sex', 'race').
        label_col: Name of the label column.

    Returns:
        Tuple of (processed DataFrame, protected attribute column name, label column name).
    """
    df = df.copy()

    # Binarize protected attributes
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    if "race" in df.columns:
        # Simplify race to binary (White vs non-White for fairness analysis)
        df["race"] = (df["race"] == "White").astype(int)

    # Binarize label
    if label_col in df.columns:
        df[label_col] = df[label_col].map({">50K": 1, "<=50K": 0})

    return df, protected_attr, label_col


def create_feature_matrix(
    df: pd.DataFrame,
    protected_attr: str,
    label_col: str,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create feature matrix with dummy encoding for categorical variables.

    Args:
        df: Preprocessed DataFrame.
        protected_attr: Protected attribute column name.
        label_col: Label column name.
        drop_cols: Additional columns to drop from features.

    Returns:
        Tuple of (X features, y labels, protected attribute series).
    """
    df = df.copy()

    # Store protected attribute and label
    protected_series = df[protected_attr].copy()
    y = df[label_col].copy()

    # Remove label from features
    X = df.drop(columns=[label_col])

    # Drop specified columns
    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    # Get categorical columns (excluding protected attribute)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Create dummies for categorical columns
    for col in cat_cols:
        if col != protected_attr:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X, dummies], axis=1).drop(columns=[col])

    return X, y, protected_series


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    Args:
        X: Feature matrix.
        y: Labels.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def create_aif360_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    protected_attr: str,
    label_name: str = "label",
    favorable_label: int = 1,
    unfavorable_label: int = 0,
) -> BinaryLabelDataset:
    """Create AIF360 BinaryLabelDataset from pandas objects.

    Args:
        X: Feature matrix (must include protected attribute).
        y: Labels.
        protected_attr: Name of protected attribute column.
        label_name: Name to use for label in dataset.
        favorable_label: Value representing favorable outcome.
        unfavorable_label: Value representing unfavorable outcome.

    Returns:
        AIF360 BinaryLabelDataset.
    """
    df = X.copy()
    df[label_name] = y.values

    return BinaryLabelDataset(
        df=df,
        label_names=[label_name],
        protected_attribute_names=[protected_attr],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
    )


def get_privileged_groups(
    dataset: str,
    protected_attr: str,
) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    """Get privileged and unprivileged group definitions.

    Args:
        dataset: Dataset name ('german' or 'adult').
        protected_attr: Protected attribute name.

    Returns:
        Tuple of (privileged_groups, unprivileged_groups) for AIF360.
    """
    config = PROTECTED_ATTR_CONFIG.get(dataset, {}).get(protected_attr, {})

    attr_col = protected_attr if protected_attr != "age" else "age_cat"

    privileged = config.get("privileged", 1)
    unprivileged = config.get("unprivileged", 0)

    return [{attr_col: privileged}], [{attr_col: unprivileged}]


def load_and_prepare_data(
    dataset: str,
    data_path: Path,
    protected_attr: str,
    test_size: float = 0.2,
    random_state: int = 42,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """End-to-end data loading and preparation pipeline.

    Args:
        dataset: Dataset name ('german' or 'adult').
        data_path: Path to the CSV file.
        protected_attr: Protected attribute to analyze.
        test_size: Proportion for test set.
        random_state: Random seed.
        sample_size: Optional limit on number of rows (for CI/testing).

    Returns:
        Dictionary with all prepared data objects.
    """
    # Load raw data
    if dataset == "german":
        df = load_german_credit(data_path)
        df, attr_col, label_col = preprocess_german(df, protected_attr)
    elif dataset == "adult":
        df = load_adult_census(data_path)
        df, attr_col, label_col = preprocess_adult(df, protected_attr)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    # Create feature matrix
    X, y, protected = create_feature_matrix(df, attr_col, label_col)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    # Get group definitions
    privileged_groups, unprivileged_groups = get_privileged_groups(dataset, protected_attr)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "protected_attr": attr_col,
        "label_col": label_col,
        "privileged_groups": privileged_groups,
        "unprivileged_groups": unprivileged_groups,
        "dataset_name": dataset,
    }
