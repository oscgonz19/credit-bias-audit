"""Smoke tests for credit bias audit pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    create_feature_matrix,
    get_privileged_groups,
    preprocess_german,
    split_data,
)
from src.fairness_metrics import (
    compute_disparate_impact,
    compute_fairness_metrics,
    compute_performance_metrics,
    compute_statistical_parity_difference,
)
from src.mitigation import apply_reweighing
from src.models import CreditRiskModel, get_predictions, train_baseline_model


@pytest.fixture
def sample_german_data():
    """Create minimal sample data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "sex": np.random.choice(["male", "female"], n_samples),
        "age_cat": np.random.choice(["aged", "young"], n_samples),
        "foreign_worker": np.random.choice(["yes", "no"], n_samples),
        "duration": np.random.randint(6, 72, n_samples).astype(float),
        "credit_amount": np.random.randint(500, 10000, n_samples).astype(float),
        "credit-risk": np.random.choice(["good", "bad"], n_samples, p=[0.7, 0.3]),
    }

    return pd.DataFrame(data)


@pytest.fixture
def processed_data(sample_german_data):
    """Preprocessed data ready for modeling."""
    df, attr_col, label_col = preprocess_german(sample_german_data, "sex")
    X, y, protected = create_feature_matrix(df, attr_col, label_col)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "protected_attr": attr_col,
    }


class TestDataPreprocessing:
    """Tests for data preprocessing functions."""

    def test_preprocess_german_binarizes_sex(self, sample_german_data):
        """Test that sex is properly binarized."""
        df, attr_col, label_col = preprocess_german(sample_german_data, "sex")

        assert df["sex"].isin([0, 1]).all()
        assert attr_col == "sex"

    def test_preprocess_german_binarizes_label(self, sample_german_data):
        """Test that credit-risk is properly binarized."""
        df, attr_col, label_col = preprocess_german(sample_german_data, "sex")

        assert df["credit-risk"].isin([0, 1]).all()

    def test_create_feature_matrix_separates_correctly(self, sample_german_data):
        """Test feature matrix creation."""
        df, attr_col, label_col = preprocess_german(sample_german_data, "sex")
        X, y, protected = create_feature_matrix(df, attr_col, label_col)

        assert label_col not in X.columns
        assert len(X) == len(y)
        assert len(protected) == len(y)

    def test_split_data_stratified(self, sample_german_data):
        """Test data splitting maintains class balance."""
        df, attr_col, label_col = preprocess_german(sample_german_data, "sex")
        X, y, _ = create_feature_matrix(df, attr_col, label_col)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)

        # Check proportions are similar
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.15

    def test_get_privileged_groups(self):
        """Test privileged group definitions."""
        priv, unpriv = get_privileged_groups("german", "sex")

        assert priv == [{"sex": 1}]
        assert unpriv == [{"sex": 0}]


class TestModels:
    """Tests for model functions."""

    def test_credit_risk_model_init(self):
        """Test model initialization."""
        model = CreditRiskModel(model_type="logreg", random_state=42)
        assert model.model_type == "logreg"
        assert not model._is_fitted

    def test_credit_risk_model_fit_predict(self, processed_data):
        """Test model training and prediction."""
        model = CreditRiskModel(model_type="logreg", random_state=42)
        model.fit(processed_data["X_train"], processed_data["y_train"])

        assert model._is_fitted

        predictions = model.predict(processed_data["X_test"])
        assert len(predictions) == len(processed_data["X_test"])
        assert set(predictions).issubset({0, 1})

    def test_train_baseline_model(self, processed_data):
        """Test baseline model training helper."""
        model = train_baseline_model(
            processed_data["X_train"],
            processed_data["y_train"],
            model_type="logreg",
        )

        assert model._is_fitted

    def test_get_predictions(self, processed_data):
        """Test prediction helper function."""
        model = train_baseline_model(
            processed_data["X_train"],
            processed_data["y_train"],
        )

        preds, proba = get_predictions(model, processed_data["X_test"])

        assert len(preds) == len(processed_data["X_test"])
        assert len(proba) == len(processed_data["X_test"])
        assert proba.min() >= 0 and proba.max() <= 1


class TestFairnessMetrics:
    """Tests for fairness metrics."""

    def test_compute_performance_metrics(self):
        """Test performance metrics computation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_proba = np.array([0.2, 0.6, 0.8, 0.9, 0.4])

        metrics = compute_performance_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_statistical_parity_difference(self):
        """Test SPD calculation."""
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        protected = np.array([1, 1, 1, 0, 0, 0])  # 1=privileged

        spd = compute_statistical_parity_difference(y_pred, protected, privileged_value=1)

        # Privileged: 2/3 positive, Unprivileged: 1/3 positive
        # SPD = 1/3 - 2/3 = -1/3
        assert abs(spd - (-1 / 3)) < 0.01

    def test_disparate_impact(self):
        """Test DI calculation."""
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        protected = np.array([1, 1, 1, 0, 0, 0])

        di = compute_disparate_impact(y_pred, protected, privileged_value=1)

        # DI = (1/3) / (2/3) = 0.5
        assert abs(di - 0.5) < 0.01

    def test_compute_fairness_metrics(self):
        """Test all fairness metrics computation."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        protected = np.array([1, 1, 1, 0, 0, 0])

        metrics = compute_fairness_metrics(y_true, y_pred, protected)

        assert "statistical_parity_difference" in metrics
        assert "disparate_impact" in metrics
        assert "equal_opportunity_difference" in metrics
        assert "average_odds_difference" in metrics


class TestMitigation:
    """Tests for mitigation strategies."""

    def test_reweighing_returns_weights(self, processed_data):
        """Test reweighing produces sample weights."""
        priv, unpriv = get_privileged_groups("german", "sex")

        weights = apply_reweighing(
            processed_data["X_train"],
            processed_data["y_train"],
            processed_data["protected_attr"],
            priv,
            unpriv,
        )

        assert len(weights) == len(processed_data["X_train"])
        assert weights.min() > 0

    def test_reweighing_with_model(self, processed_data):
        """Test training model with reweighing."""
        priv, unpriv = get_privileged_groups("german", "sex")

        weights = apply_reweighing(
            processed_data["X_train"],
            processed_data["y_train"],
            processed_data["protected_attr"],
            priv,
            unpriv,
        )

        model = train_baseline_model(
            processed_data["X_train"],
            processed_data["y_train"],
            sample_weight=weights,
        )

        assert model._is_fitted
        preds = model.predict(processed_data["X_test"])
        assert len(preds) == len(processed_data["X_test"])


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_runs(self, processed_data):
        """Test complete pipeline from data to metrics."""
        priv, unpriv = get_privileged_groups("german", "sex")

        # Train baseline
        model = train_baseline_model(
            processed_data["X_train"],
            processed_data["y_train"],
        )

        # Get predictions
        preds, proba = get_predictions(model, processed_data["X_test"])

        # Compute metrics
        protected_values = processed_data["X_test"]["sex"].values
        perf_metrics = compute_performance_metrics(processed_data["y_test"].values, preds, proba)
        fair_metrics = compute_fairness_metrics(
            processed_data["y_test"].values, preds, protected_values
        )

        # Check all metrics are present
        assert "accuracy" in perf_metrics
        assert "statistical_parity_difference" in fair_metrics
