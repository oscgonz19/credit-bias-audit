# Technical Appendix

## System Architecture

### Module Overview

```
src/
├── __init__.py          # Package initialization
├── data.py              # Data loading and preprocessing
├── models.py            # ML model wrappers
├── fairness_metrics.py  # Fairness metric computations
├── mitigation.py        # Bias mitigation algorithms
└── reporting.py         # Report generation utilities
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           run_audit.py                                  │
│                        (CLI Entry Point)                                │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│     data.py      │    │    models.py     │    │   reporting.py   │
├──────────────────┤    ├──────────────────┤    ├──────────────────┤
│ load_german()    │    │ CreditRiskModel  │    │ generate_csv()   │
│ preprocess()     │    │ train_baseline() │    │ generate_md()    │
│ split_data()     │    │ get_predictions()│    │ format_tables()  │
│ create_aif360()  │    └──────────────────┘    └──────────────────┘
└──────────────────┘              │
          │                       │
          ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│fairness_metrics.py│    │  mitigation.py   │
├──────────────────┤    ├──────────────────┤
│ compute_spd()    │    │ Reweighing       │
│ compute_di()     │    │ EqOdds           │
│ compute_eod()    │    │ apply_reweigh()  │
│ compute_aod()    │    │ apply_eq_odds()  │
└──────────────────┘    └──────────────────┘
```

---

## Data Pipeline

### 1. Data Loading (`data.py`)

```python
def load_german_credit(data_path: Path) -> pd.DataFrame:
    """Load German Credit dataset from CSV."""
    df = pd.read_csv(data_path)
    return df
```

### 2. Preprocessing Flow

```
Raw Data                    Preprocessed Data
┌─────────────────┐         ┌─────────────────┐
│ sex: "male"     │   ──▶   │ sex: 1          │
│ age_cat: "aged" │   ──▶   │ age_cat: 1      │
│ risk: "good"    │   ──▶   │ risk: 1         │
│ purpose: "car"  │   ──▶   │ purpose_car: 1  │
│                 │         │ purpose_tv: 0   │
└─────────────────┘         └─────────────────┘
```

### 3. Feature Engineering

| Original Feature | Transformation | Output |
|-----------------|----------------|--------|
| Categorical (sex, age) | Binary mapping | 0/1 |
| Categorical (purpose) | One-hot encoding | Multiple columns |
| Numerical (amount) | Passthrough | Original values |
| Label (credit-risk) | Binary mapping | 0=bad, 1=good |

### 4. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 80/20 split
    random_state=42,      # Reproducibility
    stratify=y            # Maintain class balance
)
```

---

## Model Implementation

### CreditRiskModel Class

```python
class CreditRiskModel:
    """Wrapper for credit risk classification models."""

    SUPPORTED_MODELS = ["logreg", "logreg_cv", "rf"]

    def __init__(self, model_type: str, random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()

    def fit(self, X, y, sample_weight=None):
        """Train model with optional sample weights."""
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """Return probability predictions."""
        return self.model.predict_proba(X)
```

### Model Configurations

| Model | Parameters | Use Case |
|-------|------------|----------|
| `logreg` | solver='liblinear', max_iter=1000 | Default baseline |
| `logreg_cv` | cv=5, solver='liblinear' | Cross-validated |
| `rf` | n_estimators=100, n_jobs=-1 | Non-linear relationships |

---

## Fairness Metrics Implementation

### Statistical Parity Difference

```python
def compute_statistical_parity_difference(
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> float:
    """
    SPD = P(Ŷ=1|unprivileged) - P(Ŷ=1|privileged)

    Range: [-1, 1]
    Ideal: 0
    """
    priv_mask = protected_attr == privileged_value
    unpriv_mask = ~priv_mask

    p_priv = np.mean(y_pred[priv_mask])
    p_unpriv = np.mean(y_pred[unpriv_mask])

    return p_unpriv - p_priv
```

### Disparate Impact

```python
def compute_disparate_impact(
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> float:
    """
    DI = P(Ŷ=1|unprivileged) / P(Ŷ=1|privileged)

    Range: [0, ∞)
    Ideal: 1.0
    80% Rule: DI < 0.8 indicates adverse impact
    """
    priv_mask = protected_attr == privileged_value
    unpriv_mask = ~priv_mask

    p_priv = np.mean(y_pred[priv_mask])
    p_unpriv = np.mean(y_pred[unpriv_mask])

    return p_unpriv / p_priv if p_priv > 0 else np.inf
```

### Equal Opportunity Difference

```python
def compute_equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> float:
    """
    EOD = TPR_unprivileged - TPR_privileged

    Only considers positive class (Y=1)
    Range: [-1, 1]
    Ideal: 0
    """
    priv_pos = (y_true == 1) & (protected_attr == privileged_value)
    unpriv_pos = (y_true == 1) & (protected_attr != privileged_value)

    tpr_priv = np.mean(y_pred[priv_pos])
    tpr_unpriv = np.mean(y_pred[unpriv_pos])

    return tpr_unpriv - tpr_priv
```

### Average Odds Difference

```python
def compute_average_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    privileged_value: int = 1,
) -> float:
    """
    AOD = 0.5 * [(FPR_u - FPR_p) + (TPR_u - TPR_p)]

    Combines both error types
    Range: [-1, 1]
    Ideal: 0
    """
    # Calculate TPR difference
    tpr_diff = compute_equal_opportunity_difference(
        y_true, y_pred, protected_attr, privileged_value
    )

    # Calculate FPR difference
    priv_neg = (y_true == 0) & (protected_attr == privileged_value)
    unpriv_neg = (y_true == 0) & (protected_attr != privileged_value)

    fpr_priv = np.mean(y_pred[priv_neg])
    fpr_unpriv = np.mean(y_pred[unpriv_neg])
    fpr_diff = fpr_unpriv - fpr_priv

    return 0.5 * (fpr_diff + tpr_diff)
```

---

## Mitigation Algorithms

### Reweighing (Pre-processing)

```python
class ReweighingMitigation:
    """
    Assigns weights to training samples to achieve statistical parity.

    Weight formula:
    w(X,Y) = P(Y) * P(G) / P(Y,G)

    Where:
    - G: Protected group
    - Y: Label
    """

    def __init__(self, privileged_groups, unprivileged_groups):
        self.reweigher = Reweighing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups
        )

    def get_weights(self, dataset):
        """Return sample weights for training."""
        transformed = self.reweigher.fit_transform(dataset)
        return transformed.instance_weights
```

**Weight Calculation Example:**

```
Group      Label   Count   Expected   Weight
───────────────────────────────────────────
Male       Good    483     490        1.015
Male       Bad     207     200        0.966
Female     Good    217     210        0.968
Female     Bad     93      100        1.075
```

### Equalized Odds (Post-processing)

```python
class EqOddsMitigation:
    """
    Adjusts predictions to equalize TPR and FPR across groups.

    Solves linear program:
    minimize: cost
    subject to: TPR_u = TPR_p, FPR_u = FPR_p
    """

    def __init__(self, privileged_groups, unprivileged_groups, seed=42):
        self.postprocessor = EqOddsPostprocessing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            seed=seed
        )

    def fit(self, dataset_true, dataset_pred):
        """Learn adjustment parameters from validation data."""
        self.postprocessor.fit(dataset_true, dataset_pred)
        return self

    def predict(self, dataset_pred):
        """Apply equalized odds adjustment."""
        return self.postprocessor.predict(dataset_pred)
```

---

## AIF360 Integration

### Dataset Conversion

```python
def create_aif360_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    protected_attr: str,
    label_name: str = "label"
) -> BinaryLabelDataset:
    """Convert pandas to AIF360 format."""
    df = X.copy()
    df[label_name] = y.values

    return BinaryLabelDataset(
        df=df,
        label_names=[label_name],
        protected_attribute_names=[protected_attr],
        favorable_label=1,
        unfavorable_label=0
    )
```

### Group Definitions

```python
# German Credit - Sex
privileged_groups = [{'sex': 1}]      # Male
unprivileged_groups = [{'sex': 0}]    # Female

# German Credit - Age
privileged_groups = [{'age_cat': 1}]  # Aged (>=25)
unprivileged_groups = [{'age_cat': 0}] # Young (<25)
```

---

## Testing Strategy

### Test Categories

| Category | Files | Description |
|----------|-------|-------------|
| Unit | `test_smoke.py` | Individual function tests |
| Integration | `test_smoke.py` | End-to-end pipeline tests |
| Smoke | CI workflow | Quick validation tests |

### Test Fixtures

```python
@pytest.fixture
def sample_german_data():
    """Create minimal sample data for testing."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        "sex": np.random.choice(["male", "female"], n_samples),
        "age_cat": np.random.choice(["aged", "young"], n_samples),
        "credit-risk": np.random.choice(["good", "bad"], n_samples),
        # ... other features
    })
```

### Key Test Cases

```python
def test_statistical_parity_difference():
    """Verify SPD calculation."""
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    protected = np.array([1, 1, 1, 0, 0, 0])

    spd = compute_statistical_parity_difference(y_pred, protected)

    # Priv: 2/3, Unpriv: 1/3 → SPD = -1/3
    assert abs(spd - (-1/3)) < 0.01

def test_reweighing_with_model():
    """Test model training with sample weights."""
    weights = apply_reweighing(X_train, y_train, ...)
    model = train_baseline_model(X_train, y_train, sample_weight=weights)

    assert model._is_fitted
    assert len(model.predict(X_test)) == len(X_test)
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Download sample data
        run: python scripts/download_data.py --create-sample

      - name: Run tests
        run: pytest tests/ -v

      - name: Run integration test
        run: python scripts/run_audit.py --sample-size 500
```

---

## Error Handling

### Common Issues and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: NaN in linprog` | Empty group in test data | Ensure all groups represented |
| `KeyError: protected_attr` | Missing column | Check preprocessing |
| `ZeroDivisionError` | No positive predictions | Check threshold |

### Defensive Programming

```python
def compute_disparate_impact(...):
    if p_priv == 0:
        return np.inf if p_unpriv > 0 else 1.0
    return p_unpriv / p_priv
```

---

## Performance Considerations

### Memory Usage

| Dataset Size | Memory (approx) |
|--------------|-----------------|
| 1,000 rows | ~50 MB |
| 10,000 rows | ~200 MB |
| 100,000 rows | ~1.5 GB |

### Runtime

| Operation | Time (1000 rows) |
|-----------|------------------|
| Data loading | <1s |
| Model training | ~2s |
| Metrics computation | <1s |
| Report generation | <1s |
| **Total** | **~5s** |

---

*See Mathematical Formulas for detailed derivations.*
