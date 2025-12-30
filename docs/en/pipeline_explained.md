# How the Credit Bias Audit Pipeline Works

## Pipeline Overview

The audit pipeline processes credit data through four main stages to evaluate and mitigate bias in credit risk predictions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CREDIT BIAS AUDIT PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STAGE 1          STAGE 2          STAGE 3          STAGE 4               │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐           │
│  │  DATA   │ ───▶ │  MODEL  │ ───▶ │ FAIRNESS│ ───▶ │ REPORT  │           │
│  │ PREP    │      │ TRAIN   │      │  EVAL   │      │  GEN    │           │
│  └─────────┘      └─────────┘      └─────────┘      └─────────┘           │
│       │                │                │                │                 │
│       ▼                ▼                ▼                ▼                 │
│   Load CSV         Baseline        Compute          Generate             │
│   Preprocess       Reweighed       Metrics          CSV + MD             │
│   Split            Eq Odds         Compare          Reports              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Data Preparation

### 1.1 Data Loading

```python
# Download from UCI Repository
python scripts/download_data.py --dataset german

# Load into pandas
df = pd.read_csv('data/german_credit.csv')
```

**Input:** Raw CSV file (1,000 credit applications)
**Output:** Pandas DataFrame

### 1.2 Feature Engineering

```
Raw Data                              Processed Features
─────────────────────────────────────────────────────────────────
checking_status: "no checking"    →   checking_no_checking: 1
                                      checking_<0: 0
                                      checking_0<=X<200: 0

sex: "male"                       →   sex: 1

age_cat: "aged"                   →   age_cat: 1

credit-risk: "good"               →   credit-risk: 1
```

### 1.3 Train/Test Split

```
┌─────────────────────────────────────────────────┐
│              Original Dataset (1000)            │
├─────────────────────────────────────────────────┤
│                                                 │
│   ┌─────────────────────┐  ┌────────────────┐  │
│   │   Training Set      │  │   Test Set     │  │
│   │      (800)          │  │    (200)       │  │
│   │                     │  │                │  │
│   │  Good: 560 (70%)    │  │ Good: 140 (70%)│  │
│   │  Bad:  240 (30%)    │  │ Bad:   60 (30%)│  │
│   └─────────────────────┘  └────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
                    Stratified Split
```

---

## Stage 2: Model Training

### 2.1 Baseline Model

```python
# Standard logistic regression
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)
```

**No mitigation applied** - This establishes the baseline for comparison.

### 2.2 Reweighing (Pre-processing Mitigation)

```
REWEIGHING ALGORITHM
────────────────────────────────────────────────────

Step 1: Calculate expected frequencies
┌──────────────────────────────────────────────────┐
│  For each (group, label) combination:            │
│  Expected = P(group) × P(label) × N              │
└──────────────────────────────────────────────────┘

Step 2: Calculate weights
┌──────────────────────────────────────────────────┐
│  weight = expected / observed                    │
│                                                  │
│  Example:                                        │
│  Male + Good:   weight = 490/483 = 1.015        │
│  Female + Good: weight = 210/217 = 0.968        │
└──────────────────────────────────────────────────┘

Step 3: Train with weights
┌──────────────────────────────────────────────────┐
│  model.fit(X_train, y_train,                     │
│            sample_weight=weights)                │
└──────────────────────────────────────────────────┘
```

**Visual representation of weight effect:**

```
Before Reweighing              After Reweighing
(Original distribution)        (Adjusted weights)

Male   ████████████████████    Male   ████████████████████
       70% good rate                  ~68% effective rate

Female █████████████            Female █████████████████
       66% good rate                  ~68% effective rate
                                      ↑
                               Weights adjust influence
```

### 2.3 Equalized Odds (Post-processing Mitigation)

```
EQUALIZED ODDS ALGORITHM
────────────────────────────────────────────────────

Step 1: Get baseline predictions
┌──────────────────────────────────────────────────┐
│  y_pred_baseline = model.predict(X_test)         │
└──────────────────────────────────────────────────┘

Step 2: Calculate group-specific error rates
┌──────────────────────────────────────────────────┐
│  TPR_male = TP_male / P_male                     │
│  TPR_female = TP_female / P_female               │
│  FPR_male = FP_male / N_male                     │
│  FPR_female = FP_female / N_female               │
└──────────────────────────────────────────────────┘

Step 3: Solve optimization (linear program)
┌──────────────────────────────────────────────────┐
│  Find adjustment probabilities that:             │
│  - Equalize TPR across groups                    │
│  - Equalize FPR across groups                    │
│  - Minimize accuracy loss                        │
└──────────────────────────────────────────────────┘

Step 4: Apply probabilistic adjustments
┌──────────────────────────────────────────────────┐
│  For each prediction:                            │
│  - With probability p: flip 0→1 or 1→0          │
│  - Probability depends on group membership       │
└──────────────────────────────────────────────────┘
```

---

## Stage 3: Fairness Evaluation

### 3.1 Metrics Computation Flow

```
                    Predictions + Ground Truth
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌───────────┐       ┌───────────┐       ┌───────────┐
    │Performance│       │ Fairness  │       │  Group    │
    │  Metrics  │       │  Metrics  │       │ Breakdown │
    └───────────┘       └───────────┘       └───────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    • Accuracy          • SPD               • TPR by group
    • Balanced Acc      • DI                • FPR by group
    • AUC               • EOD               • Accuracy by group
    • Precision         • AOD
    • Recall
    • F1
```

### 3.2 Fairness Metrics Calculation

```python
# For each model variant (baseline, reweighed, eq_odds):

protected_values = X_test['sex'].values

metrics = {
    'performance': compute_performance_metrics(y_test, y_pred, y_proba),
    'fairness': compute_fairness_metrics(y_test, y_pred, protected_values),
}
```

### 3.3 Metrics Comparison Table

```
┌─────────────────────────────────────────────────────────────────────┐
│                     METRICS COMPARISON                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Model        Accuracy    SPD      DI      EOD      AOD            │
│  ─────────────────────────────────────────────────────────         │
│  Baseline     70.5%      -0.083   0.889   -0.115   -0.045          │
│  Reweighed    70.5%      -0.045   0.938   -0.070   -0.010          │
│  Eq Odds      63.5%      -0.012   0.982    0.005   -0.010          │
│                                                                     │
│  ─────────────────────────────────────────────────────────         │
│  Best for:    Reweighed  Eq Odds  Eq Odds Eq Odds  Both            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 4: Report Generation

### 4.1 CSV Output

```csv
model,mitigation,category,metric,value
logreg,none,performance,accuracy,0.705
logreg,none,fairness,statistical_parity_difference,-0.083
logreg,reweighing,performance,accuracy,0.705
logreg,reweighing,fairness,statistical_parity_difference,-0.045
logreg,eq_odds,performance,accuracy,0.635
logreg,eq_odds,fairness,statistical_parity_difference,-0.012
```

### 4.2 Markdown Report Structure

```
reports/report.md
├── Header & Timestamp
├── Audit Configuration
├── Summary
├── Results Comparison Table
├── Detailed Analysis
│   ├── Baseline
│   ├── Reweighing
│   └── Equalized Odds
├── Trade-off Analysis
├── Limitations
└── Recommendations
```

---

## End-to-End Flow Example

```python
# 1. DATA PREPARATION
data = load_and_prepare_data(
    dataset='german',
    data_path='data/german_credit.csv',
    protected_attr='sex',
    random_state=42
)

# 2. BASELINE MODEL
model = train_baseline_model(data['X_train'], data['y_train'])
y_pred, y_proba = get_predictions(model, data['X_test'])

# 3. FAIRNESS EVALUATION
metrics_baseline = compute_all_metrics(
    data['y_test'], y_pred, y_proba,
    data['X_test']['sex'].values
)

# 4. REWEIGHING MITIGATION
weights = apply_reweighing(
    data['X_train'], data['y_train'], 'sex',
    data['privileged_groups'], data['unprivileged_groups']
)
model_rw = train_baseline_model(
    data['X_train'], data['y_train'],
    sample_weight=weights
)

# 5. EQUALIZED ODDS MITIGATION
y_pred_eq = apply_eq_odds_postprocessing(
    data['X_test'], data['y_test'], y_pred, 'sex',
    data['privileged_groups'], data['unprivileged_groups']
)

# 6. GENERATE REPORTS
generate_metrics_csv(results, 'reports/metrics.csv')
generate_markdown_report(results, config, 'reports/report.md')
```

---

## Pipeline Configuration

### Command Line Arguments

```bash
python scripts/run_audit.py \
    --dataset german \           # Dataset: german or adult
    --protected-attr sex \       # Protected attribute
    --model logreg \             # Model type
    --mitigation all \           # none, reweighing, eq_odds, all
    --seed 42 \                  # Random seed
    --sample-size 1000 \         # Optional: limit data size
    --out-dir reports/           # Output directory
```

### Configuration Matrix

| Parameter | Options | Default | Effect |
|-----------|---------|---------|--------|
| dataset | german, adult | german | Data source |
| protected-attr | sex, age, race | sex | Fairness analysis target |
| model | logreg, logreg_cv, rf | logreg | Classifier type |
| mitigation | none, reweighing, eq_odds, all | all | Strategies to apply |
| seed | int | 42 | Reproducibility |

---

## Pipeline Extensibility

### Adding New Datasets

```python
# In src/data.py
def load_new_dataset(data_path):
    df = pd.read_csv(data_path)
    # Dataset-specific preprocessing
    return df

# Add to PROTECTED_ATTR_CONFIG
PROTECTED_ATTR_CONFIG['new_dataset'] = {
    'protected_attr': {
        'privileged': 1,
        'unprivileged': 0,
        'map': {'group_a': 1, 'group_b': 0}
    }
}
```

### Adding New Mitigation Strategies

```python
# In src/mitigation.py
class NewMitigation:
    def __init__(self, privileged_groups, unprivileged_groups):
        # Initialize
        pass

    def fit(self, dataset):
        # Learn parameters
        pass

    def transform(self, dataset):
        # Apply mitigation
        pass

# Add to factory function
def get_mitigation_strategy(name, ...):
    if name == 'new_strategy':
        return NewMitigation(...)
```

---

*See Technical Appendix for implementation details.*
