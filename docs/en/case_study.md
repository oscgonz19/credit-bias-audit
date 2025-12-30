# Credit Bias Audit: A Complete Case Study in Responsible ML

## Overview

This case study demonstrates a comprehensive approach to auditing and mitigating algorithmic bias in credit risk models. We address a critical question facing financial institutions: **How can we build credit scoring systems that are both accurate and fair across demographic groups?**

---

## The Problem

### Business Context

Financial institutions rely on machine learning models to assess credit risk and make lending decisions. These models analyze applicant data to predict the likelihood of loan default. However, historical lending data often reflects societal biases, which can lead to:

- **Discriminatory outcomes** against protected groups (women, minorities, etc.)
- **Regulatory violations** under fair lending laws (ECOA, Fair Housing Act)
- **Reputational damage** and loss of customer trust
- **Legal liability** from disparate impact claims

### The Challenge

We need to:
1. Train a credit risk model that performs well overall
2. Measure fairness across protected demographic groups
3. Apply bias mitigation techniques
4. Quantify the trade-off between accuracy and fairness
5. Document decisions for regulatory compliance

---

## Dataset: German Credit

### Description

We use the **German Credit Dataset** from the UCI Machine Learning Repository, a classic benchmark for fairness research containing 1,000 credit applications from a German bank.

| Attribute | Description |
|-----------|-------------|
| **Samples** | 1,000 credit applications |
| **Features** | 20 attributes (demographic + financial) |
| **Label** | Credit risk: Good (70%) / Bad (30%) |
| **Protected Attributes** | Sex, Age, Foreign Worker Status |

### Key Features

**Numerical:**
- Loan duration (months)
- Credit amount (DM)
- Age (years)
- Number of existing credits

**Categorical:**
- Checking account status
- Credit history
- Purpose of loan
- Employment status
- Housing situation

### Data Distribution

```
Sex Distribution:
├── Male:   690 (69%)
└── Female: 310 (31%)

Credit Risk:
├── Good: 700 (70%)
└── Bad:  300 (30%)

Positive Rate by Sex (Ground Truth):
├── Male:   72.3% approved
└── Female: 66.8% approved
└── Gap:    5.5 percentage points
```

---

## Methodology

### 1. Baseline Model

We train a **Logistic Regression** classifier as our baseline:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)
```

**Why Logistic Regression?**
- Interpretable coefficients
- Probability outputs
- Industry standard for credit scoring
- Regulatory acceptance

### 2. Fairness Evaluation Framework

We evaluate four key fairness metrics:

| Metric | Question It Answers |
|--------|---------------------|
| **Statistical Parity Difference** | Are positive predictions equally distributed? |
| **Disparate Impact** | What's the ratio of positive rates? |
| **Equal Opportunity Difference** | Are qualified applicants treated equally? |
| **Average Odds Difference** | Are error rates balanced? |

### 3. Bias Mitigation Strategies

We implement two complementary approaches:

#### Pre-processing: Reweighing
- Adjusts training sample weights
- Achieves statistical parity without changing labels
- Applied before model training

#### Post-processing: Equalized Odds
- Adjusts prediction thresholds per group
- Equalizes true positive and false positive rates
- Applied after model training

---

## Results

### Performance Metrics

| Model | Accuracy | Balanced Acc | AUC | F1 |
|-------|----------|--------------|-----|-----|
| Baseline | 70.5% | 63.7% | 0.760 | 0.793 |
| + Reweighing | 70.5% | 64.2% | 0.756 | 0.792 |
| + Eq Odds | 63.5% | 57.7% | N/A | 0.735 |

### Fairness Metrics

| Model | SPD | DI | EOD | AOD |
|-------|-----|-----|-----|-----|
| Baseline | -0.083 | 0.889 | -0.115 | -0.045 |
| + Reweighing | -0.045 | 0.938 | -0.070 | -0.010 |
| + Eq Odds | -0.012 | 0.982 | 0.005 | -0.010 |

### Visual Summary

```
Fairness-Accuracy Trade-off

Accuracy  │ ● Baseline (70.5%)
    70% ──┼──●─Reweighing (70.5%)
          │
    65% ──┼────────────────────────● Eq Odds (63.5%)
          │
          └─────┴─────┴─────┴─────┴─────┴──────────
               0.08  0.06  0.04  0.02  0.00
                    |SPD| (closer to 0 = fairer)
```

---

## Key Findings

### 1. Baseline Model Shows Measurable Bias

The unmitigated model exhibits:
- **8.3% lower** positive prediction rate for women
- **Disparate Impact of 0.889** (below 0.8 triggers regulatory scrutiny)
- **11.5% lower** true positive rate for qualified female applicants

### 2. Reweighing Improves Fairness at No Accuracy Cost

Pre-processing mitigation achieved:
- **46% reduction** in Statistical Parity Difference
- **Disparate Impact improved** from 0.889 to 0.938
- **Zero accuracy loss** (70.5% maintained)
- Slight improvement in balanced accuracy

### 3. Equalized Odds Maximizes Fairness

Post-processing achieved near-perfect fairness:
- **SPD reduced to -0.012** (near zero)
- **Disparate Impact of 0.982** (near perfect parity)
- **EOD of 0.005** (virtually equal opportunity)
- **Trade-off:** 7% accuracy reduction

### 4. The Fairness-Accuracy Trade-off is Real

```
┌────────────────────────────────────────────────────┐
│  Strategy Comparison                               │
├────────────────────────────────────────────────────┤
│                                                    │
│  Reweighing:   ★★★★☆ Fairness  ★★★★★ Accuracy     │
│                Best balance for production         │
│                                                    │
│  Eq Odds:      ★★★★★ Fairness  ★★★☆☆ Accuracy     │
│                Maximum fairness when required      │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## Business Recommendations

### For Production Deployment

1. **Recommended: Reweighing**
   - Maintains model accuracy
   - Significantly improves fairness
   - No inference-time complexity
   - Easier to explain to regulators

2. **When to Use Equalized Odds**
   - Regulatory requirement for equal opportunity
   - Accepting accuracy trade-off
   - Protected attribute available at inference

### Monitoring Strategy

```
Ongoing Fairness Monitoring
├── Weekly: Calculate fairness metrics on predictions
├── Monthly: Compare to baseline thresholds
├── Quarterly: Full audit with fresh test data
└── Annually: Retrain with updated mitigation
```

### Regulatory Compliance

| Requirement | How We Address It |
|-------------|-------------------|
| Disparate Impact Analysis | DI ratio calculated and monitored |
| Model Documentation | Model card with fairness metrics |
| Adverse Action | Explainable rejection reasons |
| Fair Lending | Multiple fairness definitions evaluated |

---

## Limitations & Future Work

### Current Limitations

1. **Single Protected Attribute**: Analysis focuses on sex; intersectional analysis needed
2. **Binary Classification**: Real credit decisions may be more nuanced
3. **Historical Data**: German Credit dataset from 1990s may not reflect current patterns
4. **Sample Size**: 1,000 samples limits statistical power for subgroup analysis

### Future Enhancements

- [ ] Multi-attribute intersectional analysis (sex × age × nationality)
- [ ] In-processing methods (Adversarial Debiasing)
- [ ] Causal fairness approaches
- [ ] Individual fairness metrics
- [ ] Continuous monitoring dashboard

---

## Conclusion

This case study demonstrates that **algorithmic fairness is achievable without sacrificing model performance**. By implementing systematic bias auditing and mitigation:

- We identified measurable bias in the baseline model
- **Reweighing eliminated 46% of statistical disparity at zero accuracy cost**
- **Equalized Odds achieved near-perfect fairness** when maximum fairness is required
- We documented trade-offs for informed decision-making

The tools and methodology presented here provide a **reproducible framework** for responsible ML in high-stakes domains like credit scoring.

---

## References

1. Hardt, M., Price, E., & Srebro, N. (2016). *Equality of opportunity in supervised learning*. NeurIPS.
2. Kamiran, F., & Calders, T. (2012). *Data preprocessing techniques for classification without discrimination*. KAIS.
3. Bellamy, R.K., et al. (2019). *AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias*. IBM Journal of R&D.
4. Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*. fairmlbook.org.

---

*This case study is part of the [credit-bias-audit](https://github.com/your-username/credit-bias-audit) portfolio project demonstrating responsible ML practices.*
