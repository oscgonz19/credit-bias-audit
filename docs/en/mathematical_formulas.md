# Fairness Metrics: Mathematical Foundations

## Notation

| Symbol | Description |
|--------|-------------|
| $Y$ | True label (ground truth) |
| $\hat{Y}$ | Predicted label |
| $A$ | Protected attribute (e.g., sex) |
| $a$ | Privileged group value |
| $\bar{a}$ | Unprivileged group value |
| $P(\cdot)$ | Probability |
| $TP, FP, TN, FN$ | Confusion matrix components |

---

## 1. Statistical Parity Difference (SPD)

### Definition

Statistical Parity Difference measures the difference in positive prediction rates between unprivileged and privileged groups.

$$\text{SPD} = P(\hat{Y} = 1 | A = \bar{a}) - P(\hat{Y} = 1 | A = a)$$

### Interpretation

| Value | Interpretation |
|-------|----------------|
| SPD = 0 | Perfect parity |
| SPD < 0 | Unprivileged group disadvantaged |
| SPD > 0 | Privileged group disadvantaged |
| \|SPD\| < 0.1 | Generally acceptable |

### Calculation

$$\text{SPD} = \frac{\sum_{i: A_i = \bar{a}} \hat{Y}_i}{|\{i: A_i = \bar{a}\}|} - \frac{\sum_{i: A_i = a} \hat{Y}_i}{|\{i: A_i = a\}|}$$

### Example

```
Group        Total    Predicted Positive    Rate
─────────────────────────────────────────────────
Male (a)     138      103                   74.6%
Female (ā)    62       41                   66.1%

SPD = 0.661 - 0.746 = -0.085
```

---

## 2. Disparate Impact (DI)

### Definition

Disparate Impact is the ratio of positive prediction rates between groups.

$$\text{DI} = \frac{P(\hat{Y} = 1 | A = \bar{a})}{P(\hat{Y} = 1 | A = a)}$$

### The 80% Rule

The Equal Employment Opportunity Commission (EEOC) established that:

$$\text{DI} < 0.8 \implies \text{Evidence of adverse impact}$$

### Interpretation

| Value | Interpretation |
|-------|----------------|
| DI = 1.0 | Perfect parity |
| DI ∈ [0.8, 1.25] | Generally acceptable |
| DI < 0.8 | Adverse impact against unprivileged |
| DI > 1.25 | Adverse impact against privileged |

### Calculation

$$\text{DI} = \frac{\frac{n_{\bar{a},+}}{n_{\bar{a}}}}{\frac{n_{a,+}}{n_a}}$$

Where:
- $n_{\bar{a},+}$ = Positive predictions for unprivileged group
- $n_{\bar{a}}$ = Total in unprivileged group
- $n_{a,+}$ = Positive predictions for privileged group
- $n_a$ = Total in privileged group

### Example

```
DI = 0.661 / 0.746 = 0.886

Since 0.886 > 0.8, the 80% rule is satisfied.
```

---

## 3. Equal Opportunity Difference (EOD)

### Definition

Equal Opportunity Difference measures the difference in True Positive Rates (TPR) between groups, focusing only on qualified individuals (Y=1).

$$\text{EOD} = P(\hat{Y} = 1 | Y = 1, A = \bar{a}) - P(\hat{Y} = 1 | Y = 1, A = a)$$

Equivalently:

$$\text{EOD} = \text{TPR}_{\bar{a}} - \text{TPR}_a$$

### True Positive Rate

$$\text{TPR}_g = \frac{TP_g}{TP_g + FN_g} = \frac{TP_g}{P_g}$$

Where $P_g$ is the number of actual positives in group $g$.

### Interpretation

| Value | Interpretation |
|-------|----------------|
| EOD = 0 | Equal opportunity |
| EOD < 0 | Qualified unprivileged individuals less likely to be approved |
| EOD > 0 | Qualified unprivileged individuals more likely to be approved |

### Example

```
Group        True Positives    Actual Positives    TPR
────────────────────────────────────────────────────────
Male (a)          89               100            89.0%
Female (ā)        34                44            77.3%

EOD = 0.773 - 0.890 = -0.117
```

---

## 4. Average Odds Difference (AOD)

### Definition

Average Odds Difference combines both TPR and FPR differences, measuring overall error rate disparity.

$$\text{AOD} = \frac{1}{2}\left[(\text{FPR}_{\bar{a}} - \text{FPR}_a) + (\text{TPR}_{\bar{a}} - \text{TPR}_a)\right]$$

### False Positive Rate

$$\text{FPR}_g = \frac{FP_g}{FP_g + TN_g} = \frac{FP_g}{N_g}$$

Where $N_g$ is the number of actual negatives in group $g$.

### Interpretation

| Value | Interpretation |
|-------|----------------|
| AOD = 0 | Equalized odds |
| AOD ≠ 0 | Systematic difference in error rates |

### Relationship to Equalized Odds

The Equalized Odds constraint requires:

$$\text{TPR}_{\bar{a}} = \text{TPR}_a \quad \text{AND} \quad \text{FPR}_{\bar{a}} = \text{FPR}_a$$

Which implies AOD = 0.

### Example

```
Group        TPR      FPR
─────────────────────────
Male (a)    89.0%    35.0%
Female (ā)  77.3%    27.8%

AOD = 0.5 × [(0.278 - 0.350) + (0.773 - 0.890)]
    = 0.5 × [-0.072 + (-0.117)]
    = 0.5 × (-0.189)
    = -0.0945
```

---

## 5. Reweighing Algorithm

### Objective

Transform sample weights to achieve statistical parity in training data.

### Weight Formula

For each sample $(x_i, y_i)$ with protected attribute $a_i$:

$$w_i = \frac{P(Y = y_i) \cdot P(A = a_i)}{P(Y = y_i, A = a_i)}$$

### Derivation

We want:

$$P_w(\hat{Y} = 1 | A = a) = P_w(\hat{Y} = 1 | A = \bar{a})$$

Using Bayes' theorem and the independence assumption under reweighing:

$$w(a, y) = \frac{P(Y = y) \cdot P(A = a)}{P(Y = y, A = a)} = \frac{n \cdot n_y \cdot n_a}{n^2 \cdot n_{y,a}} = \frac{n_y \cdot n_a}{n \cdot n_{y,a}}$$

### Weight Calculation Example

```
Total samples: n = 800
Good credit:   n_good = 560
Bad credit:    n_bad = 240
Male:          n_male = 552
Female:        n_female = 248

Observed counts:
- Male, Good:    n_m,g = 400
- Male, Bad:     n_m,b = 152
- Female, Good:  n_f,g = 160
- Female, Bad:   n_f,b = 88

Weights:
w(male, good)   = (560 × 552) / (800 × 400) = 0.966
w(male, bad)    = (240 × 552) / (800 × 152) = 1.089
w(female, good) = (560 × 248) / (800 × 160) = 1.085
w(female, bad)  = (240 × 248) / (800 × 88)  = 0.845
```

---

## 6. Equalized Odds Post-processing

### Optimization Problem

Given base classifier predictions, find adjusted predictions that satisfy:

**Objective:**
$$\min \sum_i \mathcal{L}(\tilde{Y}_i, \hat{Y}_i)$$

**Subject to:**
$$P(\tilde{Y} = 1 | Y = y, A = a) = P(\tilde{Y} = 1 | Y = y, A = \bar{a}) \quad \forall y \in \{0, 1\}$$

### Linear Programming Formulation

Decision variables:
- $p_{a,0}$: Probability of flipping 0→1 for privileged group
- $p_{a,1}$: Probability of flipping 1→0 for privileged group
- $p_{\bar{a},0}$: Probability of flipping 0→1 for unprivileged group
- $p_{\bar{a},1}$: Probability of flipping 1→0 for unprivileged group

The adjusted TPR for group $g$ becomes:

$$\widetilde{\text{TPR}}_g = \text{TPR}_g \cdot (1 - p_{g,1}) + (1 - \text{TPR}_g) \cdot p_{g,0}$$

Similarly for FPR:

$$\widetilde{\text{FPR}}_g = \text{FPR}_g \cdot (1 - p_{g,1}) + (1 - \text{FPR}_g) \cdot p_{g,0}$$

### Constraints

Equalized TPR:
$$\widetilde{\text{TPR}}_a = \widetilde{\text{TPR}}_{\bar{a}}$$

Equalized FPR:
$$\widetilde{\text{FPR}}_a = \widetilde{\text{FPR}}_{\bar{a}}$$

---

## 7. Performance Metrics

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Balanced Accuracy

$$\text{Balanced Accuracy} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right) = \frac{\text{TPR} + \text{TNR}}{2}$$

### Area Under ROC Curve (AUC)

$$\text{AUC} = P(\hat{Y}_{i^+} > \hat{Y}_{i^-})$$

Where $i^+$ is a randomly chosen positive sample and $i^-$ is a randomly chosen negative sample.

### F1 Score

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

---

## Summary Table

| Metric | Formula | Ideal | Range |
|--------|---------|-------|-------|
| SPD | $P(\hat{Y}=1\|A=\bar{a}) - P(\hat{Y}=1\|A=a)$ | 0 | [-1, 1] |
| DI | $\frac{P(\hat{Y}=1\|A=\bar{a})}{P(\hat{Y}=1\|A=a)}$ | 1 | [0, ∞) |
| EOD | $\text{TPR}_{\bar{a}} - \text{TPR}_a$ | 0 | [-1, 1] |
| AOD | $\frac{1}{2}[(\text{FPR}_{\bar{a}}-\text{FPR}_a)+(\text{TPR}_{\bar{a}}-\text{TPR}_a)]$ | 0 | [-1, 1] |

---

## References

1. Hardt, M., Price, E., & Srebro, N. (2016). *Equality of opportunity in supervised learning*. NeurIPS.
2. Kamiran, F., & Calders, T. (2012). *Data preprocessing techniques for classification without discrimination*. Knowledge and Information Systems.
3. Feldman, M., et al. (2015). *Certifying and removing disparate impact*. KDD.
4. Chouldechova, A. (2017). *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments*. Big Data.

---

*For implementation, see Technical Appendix*
