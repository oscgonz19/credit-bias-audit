# Model Card: Credit Risk Classification

## Model Details

- **Model Type:** Logistic Regression (baseline)
- **Framework:** scikit-learn
- **Version:** 1.0
- **Date:** 2024
- **License:** MIT

## Intended Use

### Primary Intended Uses
- Educational demonstration of fairness auditing in credit risk models
- Research on bias detection and mitigation techniques
- Portfolio demonstration of responsible ML practices

### Primary Intended Users
- Data scientists learning about ML fairness
- Researchers studying algorithmic bias
- Students in responsible AI courses

### Out-of-Scope Uses
- **NOT intended for production credit decisions**
- Should not be used for actual loan approval/denial
- Not validated for regulatory compliance

## Training Data

### German Credit Dataset
- **Source:** UCI Machine Learning Repository
- **Size:** 1,000 instances
- **Features:** 20 attributes (demographic, financial)
- **Label:** Credit risk (good/bad)
- **Protected Attributes:** sex, age, foreign worker status

See `data_sheet.md` for detailed data documentation.

## Evaluation Data

- 20% holdout from training data
- Stratified split to maintain class balance
- Same preprocessing as training data

## Metrics

### Performance Metrics
| Metric | Description |
|--------|-------------|
| Accuracy | Overall prediction correctness |
| Balanced Accuracy | Average recall across classes |
| AUC | Area under ROC curve |
| F1 Score | Harmonic mean of precision and recall |

### Fairness Metrics
| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| Statistical Parity Difference | Difference in positive rates | 0 |
| Disparate Impact | Ratio of positive rates | 1.0 (acceptable: 0.8-1.25) |
| Equal Opportunity Difference | Difference in TPR | 0 |
| Average Odds Difference | Average of TPR and FPR differences | 0 |

## Ethical Considerations

### Known Limitations
1. **Historical Bias:** Training data reflects historical lending decisions that may embed societal biases
2. **Proxy Discrimination:** Features may correlate with protected attributes
3. **Single Protected Attribute:** Analysis focuses on one attribute at a time; intersectional effects not captured
4. **Sample Size:** Some subgroups have limited representation

### Risks and Harms
- Could perpetuate historical discrimination if deployed
- May have different error rates across demographic groups
- Post-processing mitigations may not generalize to new populations

### Mitigation Strategies Implemented
1. **Reweighing (Pre-processing):** Adjusts sample weights to achieve statistical parity in training
2. **Equalized Odds (Post-processing):** Adjusts predictions to equalize TPR and FPR across groups

## Caveats and Recommendations

### Recommendations for Use
1. This model is for educational/demonstration purposes only
2. Always audit fairness before any deployment consideration
3. Consider multiple fairness definitions based on context
4. Engage stakeholders in defining acceptable trade-offs
5. Monitor for distributional shift in deployment

### Technical Caveats
- Results depend on protected attribute definition
- Different random seeds may yield different fairness metrics
- Post-processing requires access to protected attributes at inference

## Additional Information

### Contact
- Repository: [credit-bias-audit](https://github.com/oscgonz19/credit-bias-audit)

### References
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning.
- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination.
- Bellamy, R. K., et al. (2019). AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias.
