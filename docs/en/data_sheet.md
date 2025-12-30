# Datasheet: German Credit Dataset

## Motivation

### Purpose
The German Credit dataset was created to study credit risk classification. This datasheet documents its use in fairness auditing.

### Creators
- Original dataset: Prof. Dr. Hans Hofmann, Institut für Statistik und Ökonometrie, Universität Hamburg
- Donated to UCI ML Repository: 1994

### Funding
Academic research

## Composition

### Instances
- **Total instances:** 1,000
- **Instance type:** Individual credit applicants
- **Time period:** Historical (pre-1994 Germany)

### Features

#### Numerical (7)
| Feature | Description | Range |
|---------|-------------|-------|
| duration | Loan duration in months | 4-72 |
| credit_amount | Loan amount in DM | 250-18,424 |
| installment_commitment | Installment rate (% of income) | 1-4 |
| residence_since | Years at current residence | 1-4 |
| age | Age in years | 19-75 |
| existing_credits | Number of existing credits | 1-4 |
| num_dependents | Number of dependents | 1-2 |

#### Categorical (13)
| Feature | Categories |
|---------|------------|
| checking_status | <0, 0<=X<200, >=200, no checking |
| credit_history | no credits, all paid, existing paid, delayed, critical |
| purpose | new car, used car, furniture, radio/tv, appliances, repairs, education, vacation, retraining, business, other |
| savings_status | <100, 100<=X<500, 500<=X<1000, >=1000, no known savings |
| employment | unemployed, <1, 1<=X<4, 4<=X<7, >=7 |
| other_parties | none, co-applicant, guarantor |
| property_magnitude | real estate, life insurance, car, no known property |
| other_payment_plans | bank, stores, none |
| housing | rent, own, for free |
| job | unemp/unskilled non-res, unskilled resident, skilled, high qualif/mgmt |
| own_telephone | none, yes |
| foreign_worker | yes, no |
| marital_status | single, div/sep, mar/wid |

#### Protected Attributes
| Attribute | Values | Notes |
|-----------|--------|-------|
| sex | male, female | Derived from personal_status_sex |
| age_cat | young (<25), aged (>=25) | Derived from age |
| foreign_worker | yes, no | Original feature |

#### Label
- **credit-risk:** good (700), bad (300)
- Class imbalance: 70% good, 30% bad

### Missing Data
- No missing values in processed version
- Original encoding used specific codes for "unknown"

### Relationships
- Instances are independent credit applications
- No known duplicate individuals

### Splits
- Standard practice: 80% train, 20% test
- Stratified by label to maintain class balance

## Collection Process

### Data Collection
- Collected from German bank records
- Specific collection methodology not documented

### Sampling Strategy
- Not documented; appears to be convenience sample
- May not be representative of all credit applicants

### Time Frame
- Historical data predating 1994
- Single snapshot, not longitudinal

### Ethical Review
- No documented IRB or ethics review
- Common practice for historical datasets

## Preprocessing

### Original Preprocessing
- Symbolic attributes encoded with codes (A11, A12, etc.)
- Numerical attributes scaled

### Our Preprocessing
1. Decode symbolic attributes to readable labels
2. Extract sex from combined personal_status_sex field
3. Create binary age category (young/aged with cutoff at 25)
4. Binarize protected attributes and label
5. Create dummy variables for categorical features

## Uses

### Intended Uses
- Benchmark for credit scoring algorithms
- Fairness research and education
- ML bias auditing demonstrations

### Inappropriate Uses
- **Production credit decisions** (data is historical and German-specific)
- Generalizing to other countries/time periods
- Individual-level credit assessment

### Prior Uses
- Widely used in ML fairness literature
- Featured in AIF360 toolkit examples
- Common benchmark in academic papers

## Distribution

### License
- Public domain via UCI ML Repository
- No restrictions on use

### Access
- UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
- AIF360: Available through `aif360.sklearn.datasets`

## Maintenance

### Curator
- UCI ML Repository maintains the dataset
- No ongoing updates (static historical data)

### Versioning
- Single version; no planned updates

## Limitations and Biases

### Known Biases
1. **Historical bias:** Reflects lending practices of 1990s Germany
2. **Demographic imbalance:** More male applicants than female
3. **Geographic specificity:** German banking context
4. **Temporal obsolescence:** Economic conditions have changed

### Representation Gaps
- Limited representation of some demographic groups
- Foreign workers potentially underrepresented
- Young applicants (<25) are minority

### Measurement Issues
- Sex derived from marital status encoding (potential errors)
- Age threshold of 25 for "young" is arbitrary
- Credit risk label based on historical assessments (potentially biased)

## Ethical Considerations

### Sensitive Attributes
- Sex, age, foreign worker status are protected in many jurisdictions
- Using these features directly may violate anti-discrimination laws

### Privacy
- Dataset is anonymized
- No directly identifying information
- Re-identification risk considered low due to age of data

### Consent
- Original consent documentation not available
- Standard practice for public datasets of this era

## Additional Notes

### Citation
```
@misc{german_credit,
  author = {Hofmann, Hans},
  title = {Statlog (German Credit Data)},
  year = {1994},
  publisher = {UCI Machine Learning Repository},
  url = {https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)}
}
```

### Related Datasets
- Adult Census Income (similar fairness research use)
- COMPAS (criminal justice fairness)
- Bank Marketing (similar domain)
