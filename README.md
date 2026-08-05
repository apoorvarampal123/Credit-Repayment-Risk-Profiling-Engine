# Credit Repayment Risk Profiling Engine

A machine learning pipeline that estimates a bank customer's likelihood of missing their next credit card payment and sorts them into risk tiers (Very Low to Very High Risk), moving a credit team beyond binary accept/reject decisions toward proactive risk triage.

## The Problem

Credit card repayment datasets suffer from severe class imbalance, with high-risk customers representing a small minority (~2.8% here). A lender's triage workflow needs a probability and a risk tier for every customer, not just a yes/no call, and it needs evaluation numbers that hold up on data the model has never seen.

## Dataset

- 24,028 customers, 26 features: demographics, credit limit, 6 months of repayment status (PAY_0 to PAY_6), bill amounts, and payment amounts
- Target: whether a customer fell behind on their payment the following month
- Class split: ~97.2% low-risk vs. ~2.8% high-risk
- Scale mismatch: bill amount and credit limit run into the tens of thousands while repayment status columns are small integers, which matters for any distance-based method
- Split: 70:30 stratified outer train/test (test set sealed until final blind evaluation), with an identical 80:20 inner train/validation split used across every experiment notebook

## Approach

- **Leakage-free preprocessing:** train/test split performed first, before any imputation, encoding, or resampling; all statistics learned from training data only
- **Resampling strategies compared:** random undersampling, NearMiss-2 undersampling, and SMOTE oversampling
- **Models compared:** Random Forest, XGBoost, and LightGBM, giving 9 model/resampling combinations, each tuned with RandomizedSearchCV (5-fold CV)
- **Overfitting loop:** every candidate is checked for a train/validation gap; if it overfits, hyperparameters are retuned before moving to model comparison
- **Feature selection:** built-in feature importance plus SHAP analysis narrowed 49 encoded features down to a final 14, dominated by the PAY columns, credit limit, and one bill amount feature
- **Leakage inside cross-validation:** applying SMOTE once before RandomizedSearchCV let synthetic samples leak information across folds; fixed by wrapping SMOTE and the model in an ImbPipeline so SMOTE refits separately inside each fold
- **Risk tiering:** the fine-tuned model's predicted probabilities on the sealed blind-test set are split into five tiers using auto-detected density troughs, turning a raw score into categories a credit team can act on

## Final Model

**XGBoost on SMOTE**, wrapped in an ImbPipeline and fine-tuned on the merged inner-train and inner-validation data using the 14 selected features. SMOTE keeps every real customer record intact rather than discarding data, and XGBoost is a strong, well-established choice for structured tabular data like this.

Final blind-test risk tiers (customer counts):

| Risk Tier | Customers |
|---|---|
| Very Low Risk | 1,848 |
| Low Risk | 1,449 |
| Moderate Risk | 156 |
| High Risk | 81 |
| Very High Risk | 57 |

Most customers cluster in the lower-risk tiers, consistent with the ~97% base rate, giving a credit team a workable segmentation: tighter scrutiny on a small high-risk group, minimal friction for everyone else.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, LightGBM, imbalanced-learn, SHAP, Jupyter

## Repository Structure

```
01_data_loading_exploration.ipynb   # Data audit, missing values, class balance
02_eda.ipynb                        # Feature correlations, risk patterns
03_preprocessing.ipynb              # Leakage-free split, imputation, encoding
04_baseline_models.ipynb            # RF / XGBoost / LightGBM, random undersampling
05_near_miss_2_models.ipynb         # RF / XGBoost / LightGBM, NearMiss-2
06_SMOTE_models.ipynb               # RF / XGBoost / LightGBM, SMOTE
07_Run_best_model.ipynb             # Fine-tuning, ImbPipeline fix, final blind-test evaluation
```

