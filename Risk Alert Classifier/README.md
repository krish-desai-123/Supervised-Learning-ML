# 🏦 Risk Alert Classifier
### Supervised Learning 
> **"Shaping Skills for Scaling Higher"**

---

## 📌 Project Overview

This project builds an **Early Warning System** for a digital banking platform to identify **high-risk customers** who are likely to default on payments or engage in fraudulent behavior.

The system implements a complete **classification pipeline** — from data preparation and handling class imbalance, to model building, hyperparameter tuning, and final evaluation using industry-relevant metrics.

---

## 🎯 Objective

- Accurately identify high-risk customers using machine learning
- Handle highly imbalanced data using sampling techniques
- Evaluate models using appropriate classification metrics (not just accuracy)
- Improve performance using hyperparameter tuning

---

## 📂 Project Structure

```
Risk-Alert-Classifier/
│
├── Risk_Alert_Classifier.ipynb        # Main Jupyter Notebook (all tasks)
├── Risk_Alert_Classifier_Dataset.csv  # Dataset
├── README.md                          # Project documentation
│
└── outputs/                           # Generated plots & charts
    ├── class_distribution.png
    ├── missing_values.png
    ├── sampling_comparison.png
    ├── dt_vs_rf.png
    ├── roc_comparison.png
    └── feature_importance.png
```

---

## 📊 Dataset Description

| Feature | Description |
|--------|-------------|
| `customer_id` | Unique customer identifier |
| `age` | Customer age |
| `gender` | Male / Female |
| `region` | Geographic region |
| `employment_type` | Salaried / Self-Employed / Unemployed |
| `annual_income_inr` | Annual income in INR |
| `credit_score` | Credit score (300–900) |
| `credit_utilization_ratio` | Ratio of credit used vs available |
| `missed_payments_12m` | Missed payments in last 12 months |
| `avg_late_payment_days` | Average days late on payments |
| `monthly_transaction_count` | Monthly number of transactions |
| `monthly_spend_inr` | Monthly spending in INR |
| `cash_advance_count_6m` | Cash advances taken in last 6 months |
| `complaints_last_6m` | Complaints filed in last 6 months |
| `failed_login_attempts_3m` | Failed login attempts in last 3 months |
| `account_tenure_months` | How long the customer has had the account |
| `debt_balance_inr` | Current outstanding debt |
| `risk_status` | **Target** → 0 = Low Risk, 1 = High Risk |

**Dataset Size:** 4,600 records | **Class Imbalance:** ~88% Low Risk vs ~12% High Risk

---

## 🧠 Theory Concepts Covered

1. **Logistic Regression** — sigmoid function, probability-based classification
2. **Classification Metrics** — why accuracy alone is insufficient for imbalanced data
3. **Type-I & Type-II Errors** — False Positives vs False Negatives in banking context
4. **Precision, Recall, F1-Score, TPR, FPR** — formulas and interpretations
5. **AUC-ROC** — threshold-independent model evaluation
6. **Class Imbalance** — causes, effects, and solutions

---

## ⚙️ Implementation Steps

### Part B — Data Preparation
- Identified input features and target variable
- Stratified Train-Test Split (80/20) to preserve class distribution
- Missing value analysis and **KNN Imputation** (k=5 neighbors)
- Feature scaling using `StandardScaler`

### Part C — Baseline Model
- **Logistic Regression** as baseline
- Generated Confusion Matrix, Accuracy, Precision, Recall, F1
- Identified Type-I (FP) and Type-II (FN) errors

### Part D — Handling Imbalanced Data
Applied 4 sampling techniques and compared performance:

| Technique | Description |
|-----------|-------------|
| **Under-Sampling** | Randomly reduce majority class |
| **Over-Sampling** | Randomly duplicate minority class |
| **SMOTE** | Synthetic Minority Oversampling Technique |
| **ADASYN** | Adaptive Synthetic Sampling |

### Part E — Tree-Based Models
- **Decision Tree** — trained and analyzed for overfitting
- **Random Forest** — ensemble of 100 trees
- Compared both models on accuracy and generalization

### Part F — Hyperparameter Tuning
- **RandomizedSearchCV** → explored broad parameter space for DT and RF
- **GridSearchCV** → fine-tuned best model with narrow parameter grid
- Compared tuned vs untuned performance

### Part G — ROC Analysis
- Plotted ROC curves for all models on the same graph
- Computed AUC-ROC scores for comparison
- Selected best model based on business requirement: **minimize False Negatives**

---

## 📈 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| LR (No Balancing) | ~0.88 | High | Very Low | Low | ~0.80 |
| LR + SMOTE | Moderate | Moderate | High | Moderate | ~0.85 |
| LR + ADASYN | Moderate | Moderate | High | Moderate | ~0.85 |
| Decision Tree | Moderate | Moderate | Moderate | Moderate | ~0.80 |
| Decision Tree (Tuned) | Good | Good | Good | Good | ~0.83 |
| Random Forest | Good | High | Good | Good | ~0.90 |
| **Random Forest (Tuned)** | **Best** | **High** | **High** | **Best** | **~0.92** |

> ✅ **Best Model: Tuned Random Forest with SMOTE balancing**

---

## 🔍 Key Findings

1. **Class imbalance is a major challenge** — baseline model achieves ~88% accuracy but misses most high-risk customers
2. **SMOTE and ADASYN** significantly improve Recall on the minority class from ~10% to ~70%+
3. **Random Forest outperforms Decision Tree** due to ensemble averaging reducing variance
4. **Hyperparameter tuning** further improves AUC-ROC and F1-Score
5. **Feature importance** shows `missed_payments_12m`, `credit_score`, `debt_balance_inr`, and `credit_utilization_ratio` are the strongest predictors

---

## 💼 Business Interpretation

| Error Type | Meaning | Business Impact |
|------------|---------|----------------|
| **False Positive (Type-I)** | Low-risk customer flagged as high-risk | Unnecessary account restriction — customer dissatisfaction |
| **False Negative (Type-II)** | High-risk customer missed | Fraud goes undetected — **direct financial loss** |

**Recommendation:** Deploy the Tuned Random Forest model with a **lowered classification threshold (0.3 instead of 0.5)** to maximize Recall — catching more actual fraudsters at the cost of slightly more false alarms.

---

## 🛠️ Technologies Used

- **Python 3.11**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — ML models, metrics, tuning
- **imbalanced-learn** — SMOTE, ADASYN, sampling
- **Jupyter Notebook** — development environment

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/risk-alert-classifier.git
cd risk-alert-classifier

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter

# 3. Launch Jupyter Notebook
jupyter notebook Risk_Alert_Classifier.ipynb
```

---

## 👨‍💻 Author

**Krish Desai**

* GitHub: [@krish-desai-123](https://github.com/krish-desai-123)

---

<div align="center">
<i>If this project helped you understand regression and gradient descent better, consider dropping a ⭐!</i>
</div>
