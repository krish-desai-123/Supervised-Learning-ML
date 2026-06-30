<div align="center">

# 📉 Telco Customer Churn Prediction 🚀

**An end-to-end supervised machine learning project to predict at-risk subscribers and drive targeted retention strategies.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-success?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

</div>

<br>

## 🌟 Project Overview

A telecom company (modeled on Jio/Airtel) loses paying subscribers every month and needs to predict who is likely to leave. This practical exam project for **Red & White Skill Education** trains and compares four classifiers—**KNN, Naive Bayes, SVM, and Decision Tree**—on the IBM Telco Customer Churn dataset (**7,043 customers, 21 features**).

The goal is to evaluate the models on business-relevant metrics and package the best-performing algorithm as a deployable pipeline to help the retention team prioritize at-risk subscribers.

**Topics:** `python` `machine-learning` `supervised-learning` `customer-churn` `scikit-learn` `smote` `data-science`

---

# 📂 Repository Structure

| File | Description |
|------|-------------|
| 📜 `CustomerChurn_SupervisedLearning.ipynb` | Fully executed notebook containing EDA, preprocessing, model training, tuning, evaluation, and error analysis. |
| 📜 `churn_model.pkl` | Final saved Scikit-Learn Pipeline (preprocessing + best model) using Joblib. |
| 📜 `summary_report.md` | Business problem, preprocessing strategy, model recommendation, churn insights, and future improvements. |
| 📜 `requirements.txt` | Python dependencies required to run the project. |
| 📜 `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original IBM Telco Customer Churn dataset. |

---

# 📊 Key Results & Model Selection

- **Primary Metric:** **Recall** for the Churn class.
  - Missing a churning customer (**False Negative**) is much more expensive than offering retention incentives to a loyal customer (**False Positive**).

- **Recommended Model:** **Naive Bayes**
  - ✅ Recall ≈ **0.78**
  - ✅ Best **AUC-ROC ≈ 0.82**
  - ✅ Extremely fast training and inference
  - ✅ Suitable for daily production predictions

- **Runner-Up:** **Tuned KNN**
  - Recall ≈ **0.80**
  - Considered as an ensemble or secondary model.

- **Decision Tree**
  - Retained because of its interpretability through feature importance and visualization.

- **Handling Class Imbalance**
  - The dataset contains approximately **26.5% churners**.
  - **SMOTE** was applied **only on the training set after train-test splitting**, preventing data leakage.

> Full evaluation metrics, confusion matrices, ROC curves, and comparison tables are available in the notebook and `summary_report.md`.

---

# 🚩 Top Churn Signals

1. 📅 **Month-to-Month Contracts**
   - Customers with month-to-month contracts are significantly more likely to churn.

2. ⏳ **Low Tenure (< 12 Months)**
   - Newly acquired customers leave at a much higher rate.

3. 💰 **High Monthly Charges**
   - Price-sensitive customers, particularly Fiber Internet users, exhibit higher churn.

> See `summary_report.md` for business recommendations and retention strategies.

---

# ⚙️ How to Run

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/krish-desai-123/telco-churn-supervised-learning.git
cd telco-churn-supervised-learning
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure the dataset file is located in the project root:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## 3️⃣ Launch the Notebook

Run the notebook from start to finish to reproduce the complete workflow.

```bash
jupyter notebook CustomerChurn_SupervisedLearning.ipynb
```

---

## 4️⃣ Use the Saved Model

```python
import joblib

# Load trained pipeline
with open('churn_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Predict churn probability
churn_probabilities = pipeline.predict_proba(new_customer_df)

# Predict churn labels
predictions = pipeline.predict(new_customer_df)
```

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- imbalanced-learn (SMOTE)
- Joblib
- Jupyter Notebook

---

# 📈 Machine Learning Workflow

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Encoding
- Feature Scaling
- Train-Test Split
- SMOTE Oversampling
- Model Training
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Decision Tree
- Hyperparameter Tuning
- Model Evaluation
- ROC-AUC Analysis
- Model Deployment using Joblib

---

# 📺 Video Walkthrough

🎥 **Project Explanation & Code Walkthrough**

> Coming Soon

---

# 👨‍💻 Author

**Krish Desai**

Practical Exam — **Supervised Learning (Set B)**  
**Red & White Skill Education**

### Connect with Me

- GitHub: https://github.com/krish-desai-123

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

</div>