# 📩 Message Intelligence System

A Machine Learning project for automatically classifying digital messages as **Spam** or **Legitimate** using multiple classification algorithms and probability concepts.

---

## 🚀 Live Demo

https://message-intelligence-system-pr-4.streamlit.app/

---

## 📌 Project Objective

The objective of this project is to design a classification system that identifies whether incoming digital messages are:

* **0 → Legitimate Message**
* **1 → Spam Message**

The project combines probability theory with distance-based, margin-based, and probabilistic classifiers to analyze and compare model performance.

---

## 🛠️ Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

---

## 📂 Dataset Features

The dataset contains message-related features extracted from text and user behavior signals.

### Input Features

* Message Length
* Word Count
* Number of URLs
* Number of Digits
* Number of Special Characters
* Spam Keyword Score
* Legitimate Keyword Score
* Sender Activity Score
* Sender Account Age (Days)
* Messages Sent in Last 24 Hours
* Hour of Day
* Day of Week

### Target Variable

**spam_label**

* **0 = Legitimate Message**
* **1 = Spam Message**

---

## 📊 Machine Learning Models Used

### 1. K-Nearest Neighbors (KNN)

* Distance-based classifier
* Uses nearest neighbors for prediction
* Simple baseline model

### 2. Support Vector Machine (SVM)

* Margin-based classifier
* Finds optimal decision boundary
* Provides high classification accuracy

### 3. Naive Bayes

* Probabilistic classifier
* Uses Bayes' Theorem
* Fast and efficient for message classification

---

## ⚙️ Project Workflow

1. Data Collection
2. Data Preprocessing
3. Missing Value Handling
4. Feature Scaling
5. Train-Test Split
6. Model Building
7. Model Evaluation
8. Model Comparison
9. Final Analysis and Reporting

---

## 📈 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 📊 Visualizations

The project includes:

* Spam vs Legitimate Distribution Plot
* Correlation Heatmap
* KNN Accuracy vs K Value Plot
* Confusion Matrix
* Model Accuracy Comparison Plot

---

## 📁 Project Structure

```text
Message_Intelligence_System/
│
├── Message_Intelligence_Dataset.csv
├── Message_Intelligence_System_Complete_Project.ipynb
├── app.py
├── requirements.txt
├── README.md
└── images/
```

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Message_Intelligence_System.git
cd Message_Intelligence_System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter notebook
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## 📊 Expected Results

* SVM generally provides the highest classification accuracy.
* Naive Bayes is the fastest model for prediction.
* KNN works as a good baseline classifier and helps understand distance-based learning.

---

## 🎯 Business Recommendation

For real-world message filtering systems:

* Use **SVM** when maximum accuracy is required.
* Use **Naive Bayes** when speed and scalability are important.
* Use **KNN** for benchmarking and educational purposes.

---

## ⭐ Conclusion

The Message Intelligence System successfully demonstrates how probability theory and machine learning algorithms can be used to classify messages as spam or legitimate. The project provides practical experience in data preprocessing, classification, evaluation metrics, and model comparison while highlighting the strengths and limitations of different machine learning approaches.
