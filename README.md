# 🛡️ SentinelCard: Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning project that detects potentially fraudulent credit card transactions using **Python, Scikit-Learn, Random Forest, and SMOTE**.

The project focuses on handling highly imbalanced transaction data, evaluating fraud detection performance, and understanding the trade-off between false alarms and missed fraud cases.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Technologies Used](#-technologies-used)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Dataset Information](#-dataset-information)
- [Handling Class Imbalance](#-handling-class-imbalance)
- [Model Evaluation](#-model-evaluation)
- [Results](#-results)
- [Installation and Setup](#-installation-and-setup)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)

---

## 🚀 Project Overview

Credit card fraud detection is an important application of Machine Learning in the financial industry. Fraudulent transactions are rare compared with genuine transactions, making them difficult for traditional classification models to identify.

**SentinelCard** demonstrates how a Machine Learning pipeline can be built to identify suspicious transactions while addressing the challenges of an imbalanced dataset.

The project uses a Random Forest Classifier and SMOTE to improve the model's ability to learn patterns associated with fraudulent transactions. Precision-Recall analysis is used to understand how different decision thresholds affect fraud detection.

## ❗ Problem Statement

In a highly imbalanced dataset, approximately 99% of transactions may be genuine and only 1% may be fraudulent.

A model that predicts every transaction as genuine could achieve approximately 99% accuracy while failing to detect any fraud.

Therefore, accuracy alone is not enough to measure the performance of a fraud detection model.

This project focuses on evaluating the model using suitable metrics and understanding the balance between:

- **False Positives:** Genuine transactions incorrectly flagged as fraudulent.
- **False Negatives:** Fraudulent transactions incorrectly classified as genuine.
- **Precision:** How many transactions flagged as fraud are actually fraudulent.
- **Recall:** How many actual fraudulent transactions the model successfully detects.

## 🎯 Project Objectives

- Build an end-to-end credit card fraud detection pipeline.
- Generate a synthetic dataset with a 99:1 genuine-to-fraud ratio.
- Explore transaction patterns using Exploratory Data Analysis (EDA).
- Split data into training and testing sets using stratified sampling.
- Handle class imbalance using SMOTE.
- Train a Random Forest classification model.
- Evaluate model performance using ROC-AUC and Precision-Recall analysis.
- Understand how decision thresholds affect fraud detection results.

## ✨ Key Features

- **Imbalanced Dataset:** Simulates a real-world challenge where fraudulent transactions are rare.
- **Exploratory Data Analysis:** Examines class distribution and transaction features.
- **Stratified Train-Test Split:** Maintains the approximate class distribution in both datasets.
- **SMOTE Sampling:** Creates synthetic minority-class examples for model training.
- **Random Forest Classifier:** Learns patterns that help distinguish genuine and fraudulent transactions.
- **ROC-AUC Evaluation:** Measures how well the model separates the two classes across different thresholds.
- **Precision-Recall Analysis:** Helps understand the balance between detecting fraud and raising unnecessary alerts.
- **Threshold Optimization:** Supports choosing a decision threshold based on business requirements.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations |
| Pandas | Data handling and analysis |
| Scikit-Learn | Model training and evaluation |
| Imbalanced-Learn | SMOTE implementation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Interactive development and experimentation |

## 🔄 Machine Learning Workflow

The project follows these steps:

1. **Data Generation:** Create 10,000 synthetic transactions with a highly imbalanced class distribution.
2. **Exploratory Data Analysis:** Visualize the class distribution and explore the available features.
3. **Data Splitting:** Divide the dataset into training and testing sets using stratified sampling.
4. **Class Balancing:** Apply SMOTE to the training data to generate synthetic fraud examples.
5. **Model Training:** Train a Random Forest Classifier using the balanced training dataset.
6. **Prediction:** Generate fraud predictions and probability scores for the test dataset.
7. **Model Evaluation:** Evaluate the model using ROC-AUC and Precision-Recall analysis.
8. **Threshold Analysis:** Study how changing the decision threshold affects false positives and false negatives.

## 📊 Dataset Information

This project uses a **synthetically generated dataset** rather than a real credit card transaction dataset.

| Property | Details |
|---|---|
| Total transactions | 10,000 |
| Genuine transactions | Approximately 99% |
| Fraudulent transactions | Approximately 1% |
| Dataset type | Synthetic |
| Learning task | Binary classification |
| Target classes | Genuine (0), Fraudulent (1) |

**Note:** The exact number of transactions in each class may depend on how the dataset is generated.

## ⚖️ Handling Class Imbalance

### What is class imbalance?

Class imbalance occurs when one category contains many more examples than another category.

In this project, genuine transactions form the majority class, while fraudulent transactions form the minority class.

### What is SMOTE?

SMOTE stands for **Synthetic Minority Over-sampling Technique**.

It creates new synthetic examples of the minority class to help the model learn patterns from a class that has fewer examples.

In this project, SMOTE is applied only to the training data.

The test data remains unchanged so that model performance can be evaluated on data that preserves the original class distribution.

### Before and after SMOTE

| Dataset | Genuine Transactions | Fraudulent Transactions |
|---|---:|---:|
| Training data before SMOTE | 7,878 | 122 |
| Training data after SMOTE | 7,878 | 7,878 |

The values above represent the reported training split for this project.

## 📈 Model Evaluation

### Random Forest Classifier

Random Forest is a Machine Learning algorithm that combines multiple decision trees to make predictions.

For this project, the model predicts whether a transaction belongs to the genuine or fraudulent class.

### Evaluation metrics

- **ROC-AUC:** Measures how well the model distinguishes between genuine and fraudulent transactions across different decision thresholds.
- **Precision:** Measures the proportion of predicted fraud cases that are actually fraudulent.
- **Recall:** Measures the proportion of actual fraud cases that the model detects.
- **F1-Score:** Combines precision and recall into a single score.
- **Confusion Matrix:** Shows correct and incorrect predictions for both classes.

### Why is the decision threshold important?

A classification model can produce a probability score for each transaction. A decision threshold determines when that score is treated as fraud.

For example, lowering the threshold may help detect more fraudulent transactions, but it may also increase false alarms.

Raising the threshold may reduce false alarms, but it can cause more fraudulent transactions to go undetected.

Precision-Recall curves help compare these trade-offs and support threshold selection based on business needs.

## 🏆 Results

| Metric | Result |
|---|---|
| Dataset size | 10,000 synthetic transactions |
| Original class distribution | Approximately 99:1 |
| Balancing technique | SMOTE |
| Classification model | Random Forest Classifier |
| Reported ROC-AUC | Approximately 0.765 |

### Result Interpretation

The reported ROC-AUC score of approximately **0.765** indicates that the model has some ability to distinguish between genuine and fraudulent transactions.

However, ROC-AUC alone does not show how many fraudulent transactions are detected or how many genuine transactions are incorrectly flagged.

Precision, recall, F1-score, the confusion matrix, and the Precision-Recall curve should also be examined when evaluating fraud detection performance.

**Important:** These results come from a synthetic dataset. They do not establish how the model would perform on real financial transactions.

## 💻 Installation and Setup

### Prerequisites

Before starting, make sure you have:

- Python installed on your computer.
- Jupyter Notebook or JupyterLab installed.
- Git installed if you want to clone the repository.

### Step 1: Clone the repository

Replace the placeholder below with your actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Open the project folder

```bash
cd SentinelCard
```

Replace `SentinelCard` with the actual folder name if it is different.

### Step 3: Install the required libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn jupyter
```

## ▶️ How to Run

1. Open a terminal in the project folder.
2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `notebook.ipynb`.
4. Run the notebook cells from top to bottom.
5. Generate the synthetic dataset.
6. Explore the visualizations and class distribution.
7. Apply SMOTE to the training data.
8. Train the Random Forest model.
9. Review the evaluation results and Precision-Recall curves.

**Important:** Make sure the notebook uses the correct file names and follows the workflow described above.

## 📁 Project Structure

The suggested project structure is:

```text
SentinelCard/
│
├── notebook.ipynb       # Main ML pipeline
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies (optional)
```

This structure is an example. Adjust it to match the actual files in your repository.

To create a `requirements.txt` file, use:

```bash
pip freeze > requirements.txt
```

## 🔮 Future Improvements

Some possible improvements for this project are:

- Test the model on a publicly available credit card fraud dataset.
- Compare Random Forest with Logistic Regression and other classifiers.
- Evaluate precision, recall, F1-score, and PR-AUC.
- Compare different methods for handling class imbalance.
- Select the decision threshold using business costs and validation data.
- Add cross-validation and parameter tuning.
- Build a Streamlit dashboard for interactive fraud prediction.
- Add model monitoring to detect changes in transaction patterns over time.

## ⚠️ Disclaimer

SentinelCard is an educational Machine Learning project built with synthetic data.

It is intended to demonstrate fraud detection concepts, class imbalance handling, model evaluation, and decision threshold analysis.

It is **not a production-ready fraud detection system** and should not be used to make real financial decisions without further testing, validation, security checks, and monitoring.

---

## 👩‍💻 Author

**Sakshi Vavale**

B.Tech Student | Artificial Intelligence & Data Science

GitHub: [sakshivavale](https://github.com/sakshivavale)

---

⭐ If you find this project useful, consider giving the repository a star!
