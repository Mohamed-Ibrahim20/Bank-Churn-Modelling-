
# Bank Churn Prediction

This project aims to predict customer churn for a bank using various machine learning models, including a Neural Network, to optimize performance and improve customer retention strategies.

## Dataset

- **Source:** [Kaggle - Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling/data)
- **Description:** The dataset consists of 10,000 customer records, with features such as credit score, geography, gender, age, tenure, balance, number of products, and more.

| Column | Description |
|:-------|:------------|
| RowNumber | Index |
| CustomerId | Customer ID |
| Surname | Customer's Surname |
| CreditScore | Customer’s credit score |
| Geography | Country of residence |
| Gender | Gender |
| Age | Age |
| Tenure | Number of years as a customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Has credit card (1 = yes, 0 = no) |
| IsActiveMember | Active member status |
| EstimatedSalary | Estimated salary |
| Exited | Churn (1 = left, 0 = stayed) |

---

## Project Workflow

1. **Data Loading and Exploration**
   - Loaded the dataset and explored the data structure.
   - Checked for missing values and basic statistics.

2. **Data Preprocessing**
   - Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`.
   - Encoded categorical variables:
     - `Gender` with Label Encoding.
     - `Geography` with One-Hot Encoding.
   - Standardized numerical features using `StandardScaler`.

3. **Data Splitting**
   - Split the dataset into features (`X`) and labels (`y`).
   - Further split into **training** and **test** datasets.

4. **Baseline Machine Learning Models**
   - Trained and evaluated classic models:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Naive Bayes
     - Decision Tree Classifier
     - Random Forest Classifier
   - Evaluated using:
     - Accuracy
     - Confusion Matrix
     - Classification Report

5. **Neural Network Model**
   - Built a **basic feed-forward neural network** using **Keras**.
   - Model architecture:
     - Input layer matching the number of features.
     - Two hidden layers with ReLU activation.
     - Output layer with a sigmoid activation for binary classification.
   - Compiled the model using:
     - Optimizer: Adam
     - Loss function: Binary Crossentropy
     - Metrics: Accuracy
   - Trained the model with a validation split to monitor performance.

---

## Results

| Model | Test Accuracy |
|:------|:--------------|
| Logistic Regression | ~80% |
| K-Nearest Neighbors | ~78% |
| Support Vector Machine | ~79% |
| Naive Bayes | ~76% |
| Decision Tree | ~79% |
| Random Forest | ~86% |
| **Neural Network** | **~85%** |

- The **Random Forest** achieved the highest accuracy among classical models.
- The **Neural Network** achieved approximately **85%** test accuracy without hyperparameter tuning.

---

## Requirements

- Python 3.x
- Jupyter Notebook
- Required libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - keras

Install the required libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

---

## How to Run

1. Clone this repository or download the notebook file.
2. Install all required libraries.
3. Open `Bank Churn Modelling.ipynb` in Jupyter Notebook.
4. Run all cells sequentially to:
   - Preprocess the data
   - Train baseline models
   - Build and train the Neural Network
   - Evaluate final model performance

---

## Future Work

- Perform hyperparameter tuning to further improve the neural network’s performance.
- Try more complex network architectures (e.g., deeper networks, dropout layers).
- Explore ensemble techniques beyond Random Forest (e.g., XGBoost, LightGBM).
- Use advanced evaluation metrics like ROC AUC Score and Precision-Recall Curve.
- Deploy the model as a web application for real-time predictions.

---

# 🚀 Summary

By applying a mix of classical machine learning and a simple feed-forward neural network, this project successfully builds a predictive model for customer churn, achieving solid accuracy and laying the groundwork for further improvements.
