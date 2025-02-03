# Oasys 

## 1. Introduction

Mental health is a crucial aspect of overall well-being, yet diagnosing and managing mental health conditions remain challenging. This project explores the use of machine learning models for classifying mental health severity using socio-demographic and behavioral data.

**Objective**: The aim is to evaluate and compare the performance of different machine learning models in predicting the severity of mental health conditions.


## 2. Dataset

**Source**: The dataset used in this study consists of 1,000 data points covering socio-demographic and behavioral attributes.

**Target**: The target variable represents the severity of mental health conditions categorized into four levels: None, Mild, Moderate, and Severe.


### 2.1 Dataset Features

- **Age**: Integer value representing the respondent’s age.
- **Gender**: Categorical variable indicating gender identity.
- **Occupation**: Job role of the respondent.
- **Stress Levels**: Self-reported stress levels.
- **Sleep Patterns**: Average daily sleep duration.
- **Physical Activity**: Frequency of physical exercise.
- **Work Hours**: Number of work hours per week.


## 3. Data Preprocessing

### 3.1 Data Cleaning

- **Handling Missing Values**: Missing numerical values were filled with the mean, while categorical features were imputed with the mode.
- **Encoding Categorical Variables**: Gender and occupation were converted to numerical data using one-hot encoding.
- **Normalization**: Min-max scaling was applied to continuous variables to ensure uniform distribution.

### 3.2 Dimensionality Reduction

- **Principal Component Analysis (PCA)**: Applied to reduce feature dimensionality while preserving key variance in the data.

### 3.3 Addressing Class Imbalance

- **Synthetic Minority Over-sampling Technique (SMOTE)**: Used to generate synthetic samples for underrepresented classes to improve model balance and fairness.


## 4. Model Development

### 4.1 Algorithms Used

- **Logistic Regression**: A simple binary classification model used as a baseline.
- **LightGBM**: A gradient boosting model designed for efficient tree-based learning.
- **Support Vector Machine (SVM)**: A model effective in high-dimensional data spaces.

### 4.2 Model Training

1. **Data Split**: The dataset was split into 80% training and 20% testing.
2. **Hyperparameter Tuning**: Grid search was applied to optimize hyperparameters for SVM and LightGBM.
3. **Cross-Validation**: A 5-fold cross-validation approach was implemented to enhance model generalization.

### 4.3 Evaluation Metrics

- **Accuracy**: Percentage of correctly classified instances.
- **Precision**: Measures the correctness of positive predictions.
- **Recall**: Measures the ability to identify actual positives.
- **F1-Score**: Harmonic mean of precision and recall.


## 5. Results and Comparison

|Model|Accuracy|Precision|Recall|F1-Score|
|---|---|---|---|---|
|Logistic Regression|50.3%|0.49|0.50|0.46|
|LightGBM|51.0%|0.52|0.52|0.48|
|SVM with PCA+SMOTE|55.5%|0.56|0.56|0.55|

**Key Observations**:

- **SVM with PCA and SMOTE performed the best**, handling high-dimensional and imbalanced data more effectively.
- **Logistic Regression and LightGBM showed lower predictive power**, suggesting that more advanced preprocessing and feature engineering are necessary.


## 6. Conclusion

This study highlights the effectiveness of machine learning in classifying mental health conditions. The **SVM model with PCA and SMOTE outperformed other models**, demonstrating the value of dimensionality reduction and class balancing techniques.

### Future Work

- **Feature Engineering**: Exploring additional socio-demographic and behavioral variables.
- **Ensemble Models**: Combining multiple models to enhance predictive accuracy.
- **Larger Dataset**: Using a more extensive dataset to improve generalizability.


## 7. How to Run the Project

### Prerequisites

- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `lightgbm`

### Steps

1. Clone the repository:
    
    ```
    git clone https://github.com/username/mental-health-classification.git
    ```
    
2. Run the Jupyter notebook:
    
    ```
    jupyter notebook mental_health_classification.ipynb
    ```
