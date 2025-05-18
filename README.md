
# 🏨 Hotel Booking Demand Cancellation Prediction

This project aims to predict whether a hotel booking will be canceled using a real-world dataset from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). It applies both **Machine Learning (ML)** and **Deep Learning (DL)** techniques to assist hotel managers in minimizing revenue loss and improving resource planning.

---

## 📂 Table of Contents

- [About the Dataset](#about-the-dataset)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

---

## 📊 About the Dataset

The dataset contains over 119,000 hotel bookings, with information such as:

- Hotel type (Resort or City)
- Booking channel and customer type
- Special requests
- Room preferences
- Lead time, stay duration, etc.
- **Target Variable:** `is_canceled` (1 = Canceled, 0 = Not canceled)

---

## 🚀 Project Overview

### Objectives

- Predict booking cancellations before they happen
- Compare traditional ML models with a Deep Learning ANN
- Perform hyperparameter tuning using GridSearchCV
- Evaluate models using classification metrics and ROC curves

### Models Used

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost
- Artificial Neural Network (ANN)

---

## 🛠 Technologies Used

- Python 3.10
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- TensorFlow / Keras

---

## 📈 Model Performance Summary

| Model         | Accuracy | F1 Score (Class 1) | ROC AUC |
|---------------|----------|--------------------|---------|
| Decision Tree | 75.59%   | 0.52               | ~0.79   |
| Random Forest | 78.9%    | 0.53               | 0.80    |
| XGBoost       | 78.9%    | 0.53               | 0.80    |
| KNN           | 78.0%    | 0.49               | 0.77    |
| KNN           | 78.0%    | 0.49               | 0.77    |
| ANN           | 78.9%    | 0.53               | 0.80    |

| Model         | Accuracy | F1 Score (Class 1) | ROC AUC |
|---------------|----------|--------------------|---------|

|------------|------|--Precision---|    Recall      |    F1-Score      | ROC AUC|
|-------|--------|----------|--------------------|---------|---|

|Model  |     Accuracy    |Class 0| Class 1 |Class 0 |Class1 | Class 0 |Class 1 |        |
|-------|---------|-------|---------|--------|-------|---------|--------|--------|
|Decision Tree| 75.59| 77.7| 72.89| 78.66| 71.76| 78.17| 72.32| 83|
|SVM| 70.44| 73.76| 66.43| 72.67| 67.66| 73.21| 67.04| -|
|Random Forest| 81.35| 82.08| 80.35| 84.99| 76.79| 83.51| 78.53| 90|
|XGBoost| 80.79| 82.1| 79.08| 83.68| 77.18| 82.88| 78.12| 89|
|KNN| 73.72| 75.56| 71.24| 77.89| 68.5| 76.71| 69.84| 83|
|Lightgbm| 79.93| 81.15| 77.84| 82.56| 76.64| 82.05| 77.24| 88|
|ANN| 74.32| 74.78| 73.63| 81.17| 65.76| 77.84| 69.47| 89|


> Note: Class 1 (cancellations) is harder to predict due to class imbalance.

---

## ▶️ How to Run

1. **Clone the repo:**

```bash
git clone https://github.com/your-username/hotel-booking-cancellation-prediction.git
cd hotel-booking-cancellation-prediction
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Launch the notebook:**

```bash
jupyter notebook
```

---

## 📊 Results

- Ensemble models like **Random Forest** and **XGBoost** performed best.
- The **ANN model** achieved competitive results, showing promise for deep learning on structured data.
- All models struggled with recall for the positive class due to imbalance.

---

## 🔮 Future Work

- Address class imbalance with techniques like **SMOTE** or **cost-sensitive learning**
- Add real-time prediction dashboard
- Test with additional features (e.g., reviews, time-series data)
- Experiment with LSTM or Transformer for temporal features

---

## 📚 References

- [Hotel Booking Demand Dataset on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Scikit-learn documentation: https://scikit-learn.org/
- TensorFlow & Keras: https://www.tensorflow.org/
- XGBoost documentation: https://xgboost.readthedocs.io/
