
# 🏨 Hotel Booking Demand Cancellation Prediction

This project aims to predict whether a hotel booking will be canceled using a real-world dataset from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). It applies both **Machine Learning (ML)** and **Deep Learning (DL)** techniques to assist hotel managers in minimizing revenue loss and improving resource planning.

---

## 📂 Table of Contents

- [About the Dataset](#-About-the-Dataset)
- [Project Overview](#-Project-Overview)
- [Technologies Used](#-Technologies-Used)
- [Model Performance](#-Model-Performance-Summary)
- [How to Run](#▶-How-to-Run)
- [Results](#-Results)
- [Real-time Prediction](#Real-time-Prediction)
- [References](#-References)

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
- Lightgbm
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
                             
|-----------------|----------|----Precision-----|------Recall-------|-----F1-Score-----| ROC AUC|
|-----------------|----------|------------------|-------------------|------------------|--------|

|Model        | Accuracy | Class 0 | Class 1 | Class 0 | Class 1 | Class 0 | Class 1 | ROC AUC |
|-------------|----------|---------|---------|---------|---------|---------|---------|---------|
|Decision Tree| 75.59%   |  77.7%  |  72.89% |  78.66% |  71.76% |  78.17% |  72.32% |   83%   |
|SVM          | 70.44%   |  73.76% |  66.43% |  72.67% |  67.66% |  73.21% |  67.04% |    -    |
|Random Forest| 81.35%   |  82.08% |  80.35% |  84.99% |  76.79% |  83.51% |  78.53% |   90%   |
|XGBoost      | 80.79%   |  82.1%  |  79.08% |  83.68% |  77.18% |  82.88% |  78.12% |   89%   |
|KNN          | 73.72%   |  75.56% |  71.24% |  77.89% |  68.5%  |  76.71% |  69.84% |   83%   |
|Lightgbm     | 79.93%   |  81.15% |  77.84% |  82.56% |  76.64% |  82.05% |  77.24% |   88%   |
|ANN          | 74.32%   |  74.78% |  73.63% |  81.17% |  65.76% |  77.84% |  69.47% |   89%   |


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
- Overall, tree-based ensemble models (Random Forest, XGBoost, LightGBM) consistently
outperformed other algorithms, making them preferable for hotel booking cancellation
prediction.
---

## Real-time Prediction

- [Streamlit App]{https://github.com/waleed-o99/Hotel-Booking-Demand-Cancelation-Streamlit-App/tree/main)

---

## 📚 References

- [Hotel Booking Demand Dataset on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Scikit-learn documentation: https://scikit-learn.org/
- TensorFlow & Keras: https://www.tensorflow.org/
- XGBoost documentation: https://xgboost.readthedocs.io/
