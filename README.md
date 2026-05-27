# Customer Churn Prediction System

An end-to-end Data Science and Analytics project focused on predicting customer churn using Machine Learning and delivering business insights through an interactive dashboard.

---

#  Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses.
This project uses Machine Learning to predict whether a customer is likely to churn and provides actionable business insights through a modern analytics dashboard.

The solution includes:

* Data cleaning & preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning with XGBoost
* Model evaluation & threshold tuning
* Interactive Streamlit dashboard
* Real-time churn prediction
* Cloud deployment

---

# Live Demo

```text
https://customer-churn-prediction-soares.streamlit.app/
```

---

# Dashboard Preview

... 

---

# Machine Learning Model

### Model Used

* XGBoost Classifier

### Metrics Achieved

| Metric            | Score |
| ----------------- | ----- |
| ROC-AUC           | 0.83  |
| Accuracy          | 0.76  |
| Recall (Churn)    | 0.74  |
| Precision (Churn) | 0.51  |

---

# Project Structure

```bash
customer-churn-prediction/
│
├── data/
├── dashboard/
│   └── dashboard.py
│
├── models/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── model_columns.pkl
│
├── notebooks/
├── src/
│
├── requirements.txt
├── README.md
└── app.py
```

---

# Tech Stack

### Data Science

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Visualization

* Plotly
* Matplotlib
* Streamlit

### Deployment

* Streamlit Cloud
* GitHub

---

# Features

Customer churn prediction, Real-time AI inference, Interactive analytics dashboard, KPI monitoring, Business insights, Customer risk scoring, Retention recommendations, Modern SaaS-style UI

---

# Key Business Insights

* Customers with month-to-month contracts have significantly higher churn rates.
* Fiber optic customers show elevated churn behavior.
* Longer tenure strongly correlates with customer retention.
* High monthly charges may increase churn probability.

---

# Real-Time Prediction Engine

The dashboard allows users to simulate customer profiles and instantly predict churn probability using the trained XGBoost model.

Inputs include:

* Contract type
* Customer tenure
* Internet service
* Monthly charges

The system returns:

* Churn risk score
* Risk level
* Retention recommendations

---

# Installation

Clone the repository:

```bash
git clone https://github.com/gsoars/customer-churn-prediction.git
```

Navigate to the project:

```bash
cd customer-churn-prediction
```

Create virtual environment:

```bash
python -m venv venv
```

Activate venv (Windows):

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run dashboard/dashboard.py
```

---

# Future Improvements

* SHAP explainability integration
* FastAPI model serving
* Docker containerization
* PostgreSQL integration
* Automated ML pipeline
* Authentication system

---

# Author

### Gabriel Soares

Data Analytics & Data Science Portfolio Project

LinkedIn:

```text
www.linkedin.com/in/gabriel-soares-42b35a16b
```

GitHub:

```text
https://github.com/gsoars
```
