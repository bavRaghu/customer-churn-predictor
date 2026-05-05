# Customer Churn Prediction System

Customer Churn Prediction System is a machine learning project that predicts whether a customer is likely to leave a service based on usage patterns, billing details, and subscription characteristics. The system uses structured telecom data, applies automated preprocessing, and provides real-time predictions through an interactive web interface.

---

## Getting Started

These instructions will help you run the churn prediction system locally for development and testing. Refer to the Deployment section for hosting the application.

---

### Prerequisites

Requirements for running the project:

- Python 3.8 or higher  
- pip (Python package manager)  
- Git 

---

### Installation

Clone the repository:

```bash
git clone https://github.com/bavRaghu/customer-churn-predictor
cd customer-churn-predictor
```

Create a virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run main.py
```

Open in browser:

```
http://localhost:8501
```

---

## Project Overview

The system performs the following steps:

- Accepts customer details as input  
- Applies preprocessing using a pipeline (encoding, scaling, imputation)  
- Uses trained machine learning models to predict churn probability  
- Displays churn risk levels (Low, Medium, High)  

---

## Models Used

### Logistic Regression
- Linear baseline model  
- Interpretable and efficient  
- Limited in capturing complex feature interactions  

### Random Forest
- Ensemble of decision trees  
- Captures non-linear relationships  
- Provides more intuitive predictions for complex patterns  

---

## Performance

| Model                | Accuracy | ROC-AUC |
|---------------------|----------|--------|
| Logistic Regression | 73.08% | 0.8316  |
| Random Forest       | 78.19% | 0.8309  |

The ROC-AUC score of ~0.83 indicates strong class separability, even with moderate accuracy due to class imbalance.

---

## Features

### Automated Preprocessing Pipeline
- Uses ColumnTransformer and Pipeline from scikit-learn  
- Handles encoding, scaling, and missing values consistently  
- Ensures identical transformations during training and inference  

### Real-Time Prediction Interface
- Built using Streamlit  
- Allows users to input customer data interactively  
- Displays churn probability and risk classification  

### Dual Model Support
- Supports both Logistic Regression and Random Forest  
- Allows users to select model type  
- Demonstrates differences between linear and non-linear models  

### Risk Interpretation
- Outputs probability-based churn risk  
- Categorizes predictions into Low, Medium, and High risk  
- Provides contextual explanation for predictions  

---


## Deployment

The application is deployed using Streamlit Cloud.

Deployment process:

- Code is pushed to GitHub  
- Streamlit Cloud is connected to the repository  
- On deployment:
  - Dependencies are installed  
  - The application is built  
  - The app is hosted with a public URL  
### Live Application
The application is deployed and accessible at:

https://customers-churn-predictor.streamlit.app/
