import streamlit as st
import pickle
import pandas as pd

pipeline_lr = pickle.load(open("pipeline_lr.pkl", "rb"))
pipeline_rf = pickle.load(open("pipeline_rf.pkl", "rb"))

st.markdown("### 🤖 Select Model")

model_choice = st.radio(
    "Choose prediction model:",
    ["Random Forest", "Logistic Regression"],
    help="LR is simpler and interpretable, RF captures complex patterns"
)
if model_choice == "Logistic Regression":
    st.info("Linear model — easier to interpret, faster, but simpler.")
else:
    st.info("Ensemble model — captures complex patterns, usually more accurate.")

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the service.")

# TENURE 
tenure = st.slider(
    "Tenure (months)",
    0, 72, 12,
    help="Number of months the customer has stayed. Lower tenure often means higher churn risk."
)

# BASIC INFO
st.markdown("### 🧾 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])

    senior = st.selectbox(
        "Senior Citizen",
        ["Yes", "No"],
        help="Senior customers sometimes show different usage patterns"
    )

    partner = st.selectbox(
        "Has Partner?",
        ["Yes", "No"],
        help="Customers with partners tend to churn less"
    )

    dependents = st.selectbox(
        "Has Dependents?",
        ["Yes", "No"],
        help="Customers with dependents are usually more stable"
    )

with col2:
    monthly = st.number_input(
        "Monthly Charges (₹)",
        0.0, 10000.0, 500.0,
        help="Higher charges may increase churn probability"
    )

    total = st.number_input(
        "Total Charges (₹)",
        0.0, 200000.0, 5000.0,
        help="Total billing amount over time"
    )

# SERVICES BLOCK
st.markdown("### 📡 Services")

col1, col2 = st.columns(2)

with col1:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
        help="Fiber optic users often churn more due to higher costs"
    )

    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# BILLING
st.markdown("### 💳 Billing & Contract")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        help="Month-to-month users churn more frequently"
    )

    billing = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

predict = st.button("🔍 Predict Churn")

# PREDICTION
if predict:

    senior_val = 1 if senior == "Yes" else 0

    input_df = pd.DataFrame([{
        "Sex": gender,
        "SeniorCitizen": senior_val,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": billing,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    if model_choice == "Logistic Regression":
        model = pipeline_lr
    else:
        model = pipeline_rf

    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📈 Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2f}")
    st.progress(int(prob * 100))

    if prob > 0.7:
        st.error("⚠️ HIGH RISK of churn")
    elif prob > 0.4:
        st.warning("⚠️ MEDIUM RISK of churn")
    else:
        st.success("✅ LOW RISK of churn")

    st.markdown("### 🧠 Why this prediction?")

    if tenure < 6 and monthly > 800 and contract == "Month-to-month":
        st.write(
            "This profile typically indicates HIGH churn risk:\n"
            "- Low tenure (new customer)\n"
            "- High monthly charges\n"
            "- Flexible contract\n\n"
            "If prediction is still low, it means other features (like services, support, or payment method) are offsetting the risk."
        )
    else:
        st.write(
            "Churn is influenced by multiple factors together. "
            "The model balances tenure, pricing, services, and contract type."
        )