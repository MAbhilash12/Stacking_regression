import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Medical Insurance Cost Prediction")
st.markdown("### Stacking Regressor")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("data/insurance_cleaned.csv")

model = joblib.load("models/stacking_model.pkl")
columns = joblib.load("models/columns.pkl")

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

st.header("📊 Dataset Overview")

tab1, tab2, tab3 = st.tabs(
    ["Dataset Head", "Shape", "Statistical Summary"]
)

with tab1:
    st.dataframe(df.head())

with tab2:
    st.write("Rows :", df.shape[0])
    st.write("Columns :", df.shape[1])

with tab3:
    st.dataframe(df.describe())

# ---------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------

st.header("📈 Data Visualizations")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(6,4))

    sns.histplot(
        df["charges"],
        kde=True,
        ax=ax
    )

    ax.set_title("Charges Distribution")

    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots(figsize=(6,4))

    sns.histplot(
        df["bmi"],
        kde=True,
        ax=ax
    )

    ax.set_title("BMI Distribution")

    st.pyplot(fig)

# ---------------------------------------------------

col3, col4 = st.columns(2)

with col3:

    fig, ax = plt.subplots(figsize=(6,4))

    sns.boxplot(
        x="smoker",
        y="charges",
        data=df,
        ax=ax
    )

    ax.set_title("Smoker vs Charges")

    st.pyplot(fig)

with col4:

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        x="region",
        data=df,
        ax=ax
    )

    ax.set_title("Region Distribution")

    st.pyplot(fig)

# ---------------------------------------------------
# HEATMAP
# ---------------------------------------------------

st.header("🔥 Correlation Heatmap")

df_encoded = pd.get_dummies(
    df,
    drop_first=True
)

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(
    df_encoded.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------

st.header("📉 Model Evaluation")

data_model = pd.get_dummies(
    df,
    drop_first=True
)

X = data_model.drop("charges", axis=1)
y = data_model["charges"]

pred = model.predict(X)

r2 = r2_score(y, pred)
mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))

c1, c2, c3 = st.columns(3)

c1.metric("R² Score", f"{r2:.3f}")
c2.metric("MAE", f"{mae:.2f}")
c3.metric("RMSE", f"{rmse:.2f}")

# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------

st.header("🎯 Actual vs Predicted")

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(
    y,
    pred
)

ax.set_xlabel("Actual Charges")
ax.set_ylabel("Predicted Charges")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------

st.header("🧮 Predict Insurance Charges")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

with col3:
    children = st.number_input(
        "Children",
        min_value=0,
        max_value=10,
        value=0
    )

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)

# ---------------------------------------------------
# CREATE INPUT DATAFRAME
# ---------------------------------------------------

input_dict = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0
}

input_df = pd.DataFrame([input_dict])

# Ensure column order matches training

for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------

if st.button("Predict Insurance Cost"):

    prediction = model.predict(input_df)[0]

    st.success(
        f"Predicted Insurance Charges: ₹ {prediction:,.2f}"
    )

    st.balloons()
