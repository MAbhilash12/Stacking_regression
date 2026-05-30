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

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Stacking Regressor",
    layout="wide"
)

st.title("🏠 House Price Prediction Using Stacking Regressor")

# ---------------------------
# LOAD DATA
# ---------------------------

df = pd.read_csv("data/housing_cleaned.csv")

model = joblib.load("models/stacking_model.pkl")
columns = joblib.load("models/columns.pkl")

# ---------------------------
# DATA OVERVIEW
# ---------------------------

st.header("Dataset Head")

st.dataframe(df.head())

st.header("Statistical Summary")

st.dataframe(df.describe())

# ---------------------------
# VISUALIZATIONS
# ---------------------------

st.header("Data Visualization")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots()

    sns.histplot(
        df["median_house_value"],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

with col2:

    fig, ax = plt.subplots()

    sns.scatterplot(
        x=df["median_income"],
        y=df["median_house_value"],
        ax=ax
    )

    st.pyplot(fig)

# ---------------------------
# MODEL EVALUATION
# ---------------------------

st.header("Model Evaluation")

df2 = df.dropna()

df2 = pd.get_dummies(
    df2,
    columns=["ocean_proximity"],
    drop_first=True
)

X = df2.drop("median_house_value", axis=1)
y = df2["median_house_value"]

pred = model.predict(X)

r2 = r2_score(y, pred)
mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mean_squared_error(y, pred))

c1, c2, c3 = st.columns(3)

c1.metric("R² Score", f"{r2:.3f}")
c2.metric("MAE", f"{mae:.2f}")
c3.metric("RMSE", f"{rmse:.2f}")

# ---------------------------
# PREDICTION SECTION
# ---------------------------

st.header("Predict House Price")

input_data = {}

for col in columns:

    if "ocean_proximity" not in col:

        value = st.number_input(
            col,
            value=float(X[col].mean())
        )

        input_data[col] = value

for col in columns:

    if "ocean_proximity" in col:

        input_data[col] = 0

input_df = pd.DataFrame([input_data])

if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    st.success(
        f"Predicted House Price: ${prediction:,.2f}"
    )

# ---------------------------
# ACTUAL VS PREDICTED
# ---------------------------

st.header("Actual vs Predicted")

sample_pred = model.predict(X.head(100))

fig, ax = plt.subplots()

ax.scatter(
    y.head(100),
    sample_pred
)

ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")

st.pyplot(fig)