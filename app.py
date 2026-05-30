import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="Stacking Classification",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction Using Stacking Classifier")

# Load Dataset
df = pd.read_csv("data/heart_cleaned.csv")

# Load Model
model = joblib.load(
    "models/stacking_classifier.pkl"
)

columns = joblib.load(
    "models/columns.pkl"
)

# ------------------
# DATASET HEAD
# ------------------

st.header("Dataset Head")

st.dataframe(df.head())

# ------------------
# SUMMARY
# ------------------

st.header("Statistical Summary")

st.dataframe(df.describe())

# ------------------
# CLASS DISTRIBUTION
# ------------------

st.header("Target Distribution")

fig, ax = plt.subplots()

sns.countplot(
    x="target",
    data=df,
    ax=ax
)

st.pyplot(fig)

# ------------------
# HEATMAP
# ------------------

st.header("Correlation Heatmap")

fig, ax = plt.subplots(
    figsize=(10,8)
)

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# ------------------
# EVALUATION
# ------------------

X = df.drop("target", axis=1)
y = df["target"]

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
pre = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{acc:.3f}")
c2.metric("Precision", f"{pre:.3f}")
c3.metric("Recall", f"{rec:.3f}")
c4.metric("F1 Score", f"{f1:.3f}")

# ------------------
# CONFUSION MATRIX
# ------------------

st.header("Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

st.pyplot(fig)

# ------------------
# USER INPUT
# ------------------

st.header("Predict Heart Disease")

user_input = {}

for col in columns:

    value = st.number_input(
        col,
        value=float(X[col].mean())
    )

    user_input[col] = value

input_df = pd.DataFrame([user_input])

if st.button("Predict"):

    pred = model.predict(input_df)[0]

    prob = model.predict_proba(input_df)[0]

    if pred == 1:
        st.error(
            f"Heart Disease Detected ({prob[1]*100:.2f}%)"
        )
    else:
        st.success(
            f"No Heart Disease ({prob[0]*100:.2f}%)"
        )
