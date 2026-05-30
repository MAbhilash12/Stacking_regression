import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

df = pd.read_csv(
    "data/insurance_cleaned.csv"
)

df = pd.get_dummies(
    df,
    drop_first=True
)

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

base_models = [
    (
        "rf",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    ),
    (
        "gb",
        GradientBoostingRegressor(
            random_state=42
        )
    )
]

meta_model = LinearRegression()

model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("R2 :", r2_score(y_test,pred))
print("MAE :", mean_absolute_error(y_test,pred))
print(
    "RMSE :",
    mean_squared_error(
        y_test,
        pred
    ) ** 0.5
)

joblib.dump(
    model,
    "models/stacking_model.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/columns.pkl"
)

print("Model Saved")