import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("SCRIPT STARTED")


# 1. LOAD DATA

file_path = "data/Water_Consumption_Dataset.csv"
df = pd.read_csv(file_path)

print("DATA LOADED SUCCESSFULLY")

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATA SHAPE:")
print(df.shape)

print("\nCOLUMN INFO:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum())


# 2. CLEANING

df = df.drop("Household_ID", axis=1)


# 3. ENCODING

X = df.drop("Monthly_Water_Consumption_Litres", axis=1)
y = df["Monthly_Water_Consumption_Litres"]

X = pd.get_dummies(X, drop_first=True)

print("\nENCODING CATEGORICAL VARIABLES...")


# 4. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nDATA SPLIT COMPLETE")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# 5. TRAIN MODELS

print("\nTRAINING MODELS...")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}
best_model = None
best_score = -1

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    results[name] = r2

    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    # track best model
    if r2 > best_score:
        best_score = r2
        best_model = model


# 6. SAVE BEST MODEL

joblib.dump(best_model, "outputs/best_model.pkl")
print("\nBEST MODEL SAVED")


# 7. FEATURE IMPORTANCE

if hasattr(best_model, "coef_"):
    importance = best_model.coef_
else:
    importance = best_model.feature_importances_

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nFEATURE IMPORTANCE:")
print(coefficients)


# 8. PREDICTIONS

y_pred = best_model.predict(X_test)

results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results_df.to_csv("outputs/predictions.csv", index=False)

print("\nPREDICTIONS SAVED")
print("\nSCRIPT FINISHED SUCCESSFULLY")