import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from preprocessing import preproccess_data

# Load dataset
df = pd.read_csv("steel.csv")
X = df.drop("tensile_strength", axis=1)
y = df["tensile_strength"]

X, y = preproccess_data(df)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate best model on test set
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

# Set up K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Default Gradient Boosting Regressor
rf_default = RandomForestRegressor(random_state=42)

# Cross-validation with default params
rf_default_scores = cross_val_score(rf_default, X_train, y_train, cv=kf, scoring='r2')
print(f"Cross-validation mean R²: {np.mean(rf_default_scores):.4f}")

# Train on full training set
rf_default.fit(X_train, y_train)

print("Default Random Forest hyperparameters:")
print(rf_default.get_params())

# Evaluate on test set
rf_default_mae, rf_default_mse, rf_default_r2 = evaluate_model(rf_default, X_test, y_test)
print(f"\nTest Set Results (Default Model):")
print(f"MAE: {rf_default_mae:.2f}, MSE: {rf_default_mse:.2f}, R²: {rf_default_r2:.2f}")

# Initialise model
rf  = RandomForestRegressor(random_state=42)

# Evaluate model using cross-validation
rf_scores = cross_val_score(rf, X_train, y_train, cv=kf, scoring='r2')

print("Random Forest mean R²:", np.mean(rf_scores))

# Random Forest Grid Search
random_forest_params = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 5, 10, 15]
}
grid_search_random_forest = GridSearchCV(rf, random_forest_params, cv=kf, scoring="neg_mean_squared_error")
grid_search_random_forest.fit(X_train, y_train)

# Best model from Grid Search
best_rf = grid_search_random_forest.best_estimator_
print("Best Random Forest parameters:", grid_search_random_forest.best_params_)

# Evaluate tuned model on the test set
rf_mae, rf_mse, rf_r2 = evaluate_model(best_rf, X_test, y_test)

print("\nRandom Forest (Tuned) Results:")
print(f"MAE: {rf_mae:.2f}, MSE: {rf_mse:.2f}, R²: {rf_r2:.2f}")

results = pd.DataFrame({
    "Model": ["Default", "Tuned"],
    "MAE": [rf_default_mae, rf_mae],
    "MSE": [rf_default_mse, rf_mse],
    "R²": [rf_default_r2, rf_r2]
})
print("\nFinal Test Results:\n", results)


# Display Results
import matplotlib.pyplot as plt

# Predictions for plotting
y_pred_default = rf_default.predict(X_test)
y_pred_tuned = best_rf.predict(X_test)

plt.figure(figsize=(14, 5))

# Default model
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_default, color='steelblue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Tensile Strength")
plt.ylabel("Predicted Tensile Strength")
plt.title(f"Default Model (R²={rf_default_r2:.3f})")

# Tuned model
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_tuned, color='forestgreen', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Tensile Strength")
plt.ylabel("Predicted Tensile Strength")
plt.title(f"Tuned Model (R²={rf_r2:.3f})")

plt.tight_layout()
plt.show()