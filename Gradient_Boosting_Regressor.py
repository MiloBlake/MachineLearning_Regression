import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
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
gbr_default = GradientBoostingRegressor(random_state=42)

# Cross-validation with default params
gbr_default_scores = cross_val_score(gbr_default, X_train, y_train, cv=kf, scoring='r2')
print(f"Cross-validation mean R²: {np.mean(gbr_default_scores):.4f}")

# Train on full training set
gbr_default.fit(X_train, y_train)

# Evaluate on test set
gbr_default_mae, gbr_default_mse, gbr_default_r2 = evaluate_model(gbr_default, X_test, y_test)
print(f"\nTest Set Results (Default Model):")
print(f"MAE: {gbr_default_mae:.2f}, MSE: {gbr_default_mse:.2f}, R²: {gbr_default_r2:.2f}")


# Initialise tuned model
gbr_tuned = GradientBoostingRegressor(random_state=42)

# Evaluate model using cross-validation
gbr_tuned_scores = cross_val_score(gbr_tuned, X_train, y_train, cv=kf, scoring='r2')

print("Gradient Boosting mean R²:", np.mean(gbr_tuned_scores))

# Hyperparameter Tuning using Grid Search
gradient_boosting_params = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.05, 0.1, 0.3]
}
# Gradient Boosting Grid Search
grid_search_gradient_boosting = GridSearchCV(gbr_tuned, gradient_boosting_params, cv=kf, scoring="neg_mean_squared_error")
grid_search_gradient_boosting.fit(X_train, y_train)

# Best model from Grid Search
best_gbr = grid_search_gradient_boosting.best_estimator_
print("Best Gradient Boosting parameters:", grid_search_gradient_boosting.best_params_)

# Evaluate tuned model on the test set
gbr_tuned_mae, gbr_tuned_mse, gbr_tuned_r2 = evaluate_model(best_gbr, X_test, y_test)

print(f"\nTest Set Results (Tuned Model):")
print(f"MAE: {gbr_tuned_mae:.2f}, MSE: {gbr_tuned_mse:.2f}, R²: {gbr_tuned_r2:.2f}")

results = pd.DataFrame({
    "Model": ["Default", "Tuned"],
    "MAE": [gbr_default_mae, gbr_tuned_mae],
    "MSE": [gbr_default_mse, gbr_tuned_mse],
    "R²": [gbr_default_r2, gbr_tuned_r2]
})
print("\nFinal Test Results:\n", results)

# Display Results in a table
import matplotlib.pyplot as plt

# Predictions for plotting
y_pred_default = gbr_default.predict(X_test)
y_pred_tuned = best_gbr.predict(X_test)

plt.figure(figsize=(14, 5))

# Default model
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_default, color='steelblue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Tensile Strength")
plt.ylabel("Predicted Tensile Strength")
plt.title(f"Default Model (R²={gbr_default_r2:.3f})")

# Tuned model
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_tuned, color='green', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Tensile Strength")
plt.ylabel("Predicted Tensile Strength")
plt.title(f"Tuned Model (R²={gbr_tuned_r2:.3f})")

plt.tight_layout()
plt.show()