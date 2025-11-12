import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv("steel.csv")
X = df.drop("tensile_strength", axis=1)
y = df["tensile_strength"]

# Detect and remove outliers based on the target variable (tensile_strength)
Q1 = df['tensile_strength'].quantile(0.25)
Q3 = df['tensile_strength'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print how many outliers exist
outliers = df[(df['tensile_strength'] < lower_bound) | (df['tensile_strength'] > upper_bound)]
print(f"Removed {len(outliers)} outliers from dataset.")

# Remove them
df = df[(df['tensile_strength'] >= lower_bound) & (df['tensile_strength'] <= upper_bound)]

# Optional: reset index after filtering
df = df.reset_index(drop=True)


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate best models on test set
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

# Set up K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialise models
gbr = GradientBoostingRegressor(random_state=42)

# Evaluate models using cross-validation
gbr_scores = cross_val_score(gbr, X_train, y_train, cv=kf, scoring='r2')

print("Gradient Boosting mean R²:", np.mean(gbr_scores))


# Hyperparameter Tuning using Grid Search
gradient_boosting_params = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.05, 0.1, 0.3]
}
# Gradient Boosting Grid Search
grid_search_gradient_boosting = GridSearchCV(gbr, gradient_boosting_params, cv=kf, scoring="neg_mean_squared_error")
grid_search_gradient_boosting.fit(X_train, y_train)

# Best models from Grid Search
best_gbr = grid_search_gradient_boosting.best_estimator_
print("Best Gradient Boosting parameters:", grid_search_gradient_boosting.best_params_)

# Evaluate tuned models on the test set
gbr_mae, gbr_mse, gbr_r2 = evaluate_model(best_gbr, X_test, y_test)


print("\nGradient Boosting (Tuned) Results:")
print(f"MAE: {gbr_mae:.2f}, MSE: {gbr_mse:.2f}, R²: {gbr_r2:.2f}")

results = pd.DataFrame({
    "Model": ["Gradient Boosting"],
    "MAE": [gbr_mae],
    "MSE": [gbr_mse],
    "R²": [gbr_r2]
})
print("\nFinal Test Results:\n", results)


# Display Results in a table
import matplotlib.pyplot as plt

# Plot predicted vs actual for both models
y_pred_gbr = best_gbr.predict(X_test)

plt.figure(figsize=(10,5))

# Gradient Boosting
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_gbr, color='steelblue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Tensile Strength")
plt.ylabel("Predicted Tensile Strength")
plt.title("Gradient Boosting: Predicted vs Actual")