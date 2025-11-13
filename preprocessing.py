import pandas as pd

def preproccess_data(data):
    # Remove outliers
    Q1 = data['tensile_strength'].quantile(0.25)
    Q3 = data['tensile_strength'].quantile(0.75)

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data['tensile_strength'] < lower_bound) | (data['tensile_strength'] > upper_bound)]
    print(f"Removed {len(outliers)} outliers from dataset.")

    # Remove outliers    
    data = data[(data['tensile_strength'] >= lower_bound) & (data['tensile_strength'] <= upper_bound)]
    data = data.reset_index(drop=True)
    
    # Separate features and target
    X = data.drop("tensile_strength", axis=1)
    y = data["tensile_strength"]
    
    return X, y