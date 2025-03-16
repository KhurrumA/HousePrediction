import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# Load the Boston housing dataset
boston_data = load_boston()
housing_df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
housing_df['Price'] = boston_data.target

print(housing_df.describe())
sns.heatmap(housing_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Log-transform the 'CRIM' column to create a new feature
housing_df['Log_Crime'] = np.log1p(housing_df['CRIM'])


sns.scatterplot(x=housing_df['Log_Crime'], y=housing_df['Price'])
plt.title('Log-transformed Crime Rate vs House Prices')
plt.show()

# Defining the feature set (X) and the target variable (y)
features = housing_df.drop(['Price'], axis=1)
target = housing_df['Price']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# List of models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regressor': SVR()
}

# Evaluate models using cross-validation and check the average RMSE (Root Mean Squared Error)
model_performance = {}
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    model_performance[model_name] = np.mean(np.sqrt(-cv_scores))  # RMSE

# Print out the performance for each model
for model_name, score in model_performance.items():
    print(f'{model_name} - Cross-validation RMSE: {score:.2f}')

best_model = RandomForestRegressor()
best_model.fit(X_train_scaled, y_train)

# Make predictions using the trained model
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model's performance on the test set
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R² Score

# Print out the evaluation metrics for the Random Forest model
print(f'\nRandom Forest Model Performance:')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R² Score: {r2:.2f}')

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Run GridSearchCV to find the best hyperparameters for the Random Forest model
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters found during GridSearchCV
print(f'\nBest hyperparameters found: {grid_search.best_params_}')

# Evaluate the tuned model
best_rf_model = grid_search.best_estimator_
y_pred_tuned = best_rf_model.predict(X_test_scaled)

# Evaluate the tuned model’s performance
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f'\nTuned Random Forest Model Performance:')
print(f'Mean Absolute Error (MAE): {mae_tuned:.2f}')
print(f'Mean Squared Error (MSE): {mse_tuned:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_tuned:.2f}')
print(f'R² Score: {r2_tuned:.2f}')

# Visualize the importance of each feature in the Random Forest model
feature_importances = best_rf_model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(X_train.shape[1]), feature_importances[sorted_idx], align='center')
plt.yticks(range(X_train.shape[1]), [X.columns[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Model')
plt.show()

# Check the residuals (the difference between actual and predicted values)
residuals = y_test - y_pred_tuned
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.show()
