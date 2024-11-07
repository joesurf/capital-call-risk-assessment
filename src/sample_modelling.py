from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from xgboost import XGBRegressor


class ModelManager:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def run_preprocessing(self):
        # Normalization and scaling        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(scaled_features, self.y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def run_regression_model(self, X_train, X_test, y_train, y_test):
        X_train_with_const = sm.add_constant(X_train, has_constant='add')
        model_sm = sm.OLS(y_train, X_train_with_const).fit()

        X_test_with_const = sm.add_constant(X_test, has_constant='add')
        predictions_test = model_sm.predict(X_test_with_const)
        predictions_train = model_sm.predict(X_train_with_const)

        print(model_sm.summary())

        print("\nCoefficients:")
        for name, coef in zip(['const'] + list(self.X.columns), model_sm.params):
            print(f"{name}: {coef}")

        return y_train, predictions_train, y_test, predictions_test

    def run_decisiontree_model(self, X_train, X_test, y_train, y_test):
        from sklearn.tree import DecisionTreeRegressor
        model_dt = DecisionTreeRegressor(random_state=42)
        model_dt.fit(X_train, y_train)

        predictions_test = model_dt.predict(X_test)
        predictions_train = model_dt.predict(X_train)

        # Print feature importance
        feature_importances = model_dt.feature_importances_
        print("Feature Importances:")
        # Sort feature importances
        sorted_importances = sorted(zip(self.X.columns, feature_importances), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importances:
            print(f"{name}: {importance}")

        return y_train, predictions_train, y_test, predictions_test

    def run_randomforest_model(self, X_train, X_test, y_train, y_test):
        model_rf = RandomForestRegressor(random_state=42)
        model_rf.fit(X_train, y_train)

        predictions_test = model_rf.predict(X_test)
        predictions_train = model_rf.predict(X_train)

        # Print feature importance
        feature_importances = model_rf.feature_importances_
        print("Feature Importances:")
        # Sort feature importances
        sorted_importances = sorted(zip(self.X.columns, feature_importances), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importances:
            print(f"{name}: {importance}")

        # Plot feature importance
        import matplotlib.pyplot as plt
        indices = np.argsort(feature_importances)[::-1]
        plt.figure()
        plt.title("Feature importances - Random Forest")
        plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
        plt.xticks(range(X_train.shape[1]), self.X.columns[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

        return y_train, predictions_train, y_test, predictions_test

    def run_xgboost_model(self, X_train, X_test, y_train, y_test):
        model_xgb = XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')
        model_xgb.fit(X_train, y_train)

        predictions_test = model_xgb.predict(X_test)
        predictions_train = model_xgb.predict(X_train)

        # Print feature importance
        feature_importances = model_xgb.feature_importances_
        print("Feature Importances:")
        # Sort feature importances
        sorted_importances = sorted(zip(self.X.columns, feature_importances), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importances:
            print(f"{name}: {importance}")

        # Plot feature importance
        import matplotlib.pyplot as plt
        indices = np.argsort(feature_importances)[::-1]
        plt.figure()
        plt.title("Feature importances - XGBoost")
        plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
        plt.xticks(range(X_train.shape[1]), self.X.columns[indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

        return y_train, predictions_train, y_test, predictions_test

        return y_train, predictions_train, y_test, predictions_test

    def run_xgboost_model_with_tuning(self, X_train, X_test, y_train, y_test):
        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }

        # Initialize the XGBRegressor
        model_xgb = XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')

        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model_xgb = grid_search.best_estimator_

        predictions_test = best_model_xgb.predict(X_test)
        predictions_train = best_model_xgb.predict(X_train)

        # Print feature importance
        feature_importances = best_model_xgb.feature_importances_
        print("Feature Importances:")
        # Sort feature importances
        sorted_importances = sorted(zip(self.X.columns, feature_importances), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importances:
            print(f"{name}: {importance}")

        return y_train, predictions_train, y_test, predictions_test

    def evaluate_model(self, y_train, predictions_train, y_test, predictions_test):
        # Calculate metrics for the test set
        mse_test = mean_squared_error(y_test, predictions_test)
        r_squared_test = r2_score(y_test, predictions_test)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test, predictions_test)

        # Calculate metrics for the training set
        mse_train = mean_squared_error(y_train, predictions_train)
        r_squared_train = r2_score(y_train, predictions_train)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train, predictions_train)

        # Create a DataFrame to store the results
        metrics_df = pd.DataFrame({
            'Metric': ['R-squared (Train)', 'R-squared (Test)', 'MAE (Train)', 'MAE (Test)', 'RMSE (Train)', 'RMSE (Test)'],
            'Performance': [r_squared_train, r_squared_test, mae_train, mae_test, rmse_train, rmse_test],
        })

        print(metrics_df)

        # # Calculate residuals for the test set
        # residuals_test = y_test - predictions_test

        # # Plot residuals for the test set
        # plt.figure(figsize=(10, 6))
        # sns.histplot(residuals_test, kde=True, bins=30)
        # plt.title('Distribution of Residuals (Test Set)')
        # plt.xlabel('Residuals')
        # plt.ylabel('Frequency')
        # plt.show()

        # # Plot residuals vs. predicted values for the test set
        # plt.figure(figsize=(10, 6))
        # plt.scatter(predictions_test, residuals_test)
        # plt.axhline(0, color='red', linestyle='--')
        # plt.title('Residuals vs. Predicted Values (Test Set)')
        # plt.xlabel('Predicted Values')
        # plt.ylabel('Residuals')
        # plt.show()

        # # Check for homoscedasticity in the test set
        # plt.figure(figsize=(10, 6))
        # sns.scatterplot(x=predictions_test, y=residuals_test)
        # plt.axhline(0, color='red', linestyle='--')
        # plt.title('Homoscedasticity Check: Residuals vs. Predicted Values (Test Set)')
        # plt.xlabel('Predicted Values')
        # plt.ylabel('Residuals')
        # plt.show()

        # # Check for normality of residuals in the test set
        # plt.figure(figsize=(10, 6))
        # probplot(residuals_test, dist="norm", plot=plt)
        # plt.title('Normal Q-Q Plot (Test Set)')
        # plt.show()
