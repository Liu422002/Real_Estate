from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging

def split_data(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error during train-test split: {e}")
        raise

def train_linear_regression(X_train, y_train):
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("Linear Regression model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training Linear Regression model: {e}")
        raise

def train_random_forest(X_train, y_train):
    try:
        model = RandomForestRegressor(
            n_estimators=200,
            criterion="absolute_error",
            random_state=42
        )
        model.fit(X_train, y_train)
        logging.info("Random Forest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training Random Forest model: {e}")
        raise