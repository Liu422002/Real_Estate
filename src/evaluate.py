from sklearn.metrics import mean_absolute_error
import logging

def evaluate_model(model, X_train, y_train, X_test, y_test):
    try:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        logging.info(f"Model evaluation completed. Train MAE: {train_mae}, Test MAE: {test_mae}")
        return train_mae, test_mae
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise