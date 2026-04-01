import logging
from src.data_loader import load_data
from src.preprocessing import split_features_target
from src.train import split_data, train_linear_regression, train_random_forest
from src.evaluate import evaluate_model
from src.save_model import save_model

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    df = load_data("data/final.csv")
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    lr_model = train_linear_regression(X_train, y_train)
    lr_train_mae, lr_test_mae = evaluate_model(lr_model, X_train, y_train, X_test, y_test)

    print("Linear Regression Train MAE:", lr_train_mae)
    print("Linear Regression Test MAE:", lr_test_mae)

    rf_model = train_random_forest(X_train, y_train)
    rf_train_mae, rf_test_mae = evaluate_model(rf_model, X_train, y_train, X_test, y_test)

    print("Random Forest Train MAE:", rf_train_mae)
    print("Random Forest Test MAE:", rf_test_mae)

    save_model(lr_model, "models/linear_model.pkl")
    save_model(rf_model, "models/random_forest_model.pkl")

if __name__ == "__main__":
    main()