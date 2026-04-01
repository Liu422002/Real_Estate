import logging

def split_features_target(df):
    try:
        X = df.drop("price", axis=1)
        y = df["price"]
        logging.info("Features and target split successfully.")
        return X, y
    except Exception as e:
        logging.error(f"Error splitting features and target: {e}")
        raise