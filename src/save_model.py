import pickle
import logging

def save_model(model, file_path):
    try:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def load_model(file_path):
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise