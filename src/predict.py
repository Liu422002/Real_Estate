import logging

def make_prediction(model, input_data):
    try:
        prediction = model.predict(input_data)
        logging.info("Prediction made successfully.")
        return prediction
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise