import json
import os
import time
from uuid import uuid4

import numpy as np
import redis
import settings
import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.preprocessing import StandardScaler
import pickle

model = keras.models.load_model('files/model.h5')

# Load the scaler from the file
loaded_scaler = joblib.load('files/scaler.pkl')

db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID, charset="utf-8",
    decode_responses=True
)

def predict(input_data):

    """
    Load data trained and 
    received, then, run our ML model to get predictions.

    Parameters
    ----------

    # Receive input data from the API request
    
    input_data List or Tuple

    Returns
     -------
    prediction, score : tuple(str, float)
    
        Model predicted class as a string and the corresponding confidence
        score as a number.

     """     
     
    pred_probability = None
    score = 0

    # Adding an extra dimension for batch size

    input_data = np.expand_dims(input_data, axis=0)  

    # Transform new data using the loaded scaler
    scaled_new_data = loaded_scaler.transform(input_data)

    # Get the prediction from the model

    # Send the input data through the trained model to get the predictions
    
    # Choose a threshold to classify the person needs to be hospitalized or not
    
    try:

        pred_probability = model.predict(scaled_new_data)

        threshold = 0.9

        if pred_probability >= threshold:
            prediction = 'This person needs to be hospitalized next year.'
        else:
            prediction = 'This person does not need to be hospitalized.'
        
        score = (pred_probability*100)

    except Exception as e:
        prediction = 'Prediction Error: Try Again'
        score = 0

    # Return Prediction And Probability

    return prediction, score

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load data from the corresponding folder run our ML model to get predictions.
    """
    while True:

        queue_name, msg = db.brpop(settings.REDIS_QUEUE)

        msg = json.loads(msg)

        msg_data = msg['input_data']
        msg_id = msg['id']

        prediction, score = predict(msg_data)

        score = np.around(score, 2)

        s = str(score)
        chars = [']', '['] 
        result = s.translate(str.maketrans('', '', ''.join(chars)))

        msg_content = {
            "prediction": prediction,
            "score": result,
        }

        try:
            db.set(msg_id, json.dumps(msg_content))

        except:
            raise SystemExit("ERROR: Results Not Stored!")

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
