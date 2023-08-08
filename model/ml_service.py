import json
import os
import time
from uuid import uuid4

import numpy as np
import redis
import settings
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('files/model.h5')

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

    # Adding an extra dimension for batch size

    input_data = np.expand_dims(input_data, axis=1)  

    # # Reshape the original array to have shape (1, 48) filled with zeros
    # desired_shape = (1, 43)

    # # Assuming you have a numpy array of shape (1, 8)
    # original_array = input_data

    # # Create the sequence (3, 3, 3, 3) to fill the remaining elements
    # sequence = np.array([[3, 3, 3, 3]])

    # # Calculate the number of times the sequence needs to be repeated
    # num_repeats = (desired_shape[1] - original_array.shape[1]) // sequence.shape[1]

    # # Concatenate the original array and the repeated sequence to get the desired shape
    # completed_array = np.concatenate([original_array, np.tile(sequence, (1, num_repeats))], axis=1)

    # input_data = completed_array

    # Get the prediction from the model

    # Send the input data through the trained model to get the predictions
    
    # Choose a threshold to classify the person has to hospitalize or not
    
    try:

        pred_probability = model.predict(input_data)

        threshold = 0.9
        score = 0

        if pred_probability >= threshold:
            prediction = 'The person has to hospitalize next year.'
        else:
            prediction = 'The person does not has to hospitalize.'
        
        score = (pred_probability*100)

    except Exception as e:
        prediction = f'Prediction Error: {e}'
        score = 0

    # Return Prediction And Probability

    return prediction, score

def predict_temp(input_data):

    """
    Temporal function to test connection to Redis and ML model.     
    """

    try:
        total = 0
        score = 0
        for value in input_data:
            total += float(value)
            if float(value) > score:
                score = float(value)

        score = (score * 100) / total

        if score >= 80:
            prediction = 'Very healthy'
        elif score < 80 and score >= 70:
            prediction = 'Healthy'
        elif score < 70 and score >= 50:
            prediction = 'Stable, but it is recommended that you see a doctor.'
        else:
            prediction = 'Please see your doctor!'       

    except Exception as e:
        prediction = f'Prediction Error: {e}'
        score = 0

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

        msg_content = {
            "prediction": prediction,
            "score": str(score),
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
