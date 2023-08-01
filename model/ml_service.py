import json
import os
import time
from uuid import uuid4
#from src import data_utils
#from src import preprocessing

import numpy as np
import redis
import settings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler

# Normalize numerical features to a common scale (between 0 and 1)
#scaler = MinMaxScaler()

db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID, charset="utf-8",
    decode_responses=True
)

# def predict(input_data):

#     """
#     Load data trained and 
#     received, then, run our ML model to get predictions.

#     Parameters
#     ----------

#     # Receive input data from the API request
    
#     input_data List or Tuple

#     Returns
#     -------
#     pred_probability : tuple(str, float)
#         Model predicted class as a string and the corresponding confidence
#         score as a number.
#     """
    
#     # Preprocess the input data (assuming 'gender', 'weight', 'BMI', 'age', and 'blood_pressure' are provided)
#     new_person_data = np.array([
#         input_data['gender'],
#         input_data['weight'],
#         input_data['BMI'],
#         input_data['age'],
#         input_data['blood_pressure']
#     ]).reshape(1, -1)

#     # Normalize the input data using the same scaler used for training data
#     new_person_data_scaled = scaler.transform(new_person_data)

#     pred_probability = None

#     #Loading and preprocessing 

#     #new_person_data_scaled = scaler.transform(new_person_data)

#     # Get the prediction from the model

#     pred = model.predict(new_person_data_scaled)

#     # Choose a threshold to classify the person has to hospitalize or not

#     ## Agregué la variable 'prediction' para agregarla a los valores que retorna la función
#     prediction = None
#     threshold = 0.5
#     if pred[0][0] >= threshold:
#         prediction = "The person has to hospitalize next year."
#     else:
#         prediction = "The person does not has to hospitalize."

#     return prediction, pred_probability

def predict_temp(input_data):

    """
    Temporal function to test connection to Redis
    
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

        # 
        queue_name, msg = db.brpop(settings.REDIS_QUEUE)

        msg = json.loads(msg)

        msg_data = msg['input_data']
        msg_id = msg['id']

        pred_msg, pred_probability = predict_temp(msg_data)

        msg_content = {
            "prediction": pred_msg,
            "score": str(pred_probability),
        }

        try:
            db.set(msg_id, json.dumps(msg_content))
        except:
            raise SystemExit("ERROR: Results Not Stored!")

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)

# def preprocess_data():
    
#     # Preprocess Data to Train the Model

#     X_train, y_train, X_test, y_test = None

#     train_data, y_train = None

#     app_train, app_test, columns_description = data_utils.get_datasets()

#     X_train, y_train, X_test, y_test = data_utils.get_feature_target(app_train, app_test)

#     train_data, X_val, y_train, y_val = data_utils.get_train_val_sets(X_train, y_train)

#     return train_data, y_train

# def model(input_data):

#     # Create the MLP model, Input should be like this X_train.shape[1]
#     model = Sequential()
#     model.add(Dense(64, activation='relu', input_dim=input_data.shape[1]))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))

#     # Compile the model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#     epochs = 100
#     batch_size = 32

#     # From Preprocessing get train an val data

#     train_data, y_train = preprocess_data()

#     model.fit(train_data, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

#     input_data = np.expand_dims(input_data, axis=0)  # Adding an extra dimension for batch size

#     # Send the input data through the trained model to get the predictions
#     predictions = model.predict(input_data)

#     return predictions

if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
