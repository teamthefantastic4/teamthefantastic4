import json
import os
import time
from uuid import uuid4

import numpy as np
import redis
import settings
#from tensorflow.keras.applications import BestModel?
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID, charset="utf-8",
    decode_responses=True
)

model = ResNet50(include_top=True, weights="imagenet")

def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """

    class_name = None
    pred_probability = None

    #Loading and preprocessing
    # TO DO

    x_batch = ...

    x_batch = preprocess_input(x_batch)

    try:
        preds = model.predict(x_batch)

        predictionLabel = decode_predictions(preds, top = 5)
        class_name = predictionLabel[0][0][1]
        pred_probability = round(preds[0,preds.argmax()],4)

    except:
        raise SystemExit("ERROR: Failed to load the Model!")

    return class_name, pred_probability

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent

        queue_name, msg = db.brpop(settings.REDIS_QUEUE)

        msg = json.loads(msg)

        #TO DO

        msg_name = msg['msg_name']
        msg_id = msg['id']

        class_name, pred_probability = predict(msg_name)

        msg_content = {
            "prediction": class_name,
            "score": str(pred_probability),
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
