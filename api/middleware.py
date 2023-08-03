import json
import time
from uuid import uuid4

import redis
import settings

db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID, 
    charset="utf-8",
    decode_responses=True
)

def model_predict(input_data):
def model_predict(input_data):
    """
    Receives an image name and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    image_name : str
        Name for the image uploaded by the user.

    Returns
    -------
    prediction, score : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    prediction = None
    score = None

    # Assign an unique ID for this job and add it to the queue.
    # We need to assing this ID because we must be able to keep track
    # of this particular job across all the services
    
    job_id = str(uuid4())


    job_data = None

    msg = {
        "id": job_id,
        "image_name": image_name,
    }

    job_data = json.dumps(msg)

    # Send the job to the model service
    # Send the job to the model service

    db.lpush(
        settings.REDIS_QUEUE,
        job_data
        )
    
    # Loop until we received the response from our ML model
    while True:


        output = db.get(job_id)

        # Check if the text was correctly processed by our ML model
        # Don't modify the code below, it should work as expected
        if output is not None:
            output = json.loads(output)
            prediction = output["prediction"]
            score = output["score"]

            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return prediction, score