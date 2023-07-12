import os
import json

import settings
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from middleware import model_predict

router = Blueprint("app_router", __name__, template_folder="templates")

@router.route("/", methods=["GET", "POST"])
def index():
    """
    GET: Index endpoint, renders our HTML code.

    POST: Used in our frontend so we can upload and show an image.
    When it receives an image from the UI, it also calls our ML model to
    get and display the predictions.
    """
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":
        #TO DO


@router.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint used to get predictions without need to access the UI.

    Parameters
    ----------
    file : str
        Input image we want to get predictions from.

    Returns
    -------
    flask.Response
        JSON response from our API having the following format:
            {
                "success": bool,
                "prediction": str,
                "score": float,
            }

        - "success" will be True if the input file is valid and we get a
          prediction from our ML model.
        - "prediction" model predicted class as string.
        - "score" model confidence score for the predicted class as float.
    """

    # If user sends an invalid request (e.g. no file provided) this endpoint
    # should return `rpse` dict with default values HTTP 400 Bad Request code
    rpse = {"success": False, "prediction": None, "score": None}

    if request.method == "POST":

        #TO DO

        prediction, score = model_predict(...)
        rpse["success"] = True
        rpse["prediction"] = prediction
        rpse["score"] = score
        return jsonify(rpse)

    return jsonify(rpse), 400


@router.route("/feedback", methods=["GET", "POST"])

def feedback():
    """
    Store feedback from users about wrong predictions on a plain text file.

    Parameters
    ----------
    report : request.form
        Feedback given by the user with the following JSON format:
            {
                "filename": str,
                "prediction": str,
                "score": float
            }

        - "filename" corresponds to the image used stored in the uploads
          folder.
        - "prediction" is the model predicted class as string reported as
          incorrect.
        - "score" model confidence score for the predicted class as float.
    """
    # Store the reported data to a file on the corresponding path
    # already provided in settings.py module (settings.FEEDBACK_FILEPATH)
    # TODO

    report = request.form.get("report")

    if report:
        with open(settings.FEEDBACK_FILEPATH, "a") as outfile:
            outfile.write(report + "\n")

    # Don't change this line
    return render_template("index.html")
