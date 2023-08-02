import os
import json
import logging
from flask import (
    Blueprint,
    jsonify,
    render_template,
    request,
)
from middleware import model_predict

# Logging for debugging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

router = Blueprint("app_router", __name__, template_folder="templates")

@router.route("/", methods=["GET"])
def index():
    """
    GET: Index endpoint, renders our HTML code.
    """
    return render_template("index.html")

@router.route("/", methods=["POST"])
def result():
    """
    POST: Used in our frontend so somebody can upload their personal info.
    When it receives correct data from the UI, it also calls our ML model
    to get and display the predictions.
    """
    try:
        null_data = [0, '0', None, '']
        rpse = {"success": False, "prediction": None, "score": None,
                "prediction_label": None, "score_label": None
                }
        
        # Check if the form has 0 or null values
        if any(value in null_data for key, value in request.form.items()):
            rpse = {"success": False, "prediction": "Please complete the form, thank you!!!", "score": 0}
            return jsonify(rpse), 400
        
        # Get the input data from the form as a list
        input_data = [float(value) for key, value in request.form.items()]

        # Get the prediction and score from the model
        prediction, score = model_predict(input_data)

        # Log for debugging
        logging.debug("Form data: %s", request.form)
        logging.debug("Input data: %s", input_data)

        rpse = {
        "success": True,
        "prediction": prediction,
        "score": score,
        "prediction_label": "Health Status: ",
        "score_label": "Score: "
        }

    except Exception as e:
        print(f'Error: {e}')
        return jsonify(rpse), 400

    return jsonify(rpse)


# @router.route("/", methods=['POST'])
# def result():
#     try:
#         null_data = [0, '0', None, '']
#         prediction = 'Please complete the form, thank you.'
#         score = 0
#         rpse = {"success": False, "prediction": None, "score": None}

#         # check if the form has 0 or null values
#         for x, i in request.form.items():
#             if i in null_data:
#                 rpse["success"] = False
#                 rpse["prediction"] = prediction
#                 rpse["score"] = score
#                 break
        
#         ###########################################################################
#         # REPLACE THIS PART WITH THE MODEL replace this part with the model
#         if rpse['prediction'] == None:
#             total = 0
#             for x, i in request.form.items():
#                 total += float(i)
#                 if float(i) > score:
#                     score = float(i)
        
#             score = (score * 100) / total

#             if score >= 80:
#                 prediction = 'Very healthy'
#             elif score < 80 and score >= 70:
#                 prediction = 'Healthy'
#             elif score < 70 and score >= 50:
#                 prediction = 'Stable, but it is recommended that you see a doctor.'
#             else:
#                 prediction = 'Please see your doctor!'
#         ###########################################################################
#             rpse["success"] = True
#             rpse["prediction"] = prediction
#             rpse["score"] = score

#         return jsonify(rpse)
        
    # except Exception as e:
    #     print(f'Error: {e}')
    #     return jsonify(rpse), 400
