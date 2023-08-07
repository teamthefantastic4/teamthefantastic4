from flask import (
    Blueprint,
    jsonify,
    render_template,
    request,
)
from middleware import model_predict

router = Blueprint("app_router", __name__, template_folder="templates")

@router.route("/", methods=["GET"])
def index():
    """
    GET: Index endpoint, renders our HTML code.
    """
    return render_template("index.html")


@router.route("/", methods=['POST'])
def result():
    try:
        null_data = [0, None, '0', '']
        
        rpse = {"success": True,
                "prediction": None,
                "score": None,
                "prediction_label": "",
                "score_label": ""}

        # Check if the form has 0 or null values
        if any(value in null_data for key, value in request.form.items()):
            rpse["prediction"] = "Cargame algo pue!" #"Please complete the form, thank you!"
            return jsonify(rpse)

        if rpse['prediction'] == None:
            # Get the input data from the form as a list
            input_data = [float(value) for key, value in request.form.items()]
            
            ### MODEL ###
            #prediction, score = model_predict(input_data)

            prediction = "Ahora si funciono wachos!!!"
            score = 9000
            
            rpse["prediction"] = prediction
            rpse["score"] = score
            rpse["prediction_label"] = "HEALTH STATUS"
            rpse["score_label"] = "SCORE"

        return jsonify(rpse)
        
    except Exception as e:
        print('Error attempting a request.')
        print(e)

        rpse = {"success": False,
                "prediction": "",
                "score": None,
                "prediction_label": None,
                "score_label": None}
        
        return jsonify(rpse), 400