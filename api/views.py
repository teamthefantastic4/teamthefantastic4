from flask import (
    Blueprint,
    jsonify,
    render_template,
    request,
)


router = Blueprint("app_router", __name__, template_folder="templates")

@router.route("/", methods=["GET"])
def index():
    if request.method == "GET":
        return render_template("index.html")


@router.route("/", methods=['POST'])
def result():
    try:
        null_data = [0, '0', None, '']
        prediction = 'Please complete the form, thank you.'
        score = 0
        rpse = {"success": False, "prediction": None, "score": None}

        # check if the form has 0 or null values
        for x, i in request.form.items():
            if i in null_data:
                rpse["success"] = False
                rpse["prediction"] = prediction
                rpse["score"] = score
                break
        
        ###########################################################################
        # REPLACE THIS PART WITH THE MODEL replace this part with the model
        if rpse['prediction'] == None:
            total = 0
            for x, i in request.form.items():
                total += float(i)
                if float(i) > score:
                    score = float(i)
        
            score = (score * 100) / total

            if score >= 80:
                prediction = 'Very healthy'
            elif score < 80 and score >= 70:
                prediction = 'Healthy'
            elif score < 70 and score >= 50:
                prediction = 'Stable, but it is recommended that you see a doctor.'
            else:
                prediction = 'Please see your doctor!'
        ###########################################################################
            rpse["success"] = True
            rpse["prediction"] = prediction
            rpse["score"] = score

        return jsonify(rpse)
        
    except Exception as e:
        print(f'Error: {e}')
        return jsonify(rpse), 400