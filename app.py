
from flask import *
from pickle import *

with open("Model.pkl", "rb") as f:
    model = load(f)

app = Flask(__name__)
app.secret_key = "diabetes_prediction_secret_key"


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        age = float(request.form.get("age"))
        bmi = float(request.form.get("bmi"))
        fs = float(request.form.get("fs"))
        hba1c = float(request.form.get("hba1c"))

        d0 = [age, bmi, fs, hba1c]

        gender = int(request.form.get("gender"))
        if gender == 1:
            d1 = [1, 0]
        else:
            d1 = [0, 1]

        tension = int(request.form.get("tension"))
        if tension == 1:
            d2 = [1, 0]
        else:
            d2 = [0, 1]

        family = int(request.form.get("family"))
        if family == 1:
            d3 = [1, 0]
        else:
            d3 = [0, 1]

        d = [d0 + d1 + d2 + d3]

        probability = model.predict_proba(d)[0][1] * 100

        if probability < 30:
            msg = f"""
Risk Level : LOW

Probability : {probability:.2f}%

Recommendation :
Maintain a healthy lifestyle, exercise regularly,
and continue routine health checkups.
"""

        elif probability < 70:
            msg = f"""
Risk Level : MODERATE

Probability : {probability:.2f}%

Recommendation :
Monitor blood sugar levels regularly and
consult a physician if symptoms appear.
"""

        else:
            msg = f"""
Risk Level : HIGH

Probability : {probability:.2f}%

Recommendation :
Please consult a healthcare professional
for further medical evaluation.
"""

        session["msg"] = msg

        return redirect(url_for("home"))

    msg = session.pop("msg", None)

    return render_template("home.html", msg=msg)

"""
if __name__ == "__main__":
    app.run(debug=True)
    """

