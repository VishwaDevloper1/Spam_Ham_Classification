from flask import Flask, render_template, request, redirect, url_for
from src.Components.main import SpamClassifier
import os

app = Flask(__name__)

classifier = SpamClassifier(data_path="data/SMSSpamCollection.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET" and classifier.trained_model is None:
        classifier.train_model()

    if request.method == "POST":
        sms_input = request.form["sms_input"]
        prediction = classifier.predict_message(sms_input)
        return redirect(url_for("index", prediction=prediction, sms_input=sms_input))
    prediction = request.args.get("prediction")
    sms_input = request.args.get("sms_input")

    return render_template("index.html", prediction=prediction, sms_input=sms_input)



if __name__ == "__main__":
    app.run(debug=True)
