# importing required libraries
from flask import Flask, request, render_template
import numpy as np
import os
import warnings
import pickle
from feature import FeatureExtraction  # Make sure feature.py has a class FeatureExtraction(url)

warnings.filterwarnings('ignore')

# Load the trained model from the pickle file
base_path = os.path.dirname(os.path.abspath(__file__))  
model_path = os.path.join(base_path, 'pickle', 'model6.pkl')

with open(model_path, "rb") as file:
    forest = pickle.load(file)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)  # This should accept the URL as input
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = forest.predict(x)[0]
        y_pro_non_phishing = forest.predict_proba(x)[0, 1]

        return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url)

    return render_template("index.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/faq.html")
def faq():
    return render_template("faq.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
