# Small web app that uses the trained model
import joblib
from flask import Flask, request

app = Flask(__name__)
saved = joblib.load("model.joblib")
model = saved["model"]
NAMES = ["Setosa", "Versicolor", "Virginica"]

PAGE = """
<html>
<body style="text-align:center;font-family:Verdana;margin:1%;">
  <h1 style="color:navy">Iris Flower Predictor</h1>
  <p>ML model deployed from GitHub to EC2 with AWS CodePipeline</p>
  <h3>Model Version: {version} &nbsp; | &nbsp; Accuracy: {accuracy}</h3>
  <hr>
  <form method="post">
    Sepal length <input name="f1" value="5.1" size="4">
    Sepal width  <input name="f2" value="3.5" size="4"><br><br>
    Petal length <input name="f3" value="1.4" size="4">
    Petal width  <input name="f4" value="0.2" size="4"><br><br>
    <button type="submit">Predict</button>
  </form>
  <h2 style="color:green">{result}</h2>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        try:
            values = [[float(request.form[f]) for f in ("f1", "f2", "f3", "f4")]]
            result = "Prediction: " + NAMES[int(model.predict(values)[0])]
        except ValueError:
            result = "Please enter numbers only"
    return PAGE.format(version=saved["version"], accuracy=saved["accuracy"], result=result)
