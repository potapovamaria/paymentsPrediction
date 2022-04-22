import os

import prediction_sigma

from flask import Flask, Response, jsonify, request, redirect, url_for, flash
import pandas as pd

def create_application() -> Flask:
    app = Flask(__name__)
    @app.route("/getpred", methods=["POST"])
    def dummy():
        if request.method == 'POST':
            file = request.files['file']
            file.save(file.filename)
            # model_num = request.form.get("model_num")
            # model_num = int(model_num)
            y_pred = prediction_sigma.get_answer(file.filename, 1)
            y_pred = y_pred.to_dict()
            y_pred = y_pred['PAY']
            # y_pred = y_pred.to_json(orient="records")
            # print(y_pred)
            # print(y_pred.to_json)
            # index = y_pred.index
            # index = index.values.tolist()
            # y_pred = y_pred.values.tolist()
            # return jsonify(y_pred), jsonify(index)
            return jsonify(y_pred)
    return app

if __name__== "__main__":
    app = create_application()
    app.run(host="127.0.0.1", port=5000, debug=True)