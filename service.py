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
            # file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file.filename)
            # file = "./pay2021-11-24.csv"
            y_pred = prediction_sigma.get_answer(file.filename)
            y_pred = y_pred.values.tolist()
            return jsonify(y_pred)
    return app

if __name__== "__main__":
    app = create_application()
    app.run(host="127.0.0.1", port=5000, debug=True)