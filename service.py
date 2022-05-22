import os

import prediction_sigma

from flask import Flask, Response, jsonify, request, redirect, url_for, flash
import pandas as pd

def create_application() -> Flask:
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    @app.route("/prediction/payment", methods=["POST"])
    def dummy():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
            file = request.files['file']
            file.save(file.filename)
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            # model_num = request.form.get("model_num")
            # model_num = int(model_num)
            pick_check = request.form.get('enter_pick')
            y_pred = prediction_sigma.get_answer(file.filename, 1, start_date, end_date, pick_check)
            y_pred = y_pred.to_dict()
            y_pred = y_pred['PAY']
            return jsonify(y_pred)
    return app

if __name__== "__main__":
    app = create_application()
    app.run(host="localhost", port=5000, debug=True)