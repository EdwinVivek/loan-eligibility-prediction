import joblib
from flask import Flask, request, jsonify, render_template
import pandas as pd
from custom_transformers import SimpleImputerWithMapping, CustomBinning
import numpy as np

import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", prediction=None, error=None), 200

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        print("hello")
        try:
            data = request.form  
            loan = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), 'data\\train_loanpred.csv')), nrows=1, usecols=lambda x: x not in ["Loan_ID", "Loan_Status"])
            numeric_cols = loan.select_dtypes(include=['int64', 'float64']).columns.to_list()
            features = loan.columns.to_list()
            response = {key: float(data[key]) if key in numeric_cols else data[key] for key in features}
            #response = {key: data[key] for key in features} 
            #response = np.array(response).reshape(1, -1)
            lr_model = joblib.load("rf_loan_prediction.pkl")
            df = pd.DataFrame([response])
            prediction = lr_model.predict(df)
            result = 'Approved' if str(prediction)[2] == 'Y' else 'Rejected'
            prob_1 = np.round(lr_model.predict_proba(df).ravel()[1], 2) * 100
            return render_template('home.html', prediction=result, probability=prob_1)
        except Exception as e:
            return render_template('home.html', error=str(e))
    if request.method == 'GET':
        return render_template('home.html',  prediction=None, error=None)



if __name__ == "__main__":
    app.run(debug=True, port=5555)

