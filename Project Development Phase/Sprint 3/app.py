import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify, redirect, render_template, request, url_for

app = Flask(__name__)
model = pickle.load(open('multi.pkl','rb'))

@app.route('/predict', methods = ['POST','GET'])
def predict():
    GRE_Score = int(request.form['gre'])
    TOEFL_Score = int(request.form['toefl'])
    University_Rating = int(request.form['Rating'])
    SOP = float(request.form['sop'])
    LOR = float(request.form['lor'])
    CGPA = float(request.form['cgpa'])
    Research = int(request.form['research'])
	
    final_features = pd.DataFrame([[GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research]])

    final_features = final_features.to_numpy()
    # final_deatures = np.append(final_features, , axis=1)
    
    # scaler=MinMaxScaler()
    # final_features = scaler.fit_transform(final_features)

    prediction = model.predict(final_features)

    output = prediction[0]*100
    output = round(output[0],2)

    message = "Good luck!"
    if output < 60:
        message = "Better Luck Next Time"
    
    return render_template('predict.html', prediction_text = "Admission Chances:  {}% ".format(output), message = message, data = final_features)

if __name__ == '__main__':
    app.run(debug = True)