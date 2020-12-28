"""
Created on Sun Dec 27 22:08:35 2020

@author: Ronith Raj
"""

from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)
# model = pickle.load(open('knn_heart_disease.pkl', 'rb'))
model = pickle.load(open('logistic_heart_disease_final.pkl', 'rb'))
std_age = pickle.load(open('age_scaler.pkl','rb'))
std_trestbps = pickle.load(open('trestbps_scaler.pkl','rb'))
std_chol = pickle.load(open('chol_scaler.pkl','rb'))
std_thalach = pickle.load(open('thalach_scaler.pkl','rb'))
std_oldpeak = pickle.load(open('oldpeak_scaler.pkl','rb'))

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        sex = request.form['gender']
        if sex == "0":
            sex_1 = 0
        else:
            sex_1 = 1
        age = float(request.form['age'])
        blood_pressure = float(request.form['trestbps'])
        cholestrol = float(request.form['chol'])
        heart_rate = float(request.form['thalach'])
        old_peak = float(request.form['peak'])
        chest_pain = float(request.form['chest'])
        if chest_pain =="0":
            cp_1 = 0
            cp_2 = 0
            cp_3 = 0
        elif chest_pain =="1":
            cp_1 = 1
            cp_2 = 0
            cp_3 = 0
        elif chest_pain =="2":
            cp_1 = 0
            cp_2 = 1
            cp_3 = 0
        else:
            cp_1 = 0
            cp_2 = 0
            cp_3 = 1
        blood_sugar = request.form['fbs']
        if blood_sugar == "1":
            fbs_1 = 1
        else: 
            fbs_1 = 0
        ecg = request.form['restecg']
        if ecg == "0":
            restecg_1 = 0
            restecg_2 = 0
        elif ecg == "1":
            restecg_1 = 1
            restecg_2 = 0
        else:
            restecg_1 = 0 
            restecg_2 = 1
        exercise = request.form['exang']
        if exercise == "0":
            exang_1 = 0
        else:
            exang_1 = 1
        vessels = request.form['ca']
        if vessels == "0":
            ca_1 = 0
            ca_2 = 0
            ca_3 = 0
            ca_4 = 0
        elif vessels == "1":
            ca_1 = 1
            ca_2 = 0
            ca_3 = 0
            ca_4 = 0
        elif vessels == "2":
            ca_1 = 0
            ca_2 = 1
            ca_3 = 0
            ca_4 = 0
        elif vessels == "3":
            ca_1 = 0
            ca_2 = 0
            ca_3 = 1
            ca_4 = 0
        else:
            ca_1 = 0
            ca_2 = 0
            ca_3 = 0
            ca_4 = 1
        thalassemia = request.form['thal']
        if thalassemia =="0":
            thal_1 = 0
            thal_2 = 0
            thal_3 = 0
        elif thalassemia =="1":
            thal_1 = 1
            thal_2 = 0
            thal_3 = 0
        elif thalassemia =="2":
            thal_1 = 0
            thal_2 = 1
            thal_3 = 0
        else:
            thal_1 = 0
            thal_2 = 0
            thal_3 = 1
        slope = request.form['slope']
        if slope == "0":
            slope_1 = 0
            slope_2 = 0
        elif slope == "1":
            slope_1 = 1
            slope_2 = 0
        else:
            slope_1 = 0
            slope_2 = 1
        # std_scalar = StandardScaler()
        # columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        age = std_age.transform(np.array(age).reshape(-1,1))
        blood_pressure = std_trestbps.transform(np.array(blood_pressure).reshape(-1,1))
        cholestrol = std_chol.transform(np.array(cholestrol).reshape(-1,1))
        heart_rate = std_thalach.transform(np.array(heart_rate).reshape(-1,1))
        old_peak = std_oldpeak.transform(np.array(old_peak).reshape(-1,1))

        prediction = model.predict_proba([[
            age,
            blood_pressure,
            cholestrol,
            heart_rate,
            old_peak,
            sex_1,
            cp_1,
            cp_2,
            cp_3,
            fbs_1,
            restecg_1,
            restecg_2,
            exang_1,
            slope_1,
            slope_2,
            ca_1,
            ca_2,
            ca_3,
            ca_4,
            thal_1,
            thal_2,
            thal_3
        ]])

        yes_chance = round(prediction[0][1]*100,2)
        if yes_chance < 50.00:
            if yes_chance < 1.0:
                return render_template("index.html", result = "Congratulations!You donot have any chance of getting a heart disease.",news = "Stay happy and safe!")
            else:
                return render_template("index.html", result = "Congratulations!You have a low chance of getting a heart disease which is : {}.".format(yes_chance),news = "Stay happy and safe!")
        else:
            return render_template("index.html", result = "Sorry!Your chance of getting a heart disease is : {}.".format(yes_chance),news = "Get well soon!")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
