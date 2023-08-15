import pandas as pd
import pickle
from flask import Flask, request, render_template
from predict import predict_salary
from availables import *

scaler_model = pickle.load(open("Scaler.pkl"))
gb_model = pickle.load(open("Gradientmodel.pkl"))


@app.route('/')
def index():
    content = {
        'workclass_list':unique_workclass,
        'education_list':unique_education,
        'marital_list':unique_marital_status,
        'occupation_list':unique_occupation,
        'race_list':unique_race,
        'relationship_list':unique_relationship,
        'sex_list':unique_sex,
        'country_list':unique_country
    }
    return render_template('index.html', content = content)

@app.route('/predict', methods=['POST'])
def predict():
    inputs = {
        'age': int(request.form['age']),
        'workclass': request.form['workclass'],
        'fnlwgt': int(request.form['fnlwgt']),
        'education': request.form['education'],
        'marital-status': request.form['marital-status'],
        'occupation': request.form['occupation'],
        'relationship': request.form['relationship'],
        'race': request.form['race'],
        'sex': request.form['sex'],
        'capital-gain': int(request.form['capital-gain']),
        'capital-loss': int(request.form['capital-loss']),
        'hours-per-week': int(request.form['hours-per-week']),
        'country': request.form['country']
    }

    predicted_salary = predict_salary(inputs, scaler_model, gb_model)
    return render_template('result.html', predicted_salary=predicted_salary)


if __name__ == '__main__':
    app.run()
