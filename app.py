import flask
from flask import Flask, render_template
import pickle
import pandas as pd


# Use pickle to load in the pre-trained model
with open(f'model/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/main', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        age = flask.request.form['age']
        gender = flask.request.form['gender']
        height = flask.request.form['height']
        weight = flask.request.form['weight']
        bp_hi = flask.request.form['bp_hi']
        bp_lo = flask.request.form['bp_lo']
        cholesterol = flask.request.form['cholesterol']
        gluc = flask.request.form['gluc']
        smoke = flask.request.form['smoke']
        alco = flask.request.form['alco']
        active = flask.request.form['active']
        # Make DataFrame for model
        input_variables = pd.DataFrame([[age, gender, height,weight, bp_hi,bp_lo, cholesterol,gluc,smoke,alco,active]],
                                       columns=['age', 'gender', 'height','weight','bp_hi','bp_lo','cholesterol','gluc','smoke','alco','active'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        
        prediction = model.predict(input_variables)[0]
        if prediction == 0:
            first = "No Heart Disesase!"

        else:
            first = "You are at Risk of Heart Disease!"
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'Age':age,
                                                     'Gender':gender,
                                                     'Height':height,
                                                     'Weight':weight,
                                                     'Systolic BP':bp_hi,
                                                     'Diastolic BP':bp_lo,
                                                     'Cholesterol':cholesterol,
                                                     'gluc':gluc,
                                                     'smoke':smoke,
                                                     'alco':alco,
                                                     'active':active},
                                     result=first
                                     )



if __name__ == '__main__':
    app.run()
