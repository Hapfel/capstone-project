from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
#import config
import pickle
import io
#from utils.model import ResNet9

# Loading model

with open('models/RF_pkl' , 'rb') as f:
    crop_recommendation_model = pickle.load(f)

# =========================================================================================

# Custom functions for calculations

#Terry to code
def weather_fetch(long,lat,year):
    """
    Fetch and returns the temperature and humidity of a location
    :params: longitute, latitude, year
    :return: temperature, humidity in given year
    """
#Hank to code
def soil_fetch(long,lat):
    """
    Fetch and returns the N and ph of a location
    :params: longitute, latitude
    :return: N, ph in given location
    """

# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'CropPick'
    return render_template('index.html', title=title)

@ app.route('/contact')
def contact():
    title = 'Contact'
    return render_template('contact.html', title=title)

@ app.route('/climate')
def climate():
    title = 'climate'
    return render_template('climate.html', title=title)

# render crop recommendation form page

@ app.route('/Recommend')
def crop_recommend():
    title = 'CropPick - Recommend'
    return render_template('Recommend.html', title=title)

# RENDER PREDICTION PAGE

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'CropPick Recommendations'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        longitude = float(request.form.get("longitude"))
        latitude = float(request.form.get("latitude"))
        
        #override for test
        temperature = 80
        humidity = 0.8

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('results.html', prediction=final_prediction, l=latitude, title=title)

        #end override for test
        
        #use below when weather is functional
        """ if weather_fetch(long,lat,year) != None:
            temperature, humidity = weather_fetch(long,lat,year)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title) """

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)