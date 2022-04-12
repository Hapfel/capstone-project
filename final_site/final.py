from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
#import config
import pickle
import io
import os
#from utils.model import ResNet9
from datetime import date

# Loading model

with open('models/RF_pkl' , 'rb') as f:
    crop_recommendation_model = pickle.load(f)

# Reading climate data
df = pd.read_csv('Localized_Weather_Clean.csv')
df_wb = pd.read_csv('WB_climate_prediction_Clean.csv')    
df_cp = pd.read_csv('crop_data_new.csv')

# =========================================================================================

# Custom functions for calculations

def weather_fetch(long,lat,year):
    """
    Fetch and returns the temperature and humidity of a location
    :params: longitute, latitude, year
    :return: temperature, humidity in given year
    """
    # Find proxy weather station for location inputed by users
    # Based on smallest deviation in latitude/longitude between the two locations
    # As measured by Euclidean distance (Pythagorean distance).

    df["deviation"] = np.sqrt(np.power(2, np.abs(lat - df["Latitude"])) + np.power(2, np.abs(long - df["Longitude"])))
    smallest_deviation = df["deviation"].min()
    idx_smallest_deviation = df["deviation"].idxmin()
  
    # Get Annual mean temperature of proxy station
    # Based on historical hourly weather reports collected by weatherspark.com
    # From January 1, 1980 to December 31, 2016

    proxy = df["PROVINCE"].loc[idx_smallest_deviation]
    baseline_temp = df["Avg_Mean"].loc[idx_smallest_deviation]
    baseline_rainfall = df["Annual_Rainfall"].loc[idx_smallest_deviation]

    # Calculate changes in temperature and rainfall for prediction period
    # Based on changes in predicted province

    current_year = date.today().year
    predicted_year_temp = df_wb.loc[(df_wb["province"] == proxy) & (df_wb["year"] == year)]["avg_tmp"].item()
    current_year_temp = df_wb.loc[(df_wb["province"] == proxy) & (df_wb["year"] == current_year)]["avg_tmp"].item()
    temp_delta =  predicted_year_temp - current_year_temp

    predicted_year_rainfall = df_wb.loc[(df_wb["province"] == proxy) & (df_wb["year"] == year)]["rainfall"].item()
    current_year_rainfall = df_wb.loc[(df_wb["province"] == proxy) & (df_wb["year"] == current_year)]["rainfall"].item()
    rainfall_delta = predicted_year_rainfall - current_year_rainfall

    # Calculate temperature and rainfall in predicted year
    output_temp = baseline_temp + temp_delta
    output_rainfall = baseline_rainfall + rainfall_delta

    return output_temp, output_rainfall 

def runmodel(model,input_array):
    #Predicting probabilities associated with all classes
    # Need to ensure that sklearn method works compatably with .pickle file 
    predicted_probs = model.predict_proba(input_array)

    #reshaping result into a (class number,) numpy array rather than a (1,class number) array
    predicted_probs = predicted_probs.reshape(-1)

    #getting relevant indices
    top_class_index = np.argsort(predicted_probs)
    top_class_index = top_class_index[-3:]

    classes = model.classes_

    top_class_names = np.array(classes)[top_class_index]
    top_class_probs = np.array(predicted_probs)[top_class_index]

    #returning a dataframe of the sorted class names and probabilities associated
    result = {'Crop':top_class_names[::-1], 'Model Probability':top_class_probs[::-1]}
    frame = pd.DataFrame(result)
    return frame

def format_array(input_array):
    input_namevec = ["Nitrogen","Phosphorus","Potassium","Temperature","Humidity","Ph","Rainfall"]
    user_input_formatted = pd.DataFrame(input_array,columns=input_namevec)
    return user_input_formatted

def gen_display_frame(model,user_input_array,crop_data):
    top_3 = runmodel(model,user_input_array)
    formatted_user_input = format_array(user_input_array)

    avg_by_crops = crop_data.groupby(by='label',axis=0).mean()
    avg_by_crops['Revenue_per_acre'] = avg_by_crops.apply(lambda row: row.Price_kg*row.kg_per_acre,axis=1)
    avg_by_crops = avg_by_crops.loc[:,avg_by_crops.columns!='Price_kg']

    avg_by_crops = avg_by_crops[avg_by_crops.index.get_level_values('label').isin(top_3['Crop'])]

    final_crop_data = top_3.join(avg_by_crops,on='Crop',how='inner')
    final_crop_data = final_crop_data.round(2)

    final_crop_data = final_crop_data.rename({"Crop":"CropPick Recommendation","temperature":"Temp(C)","humidity":"Humidity","rainfall":"Rainfall(cm)","kg_per_acre":"Production per acre(kg)","Revenue_per_acre":"Revenue per acre($)"},axis=1)
    formatted_user_input = formatted_user_input.rename({"Nitrogen":"N","Phosphorus":"P","Potassium":"K","Temperature":"Temp(C)","Rainfall":"Rainfall(cm)"},axis=1).round(2)
    formatted_user_input.index = ({"Your Input":"Your Input"})

    return formatted_user_input, final_crop_data

# ------------------------------------ FLASK APP -------------------------------------------------

IMAGE_FOLDER = os.path.join('static', 'images')




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# render home page


@ app.route('/')
def home():
    title = 'CropPick'
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'mango.png')
    return render_template('index.html', image=full_filename, title=title)

@ app.route('/contact')
def contact():
    title = 'Contact'
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'mango.png')
    return render_template('contact.html', title=title)

@ app.route('/climate')
def climate():
    title = 'climate'
    return render_template('climate.html', title=title)

# render crop recommendation form page

@ app.route('/Recommend')
def crop_recommend():
    title = 'CropPick - Recommend'
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'mango.png')
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
        year = int(request.form['year'])
        humidity = float(request.form['humidity']) 
        longitude = float(request.form.get("longitude"))
        latitude = float(request.form.get("latitude"))
        temperature = request.form['temperature']
        rainfall = request.form['rainfall']

        if temperature == "":
            temperature = weather_fetch(longitude, latitude, year)[0]
        if rainfall == "":    
            rainfall = weather_fetch(longitude, latitude, year)[1]/10
        
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        mod = crop_recommendation_model
        
        final_prediction = gen_display_frame(mod, data, df_cp)
        top_3 = final_prediction[1]
        user_input = final_prediction[0]

        # Retrieve data from top_3
        t0a = top_3.iloc[[0],[0]].values.tolist()[0][0]
        t0b = top_3.iloc[[0],[1]].values.tolist()[0][0]
        t0c = top_3.iloc[[0],[2]].values.tolist()[0][0]
        t0d = top_3.iloc[[0],[3]].values.tolist()[0][0]
        t0e = top_3.iloc[[0],[4]].values.tolist()[0][0]
        t0f = top_3.iloc[[0],[5]].values.tolist()[0][0]
        t0g = top_3.iloc[[0],[6]].values.tolist()[0][0]
        t0h = top_3.iloc[[0],[7]].values.tolist()[0][0]
        t0i = top_3.iloc[[0],[8]].values.tolist()[0][0]
        t0j = top_3.iloc[[0],[9]].values.tolist()[0][0]
        t0k = top_3.iloc[[0],[10]].values.tolist()[0][0]

        t1a = top_3.iloc[[1],[0]].values.tolist()[0][0]
        t1b = top_3.iloc[[1],[1]].values.tolist()[0][0]
        t1c = top_3.iloc[[1],[2]].values.tolist()[0][0]
        t1d = top_3.iloc[[1],[3]].values.tolist()[0][0]
        t1e = top_3.iloc[[1],[4]].values.tolist()[0][0]
        t1f = top_3.iloc[[1],[5]].values.tolist()[0][0]
        t1g = top_3.iloc[[1],[6]].values.tolist()[0][0]
        t1h = top_3.iloc[[1],[7]].values.tolist()[0][0]
        t1i = top_3.iloc[[1],[8]].values.tolist()[0][0]
        t1j = top_3.iloc[[1],[9]].values.tolist()[0][0]
        t1k = top_3.iloc[[1],[10]].values.tolist()[0][0]

        t2a = top_3.iloc[[2],[0]].values.tolist()[0][0]
        t2b = top_3.iloc[[2],[1]].values.tolist()[0][0]
        t2c = top_3.iloc[[2],[2]].values.tolist()[0][0]
        t2d = top_3.iloc[[2],[3]].values.tolist()[0][0]
        t2e = top_3.iloc[[2],[4]].values.tolist()[0][0]
        t2f = top_3.iloc[[2],[5]].values.tolist()[0][0]
        t2g = top_3.iloc[[1],[6]].values.tolist()[0][0]
        t2h = top_3.iloc[[1],[7]].values.tolist()[0][0]
        t2i = top_3.iloc[[1],[8]].values.tolist()[0][0]
        t2j = top_3.iloc[[1],[9]].values.tolist()[0][0]
        t2k = top_3.iloc[[1],[10]].values.tolist()[0][0]

        # Retrieve data from user_input
        u0a = user_input.iloc[[0],[0]].values.tolist()[0][0]
        u0b = user_input.iloc[[0],[1]].values.tolist()[0][0]
        u0c = user_input.iloc[[0],[2]].values.tolist()[0][0]
        u0d = user_input.iloc[[0],[3]].values.tolist()[0][0]
        u0e = user_input.iloc[[0],[4]].values.tolist()[0][0]
        u0f = user_input.iloc[[0],[5]].values.tolist()[0][0]
        u0g = user_input.iloc[[0],[6]].values.tolist()[0][0]        

        return render_template('final_results.html', 
                                T0A=t0a, T0B=t0b, T0C=t0c, T0D=t0d, T0E =t0e, T0F=t0f, T0G=t0g, T0H=t0h, T0I=t0i, T0J=t0j, T0K =t0k, 
                                T1A=t1a, T1B=t1b, T1C=t1c, T1D=t1d, T1E =t1e, T1F=t1f, T1G=t1g, T1H=t1h, T1I=t1i, T1J=t1j, T1K =t1k,
                                T2A=t2a, T2B=t2b, T2C=t2c, T2D=t2d, T2E =t2e, T2F=t2f, T2G=t2g, T2H=t2h, T2I=t2i, T2J=t2j, T2K =t2k,
                                U0A=u0a, U0B=u0b, U0C=u0c, U0D=u0d, U0E =u0e, U0F=u0f, U0G=u0g,
                                title=title)
        
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