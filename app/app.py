from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href2='')
    else:
        myage = request.form['age']
        mygender = request.form['gender']
        myacdemic = request.form['academic qualification']

        model = load('app/music-recommender.joblib')

        music = pd.read_csv('music-type.csv')
        label_encoder = LabelEncoder()
        music_encoded = label_encoder.fit_transform(music["academic-qualification"]).reshape(-1, 1)
        myacdemic_encoded = label_encoder.transform([myacdemic])

        np_arr = np.array([myage, mygender, myacdemic_encoded])
        predictions = model.predict([np_arr])  
        predictions_to_str = str(predictions).replace("[", "").replace("]", "").replace("'", "")
        #return predictions_to_str
        return render_template('index.html', href2='The suitable music type for you (age:'+str(myage)+', gender:'+str(mygender)+', academic qualification:'+str(myacdemic)+') is: '+predictions_to_str)

