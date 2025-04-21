import pickle
import json
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
# this __name__ nothing but the stariting point of my application from where my application will run
app=Flask(__name__)
# so we are openeing this in read bite mode and loading this file object
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    print('Yha tk to shi hai')
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    print("ye bala part gadbad kr rha hai sayad")
    output=regmodel.predict(new_data)
    print(output)
    return jsonify(output[0])
if __name__=="__main__":
    app.run(debug=True)