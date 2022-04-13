
# importing the necessary dependencies
import numpy as np #used for numerical analysis
import pandas as pd # used for data manipulation
from flask import Flask, render_template, request
# Flask-It is our framework which we are going to use to run/serve our application.
#request-for accessing file which was uploaded by the user on our application.
import pickle
app = Flask(__name__) # initializing a flask app
model = pickle.load(open('HDI.pkl', 'rb')) #loading the model

@app.route('/')# route to display the home page
def home():
    return render_template('home.html') #rendering the home page
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])# route to show the predictions in a web UI
def predict():
    
    #reading the inputs given by the user
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['Country','Life expectancy','Mean years of schooling','Gross national income (GNI) per         capita','Internet Users']
    
    df = pd.DataFrame(features_value, columns=features_name)
    
    # predictions using the loaded model file
    output = model.predict(df)
    print(round(output[0][0],2))
    print(type(output))
    y_pred =round(output[0][0],2)
    if(y_pred >= 0.3 and y_pred <= 0.4) :
        return render_template("resultnew.html",prediction_text = 'Low HDI'+ str(y_pred))
    elif(y_pred >= 0.4 and y_pred <= 0.7) :
        return render_template("resultnew.html",prediction_text = 'Medium HDI '+str(y_pred))
    elif(y_pred >= 0.7 and y_pred <= 0.8) :
        return render_template("resultnew.html",prediction_text = 'High HDI'+str(y_pred))
    elif(y_pred >= 0.8 and y_pred <= 0.94) :
        return render_template("resultnew.html",prediction_text = 'Very High HDI'+str(y_pred))
    else :
        return render_template("resultnew.html",prediction_text = 'The given values do not match the range of values of the model.Try giving the values in the mnetioned range'+str(y_pred))
    
    
    # showing the prediction results in a UI# showing the prediction results in a UI
    return render_template('resultnew.html', prediction_text=output)


if __name__ == '__main__':
    # running the app
    app.run(debug=True,port=5000)
