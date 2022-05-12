from flask import Flask, render_template, request,redirect
from flask_cors import CORS,cross_origin
app = Flask(__name__)
import pickle
import pandas as pd
import numpy as np

file = open('model.pkl','rb')
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')
clf = pickle.load(file)
file.close()



@app.route('/')
def home():
  return render_template('home.html')

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/contact')
def contact():
  return render_template('contact.html')

@app.route('/services')
def project():
  return render_template('services.html')




# CORONA TEST 
@app.route('/corona_project', methods=["GET","POST"])
def corona_project():

  if request.method == "POST":
    mydict = request.form
    fever = int(mydict['fever'])
    age = int(mydict['age'])
    pain = int(mydict['pain'])
    runnynose = int(mydict['runnynose'])
    diffbreath= int(mydict['diffbreath'])


     # code for inference 
    inputfeatures = [fever,pain,age,runnynose,diffbreath]
    infprob = clf.predict_proba([inputfeatures])[0][1]
    print(infprob)
    return render_template('corona_res.html',inf=round(infprob*100))
    # return render_template('show.html', inf=round(infProb*100))

  return render_template('corona.html')
        # return 'Hello, World' + str(infprob)


# SECOND HAND CAR PRICE PREDICTOR
@app.route('/car',methods=['GET','POST'])
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('car.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))





if __name__ == "__main__":
    app.run(debug=True)