import joblib
import pandas as pd

#load the saved model 
model = joblib.load("models/logistic_regression_model.pkl")  

#change below data with new data 
new_data = pd.read_csv("")

#predict churn
prediction = model.predict(new_data)[0]

if prediction == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is NOT likely to churn")
