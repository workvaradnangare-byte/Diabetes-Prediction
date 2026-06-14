import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from pickle import dump

PatientData    =   pd.read_csv("diabetes_feb26.csv")
#print(PatientData.isnull().sum())

PatientData.fillna({'Fasting_Sugar': PatientData['Fasting_Sugar'].median()}, inplace= True)
PatientData.fillna({'HbA1c':PatientData['HbA1c'].median()}, inplace=True)
#print(PatientData.isnull().sum())


FeaturesOfData  =   PatientData.drop(['Diabetes'], axis= 'columns')
TargetOfData    =   PatientData['Diabetes']
"""
print(FeaturesOfData)
print(TargetOfData)
"""
Categorical_Columns =   ['Gender','Hypertension','Family_History']
Numerical_Columns   =   ['Age','BMI','Fasting_Sugar','HbA1c']

encoder     =   OneHotEncoder(sparse_output= False)

Encoded = encoder.fit_transform(PatientData[Categorical_Columns])

encoded_df = pd.DataFrame(Encoded,columns=encoder.get_feature_names_out(Categorical_Columns))

print(encoded_df)


NewFeaturesOfData  =   pd.concat([PatientData[Numerical_Columns], encoded_df], axis= 1)


x_train, x_test, y_train, y_test    =   train_test_split(NewFeaturesOfData, TargetOfData)


model   =   LogisticRegression()
model.fit(x_train,y_train)

Report  =   classification_report(y_test, model.predict(x_test))
print(Report)

with open("Model.pkl", "wb") as f:
    dump(model, f)
    print("Model Saved")