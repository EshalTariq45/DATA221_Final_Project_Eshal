import pandas as pd

#loading dataset
df= pd.read_csv("student-mat.csv")

#dropping the leakage features

#removing two columns from dataframe: G1 and G2
df=df.drop(columns=["G1", "G2"])

#seperating features and targets

#Selecting features
selected_features= ["Medu", "Fedu", "Mjob", "Fjob", #Parents background
                    "adress", "Pstatus", "traveltime", "internet", #Living conditions
                    'famsup', "famsize", "famrel"] #Family demographics
y=df["G3"] #target


