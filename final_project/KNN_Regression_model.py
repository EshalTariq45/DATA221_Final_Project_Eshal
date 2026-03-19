import pandas as pd

#loading dataset
df= pd.read_csv("student-mat.csv")

#dropping the leakage features

#removing two columns from dataframe: G1 and G2
df=df.drop(columns=["G1", "G2"])


#Selecting features
selected_features= ["Medu", "Fedu", "Mjob", "Fjob", #Parents background
                    "address", "Pstatus", "traveltime", "internet", #Living conditions
                    'famsup', "famsize", "famrel"] #Family demographics

x=df[selected_features]
y=df["G3"] #target

#Indentifying categorical columns manually
categorical_cols= ['Mjob', 'Fjob', 'address', "Pstatus", "famsup", "internet"]



