import pandas as pd

#loading dataset
df= pd.read_csv("student-mat.csv")

#dropping the leakage features

#removing two columns from dataframe: G1 and G2 (these are earlier grades and highly correlated with G3)
#   since keeping them would make the model 'cheat' in a sense
df=df.drop(columns=["G1", "G2"])


#Selecting features- to keep it focused on our proposal question
selected_features= ["Medu", "Fedu", "Mjob", "Fjob", #Parents background
                    "address", "Pstatus", "traveltime", "internet", #Living conditions
                    'famsup', "famsize", "famrel"] #Family demographics

#split features x, and target y
x=df[selected_features] #inputs (features)
y=df["G3"] #target - output (final grade)

#one-hot encode categorical features
#Indentifying categorical columns manually
categorical_cols= ['Mjob', 'Fjob', 'address', "Pstatus", "famsup", "internet"]
x_encoded= pd.get_dummies(x, columns=categorical_cols, drop_first=True)
#   this converts categorical values into numbers (example: Mjob=teacher -> Mjob_teacher=1)

