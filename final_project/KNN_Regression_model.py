import pandas as pd

#loading dataset
df= pd.read_csv("student-mat.csv")

#dropping the leakage features

#removing two columns from dataframe: G1 and G2
df=df.drop(columns=["G1", "G2"])

#seperating features and targets
#dropping column G3 from x, otherwise our model would 'see' the answer while
#we're training it.
x= df.drop("G3", axis=1) #features only, axis=1: operating along columns instead of rows
y=df["G3"] #target
