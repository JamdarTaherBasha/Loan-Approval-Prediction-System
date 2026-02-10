# importing Required Modules

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

#Load the Dataset

df=pd.read_csv("data/loan_approval_dataset.csv")

df.columns = df.columns.str.strip()


print("column\n",df.columns)

#fill missing values

df.fillna(df.mode().iloc[0],inplace=True)

#Encode Categorical Variables

le=LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col]=le.fit_transform(df[col])


# Target column(check exact name from print output)

target_column="loan_status"

x=df.drop(target_column,axis=1)

y=df[target_column]

# Train_test split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Model 

Model=RandomForestClassifier()

Model.fit(x_train,y_train)

# Evaluation

pred=Model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,pred))

print("Confusion Matrix:\n",confusion_matrix(y_test,pred))




import pickle

# Save trained model

with open("loan_model.pkl","wb") as f:
    pickle.dump(Model,f)

print("Model saved as loan_model.pkl")



