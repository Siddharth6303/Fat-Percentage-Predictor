import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
df=pd.read_csv('bodyfat1.csv')
df=df.drop(columns=['Density'])
print(df.columns)
df=df.dropna()
x=df.drop(columns='BodyFat')
y=df['BodyFat']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)
lr=LinearRegression()
lr.fit(x_train.values,y_train.values)
pred_y=lr.predict(x_test)
print(r2_score(y_test,pred_y))
print(lr.predict([[49,86.97644038,65,38.4,118.5,113.1,113.8,61.9,38.3,21.9,32,29.8,17]]))