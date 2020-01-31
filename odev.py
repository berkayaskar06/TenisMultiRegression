# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#veri yükleme
data = pd.read_csv('tenis.csv')
print ("Datanın kendisi: \n",data)

#Encode Weather to Numeric
weather = data.iloc[:,0]
Encoder = LabelEncoder()
weather = Encoder.fit_transform(weather).reshape(-1,1)

ohe = OneHotEncoder(categorical_features='all')

weather= ohe.fit_transform(weather).toarray()

#Encode Wind to Numeric

wind = data.iloc[:,3].values
wind =Encoder.fit_transform(wind).reshape(-1,1)

play= data.iloc[:,-1].values
play = Encoder.fit_transform(play).reshape(-1,1)
#DataFrame Oluşturma

weather_df = pd.DataFrame(data=weather ,index= range(14) , columns = ['overcast','rainy','sunny'])
wind_df = pd.DataFrame(data=wind, index = range(14),columns=['wind'])
play_df =pd.DataFrame(data=play, index = range(14),columns=['play'])
others = data.drop(['outlook','windy','play'],axis=1)

#DataFrame Birleştirme

data_en = pd.concat([weather_df,others,wind_df,play_df],axis =1)


#Train & Test
x_train,x_test,y_train,y_test = train_test_split(data_en.drop(['play'],axis=1),play_df,test_size=0.33,random_state=0)

#MultiRegression 

regressor = LinearRegression()
regressor.fit(x_train,y_train)
predicted = regressor.predict(x_test)
arr=np.arange(5)
a= arr.tolist()

predict_yn = a
for x in range (5):
    if predicted[x]>=0.7:
        predict_yn[x]="yes"
    else:
        predict_yn[x]="no"
        
print (predict_yn)




