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
#kullanıcı girişi
#Hava Durumu Değerleri oluşturma
hava = int(input("For Sunny Press 1\nFor Rainy Press 2\nFor Overcast Press 3\n"))
hava_df = pd.DataFrame(data=[0],index =range(0,1), columns=['Sunny'])
hava_df2 = pd.DataFrame(data=[0],index =range(0,1), columns=['Rainy'])
hava_df3 = pd.DataFrame(data=[0],index =range(0,1), columns=['Overcast'])

if hava==1:
    hava_df=hava_df.replace(0,1)
    weather_user_data = pd.concat([hava_df3,hava_df2,hava_df],axis=1)
if hava==2:
    hava_df2=hava_df2.replace(0,1)
    weather_user_data = pd.concat([hava_df3,hava_df2,hava_df],axis=1)
if hava==3:
    hava_df3=hava_df3.replace(0,1)
    weather_user_data = pd.concat([hava_df3,hava_df2,hava_df],axis=1) 
#print (weather_user_data)
#Temprature Sorgu
sicaklik = int(input("Hava Sıcaklığı Kaç Derece(F):"))
sicaklik_df=pd.DataFrame(data=[sicaklik],index=range(0,1),columns=['Temprature'])
#print (sicaklik_df)
#Nem Sorgu
nem = int(input("Havadaki Nem Oranı Kaç:"))
nem_df=pd.DataFrame(data=[nem],index=range(0,1),columns=['Humidity'])
#print (nem_df)
#Wind Sorgu
ruzgar_df=pd.DataFrame(data=[0],index=range(0,1),columns=['Windy'])
ruzgar = int(input("Havada Rüzgar Var Mı?\nVarsa 1'e Yoksa 2'ye Basınız:"))
if ruzgar==1:
    ruzgar_df=ruzgar_df.replace(0,1)
else:
    ruzgar_df=ruzgar_df.replace(0,0)
#print (ruzgar_df)
#UserData Create
data_user=pd.concat([weather_user_data,sicaklik_df,nem_df,ruzgar_df],axis=1)
#print (data_user)
    
    
#Predict From User Data
predicted = regressor.predict(data_user)
arr=np.arange(1)
a= arr.tolist()

predict_yn = a
for x in range (0,1):
    if predicted[x]>=0.7:
        predict_yn[x]="yes"
    else:
        predict_yn[x]="no"
        
if predict_yn[0]=="yes":
    print("Tenis Oynanabilir")
if  predict_yn[0]=="no":
    print("Bu Havada Tenis mi Olur mk çocu")




