# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:23:27 2019

@author: ASPIRE E 14
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\Users\ASPIRE E 14\Music\DTS FGA 2019 - Unmul\projek akhir\diabetes.csv")
data.head()

data.dtypes #Tipe Data yakni semuanya integer atau data numerik

#Langkah awal adalah melakukan analisis statistika deskriptif dengan melihat Max,Min,Mean,Standar deviasi
data.describe()

#Kemudian mengehitung banyaknya Status atau kelompok layak dan tidak layak pada data
data['Diabetic'].value_counts() #Status '0'=Tidak dan '1'=Diabet

#bagi data menjadi 2 bagian yakni variabel terikat (y) dan variabel bebas (x)
y=data['Diabetic'].values
x=data[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values
#membuat histogram dari variabel bebas
plt.hist(x)
#dari histogram dapat diinterpretasikan bahwa data tidak mengikuti sebaran yang normal sehingga perlu dinormalisasi

#Proses Normalisasi data
from sklearn import preprocessing
x_norm=preprocessing.StandardScaler().fit(x).transform(x.astype(float))
plt.hist(x_norm)
#Setelah dinormalisasi data mengikuti sebaran yang normal

#Menentukan data training dan data testing dengan perbandingan 80:20
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_norm,y,test_size=0.2)
print('Banyaknya Data Training:',x_train.shape,y_train.shape)
print('Banyaknya Data Testing:',x_test.shape,y_test.shape)
#Proses KNN dengan k=2
from sklearn.neighbors import KNeighborsClassifier
k=2
KNN=KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)

#Hasil Prediksi
y_predict=KNN.predict(x_test)
y_predict

#perbandingan data aktual dan data prediksi
print('Data Aktual:  ',y_test)
print('Data Prediksi:',y_predict)

#menghitung nilai akurasi, semakin besar akurasi maka prediksi mendekati aktualnya
from sklearn import metrics
print('Akurasi:',metrics.accuracy_score(y_test,y_predict))

#Melakukan 10 kemungkinan nilai k
hasil=[]
for i in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=i).fit(x_train,y_train)
    prediksi=knn.predict(x_test)
    akurasi=metrics.accuracy_score(y_test,prediksi)
    hasil.append(akurasi)

print(hasil)
plt.plot(hasil)
plt.xlabel('k')
plt.ylabel('Akurasi')
plt.xticks(np.arange(10),('1','2','3','4','5','6','7','8','9','10'))
plt.savefig('KNN.png')
plt.show()