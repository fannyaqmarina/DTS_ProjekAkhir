# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:44:02 2019

@author: ASPIRE E 14
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\Users\ASPIRE E 14\Music\DTS FGA 2019 - Unmul\projek akhir\ProjekAkhir\diabetes.csv")
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


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)
LR

yhat = LR.predict(x_test)
yhat

yhat_prob = LR.predict_proba(x_test)
yhat_prob

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Logistik.png')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))

#from sklearn.metrics import log_loss
#log_loss(y_test, yhat_prob)