# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 07:52:46 2018

@author: zaheer
"""
import pickle

import pandas
from keras import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,auc,f1_score, \
    cohen_kappa_score,matthews_corrcoef

dataframe = pandas.read_csv("SmoothedPSSM_DWT_2104"".csv")

dataset = dataframe.values

X = dataset[:, 0:1040]
Y = dataset[:, 1040]

#X = dataframe.iloc [:,0:20]
#Y = dataframe.iloc [:,20]
x_train, x_test, y_train, y_test = train_test_split(X , Y , test_size=0.2, random_state=123)


dnsModel = Sequential()
dnsModel.add(Dense(80, input_dim=1040, kernel_initializer='uniform', activation='relu'))
dnsModel.add(Dense(80, kernel_initializer='uniform', activation='relu'))
dnsModel.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
dnsModel.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
# Fit the model
dnsModel.fit(x_train , y_train , epochs=30,   batch_size=20)
#dnsModel.evaluate(x_test,y_test,verbose=1)
score, acc = dnsModel.evaluate(x_test, y_test, batch_size=10)
print('Test score:', score)
print('Test accuracy:', acc)
model_json = dnsModel.to_json()
with open("model/model.json","w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
dnsModel.save_weights("model/model.h5")
print("Saved model to disk")
# later...
# load json and create model
json_file = open('model/model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model/model.h5")
print("Loaded model from disk")
result = model.predict(x_test)
print("Loaded Accuracy=", result)

probs = model.predict_proba(x_test)
#y_test= clf.predict(xpredin)
probs = probs[:, -1]
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
print("Area Under Curve: ", roc_auc)
# predict class values
yhat = model.predict(x_test)
#precision, recall = precision_recall_curve(yhat.round(), probs)
f1 = f1_score(y_test, probs.round())
kappa = cohen_kappa_score(y_test, yhat.round())
cm = confusion_matrix(y_test, yhat.round()).ravel()
tn, fp, fn, tp = confusion_matrix(y_test, yhat.round()).ravel()
print("Confusion Matrix: ", cm)
print("Accuracy", acc)
print("Sensitivity: ", tp / (tp + fn))
print("specificity: ", tn / (tn + fn))
print("F1 Measure : ", f1)
print("Kappa Statistics", kappa)
print("MCC:", matthews_corrcoef(y_test, yhat.round()))
print("Area Under Curve: ", roc_auc)
