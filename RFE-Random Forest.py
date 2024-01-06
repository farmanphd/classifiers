from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
def get_enhanced_confusion_matrix(actuals, predictions, labels):
    """"enhances confusion_matrix by adding sensivity and specificity metrics"""
    cm = confusion_matrix(actuals, predictions, labels = labels)
    sensitivity = float(cm[1][1]) / float(cm[1][0]+cm[1][1])
    specificity = float(cm[0][0]) / float(cm[0][0]+cm[0][1])
    weightedAccuracy = (sensitivity * 0.9) + (specificity * 0.1)
    return cm, sensitivity, specificity, weightedAccuracy

iRec = 'mRmR_Smoth_SFPSSM_EDP_DWT_2104_400.csv'
D = pd.read_csv(iRec)#header=None)  # Using pandas

X = D.iloc[:, :-1].values
y = D.iloc[:, -1].values

response, _  = pandas.factorize(y)

xTrain, xTest, yTrain, yTest =train_test_split(X, response, test_size = .25, random_state = 36583)
print ("building the first forest")
rf = RandomForestClassifier(n_estimators = 500, min_samples_split = 2, n_jobs = -1, verbose = 1)
rf.fit(xTrain, yTrain)
importances = pandas.DataFrame({'name':X.columns, 'imp':rf.feature_importances_
                                }).sort(['imp'], ascending = False).reset_index(drop = True)

cm, sensitivity, specificity, weightedAccuracy = get_enhanced_confusion_matrix(yTest, rf.predict(xTest), [0,1])
numFeatures = len(X.columns)

rfeMatrix = pandas.DataFrame({'numFeatures':[numFeatures],
                              'weightedAccuracy':[weightedAccuracy],
                              'sensitivity':[sensitivity],
                              'specificity':[specificity]})

print("running RFE on  %d features"%numFeatures)

for i in range(1,numFeatures,1):
    varsUsed = importances['name'][0:i]
    print ("now using %d of %s features"%(len(varsUsed), numFeatures))
    xTrain, xTest, yTrain, yTest = cross_val_score(X[varsUsed], response, test_size = .25)
    rf = RandomForestClassifier(n_estimators = 500, min_samples_split = 2,
                                n_jobs = -1, verbose = 1)
    rf.fit(xTrain, yTrain)
    cm, sensitivity, specificity, weightedAccuracy = get_enhanced_confusion_matrix(yTest, rf.predict(xTest), [0,1])
    print("\n"+str(cm))
    print('the sensitivity is %d percent'%(sensitivity * 100))
    print('the specificity is %d percent'%(specificity * 100))
    print('the weighted accuracy is %d percent'%(weightedAccuracy * 100))
    rfeMatrix = rfeMatrix.append(
                                pandas.DataFrame({'numFeatures':[len(varsUsed)],
                                'weightedAccuracy':[weightedAccuracy],
                                'sensitivity':[sensitivity],
                                'specificity':[specificity]}), ignore_index = True)
print("\n"+str(rfeMatrix))
maxAccuracy = rfeMatrix.weightedAccuracy.max()
maxAccuracyFeatures = min(rfeMatrix.numFeatures[rfeMatrix.weightedAccuracy == maxAccuracy])
featuresUsed = importances['name'][0:maxAccuracyFeatures].tolist()

print ("the final features used are %s"%featuresUsed)