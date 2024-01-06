import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, GRU, Conv1D, MaxPooling1D, Flatten, ZeroPadding1D, Dropout, Input, Concatenate
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

seed = 237
np.random.seed(seed)

trn_file = 'K-S-3Slice-Bigram-PSSM-VEGF.csv'

nb_classes = 2

dataset = np.loadtxt(trn_file, delimiter=",")
X = dataset[:, 1:401].reshape(len(dataset), 20, 20, 1)
Y = dataset[:, 0]

def create_gru_model():
    model = Sequential()
    model.add(GRU(128, activation='relu', input_shape=(20, 20)))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_1dcnn_model():
    model = Sequential()
    model.add(ZeroPadding1D(1, input_shape=(20, 20)))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(ZeroPadding1D(1))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(ZeroPadding1D(1))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy'])
    return model

def ensemble_model(gru_model, cnn_model):
    gru_input = Input(shape=(20, 20))
    cnn_input = Input(shape=(20, 20, 1))

    gru_output = gru_model(gru_input)
    cnn_output = cnn_model(cnn_input)

    merged = Concatenate()([gru_output, cnn_output])
    ensemble_output = Dense(nb_classes, activation='softmax')(merged)

    model = Model(inputs=[gru_input, cnn_input], outputs=ensemble_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

gru_model = create_gru_model()
cnn_model = create_1dcnn_model()

Acc = 0
Sen = 0
Sp = 0
Precision = 0
F1 = 0
MCC = 0
AUC = 0
meanAUC = 0
ExCM = np.array([
    [0, 0],
    [0, 0],
], dtype=int)

# Instantiate ensemble model
ensemble_model_instance = ensemble_model(gru_model, cnn_model)

for train, test in kfold.split(X, Y):
    X_train, X_test = X[train], X[test]
    Y_train, Y_test = Y[train], Y[test]

    gru_model.fit(X_train, to_categorical(Y_train, nb_classes), epochs=30, batch_size=100, verbose=0)
    cnn_model.fit(X_train, to_categorical(Y_train, nb_classes), epochs=30, batch_size=100, verbose=0)

    true_labels = np.asarray(Y_test)
    gru_predictions = np.argmax(gru_model.predict(X_test), axis=1)
    cnn_predictions = np.argmax(cnn_model.predict(X_test), axis=1)

    ensemble_input = [X_test, X_test]
    ensemble_predictions = np.argmax(ensemble_model_instance.predict(ensemble_input), axis=1)

    CM = confusion_matrix(y_pred=ensemble_predictions, y_true=true_labels)
    TN, FP, FN, TP = CM.ravel()

    print('Accuracy: {0:.2f}%\n'.format(((TP + TN) / (TP + FP + TN + FN)) * 100.0))
    Acc += round((TP + TN) / (TP + FP + TN + FN), 4)
    Sen += round((TP / (TP + FN)), 4)
    Sp += round(TN / (FP + TN), 4)
    Precision += round(TP / (TP + FP), 4)
    MCC += round(((TP * TN) - (FP * FN)) / (np.math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))), 4)
    F1 += round((2 * TP) / ((2 * TP) + FP + FN), 4)
    ExCM += CM

def PrintMean(nsplit, Acc_, Sen_, Sp_, Precision_, MCC_, F1_, AUC, ExCM):
    print('Accuracy: {0:.2f}%'.format((float(Acc_)/nsplit) * 100.0))
    print('Sensitivity: {0:.2f}%'.format((float(Sen_)/nsplit)* 100.0))
    print('Specificity: {0:.2f}%'.format((float(Sp_)/nsplit) * 100.0))
    print('Precision: {0:.2f}%'.format((float(Precision_)/nsplit)* 100.0))
    print('MCC: {0:.4f}'.format(float(MCC_)/nsplit))
    print('F1_Score: {0:.2f}%'.format((float(F1_) / nsplit) * 100.0))
    print('AUC: {0:.2f}%'.format((float(AUC) / nsplit) * 100.0))
    print('Confusion Matrix:\n '+str(ExCM))

PrintMean(5, Acc, Sen, Sp, Precision, MCC, F1, AUC, ExCM)

# serialize models to JSON
gru_model_json = gru_model.to_json()
cnn_model_json = cnn_model.to_json()
ensemble_model_json = ensemble_model_instance.to_json()  # Use ensemble_model_instance
with open("gru_model.json", "w") as json_file:
    json_file.write(gru_model_json)
with open("cnn_model.json", "w") as json_file:
    json_file.write(cnn_model_json)
with open("ensemble_model_GRU_1DCNN.json", "w") as json_file:
    json_file.write(ensemble_model_json)

# serialize weights to HDF5
gru_model.save_weights("gru_model.h5")
cnn_model.save_weights("cnn_model.h5")
ensemble_model_instance.save_weights("ensemble_model_GRU_1DCNN.h5")
print("Saved models to disk")
