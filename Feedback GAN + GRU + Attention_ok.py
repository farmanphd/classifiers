import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, matthews_corrcoef
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Activation, Dot, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam

# Load training dataset
iRec = 'RMP-PSSM_DCT-Ingio_train.csv'
D = pd.read_csv(iRec, header=None)

# Split into features (X) and classes (Y)
X = D.iloc[:, :-1].values
Y = D.iloc[:, -1].values

# Reshape input data for GRU
X = X.reshape(len(X), X.shape[1], 1)

# Define evaluation metrics variables
confusion_matrices = []
accuracies = []
sensitivities = []
specificities = []
mcc_scores = []
auc_scores = []

# Perform 5-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(X, Y):
    # Split the data into train and test sets
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Define the Feedback GAN + GRU + Attention model architecture
    input_layer = Input(shape=(X.shape[1], X.shape[2]))

    # GRU layer
    gru = GRU(64, return_sequences=True)(input_layer)

    # Attention mechanism
    attention_weights = Dense(1, activation='tanh')(gru)
    attention_weights = Activation('softmax')(attention_weights)
    attended_representation = Dot(axes=1)([attention_weights, gru])

    # Flatten and Dense layers
    flatten = Flatten()(attended_representation)
    dense1 = Dense(128, activation='relu')(flatten)
    output = Dense(1, activation='sigmoid')(dense1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Train the model
    model.fit(X_train, Y_train, batch_size=1000, epochs=2, verbose=0)

    # Evaluate the model on test data
    Y_pred = model.predict(X_test)
    Y_pred_binary = (Y_pred > 0.5).astype(int).flatten()

    # Calculate evaluation metrics
    confusion = confusion_matrix(Y_test, Y_pred_binary)
    accuracy = accuracy_score(Y_test, Y_pred_binary)
    sensitivity = recall_score(Y_test, Y_pred_binary)
    specificity = recall_score(Y_test, Y_pred_binary, pos_label=0)
    mcc = matthews_corrcoef(Y_test, Y_pred_binary)
    auc_score = roc_auc_score(Y_test, Y_pred)

    # Append the results to the lists
    confusion_matrices.append(confusion)
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    mcc_scores.append(mcc)
    auc_scores.append(auc_score)

# Calculate mean evaluation metrics
mean_cm = np.mean(confusion_matrices, axis=0)
mean_accuracy = np.mean(accuracies)
mean_sensitivity = np.mean(sensitivities)
mean_specificity = np.mean(specificities)
mean_mcc = np.mean(mcc_scores)
mean_auc_score = np.mean(auc_scores)

# Print the mean evaluation metrics
print('Mean Confusion Matrix:')
print(mean_cm)
print('Mean Accuracy:', mean_accuracy)
print('Mean Sensitivity:', mean_sensitivity)
print('Mean Specificity:', mean_specificity)
print('Mean MCC:', mean_mcc)
print('Mean AUC:', mean_auc_score)
