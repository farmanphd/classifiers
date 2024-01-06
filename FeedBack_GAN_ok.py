import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, matthews_corrcoef
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, Adamax

# Load training dataset
iRec = 'DDE_Steroid_receptor_train_1001_1029.csv'
D = pd.read_csv(iRec, header=None)

# Split into features (X) and classes (Y)
X = D.iloc[:, :-1].values
Y = D.iloc[:, -1].values

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

    # Define the generator model
    generator_input = Input(shape=(X.shape[1],))
    generator_output = Dense(128)(generator_input)
    generator_output = LeakyReLU()(generator_output)
    #generator_output = Dropout(0.5)(generator_output)
    generator_output = Dense(X.shape[1], activation='sigmoid')(generator_output)
    generator = Model(inputs=generator_input, outputs=generator_output)

    # Define the discriminator model
    discriminator_input = Input(shape=(X.shape[1],))
    discriminator_output = Dense(128)(discriminator_input)
    discriminator_output = LeakyReLU()(discriminator_output)
    #discriminator_output = Dropout(0.5)(discriminator_output)
    discriminator_output = Dense(1, activation='sigmoid')(discriminator_output)
    discriminator = Model(inputs=discriminator_input, outputs=discriminator_output)

    # Define the combined model (GAN)
    gan_input = Input(shape=(X.shape[1],))
    generated_sample = generator(gan_input)
    gan_output = discriminator(generated_sample)
    gan = Model(inputs=gan_input, outputs=gan_output)

    # Compile the discriminator model
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Compile the GAN model
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

    # Train the GAN model
    gan.fit(X_train, Y_train, batch_size=100, epochs=50, verbose=0)

    # Evaluate the discriminator on test data
    Y_pred = discriminator.predict(X_test)
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

# Save model using JSON serialization
generator_json = generator.to_json()
with open("GAN_model.json", "w") as json_file:
    json_file.write(generator_json)
generator.save_weights("GAN_model.h5")
print("Saved model to disk")

# Print the mean evaluation metrics
print('Mean Confusion Matrix:')
print(mean_cm)
print('Mean Accuracy:', mean_accuracy)
print('Mean Sensitivity:', mean_sensitivity)
print('Mean Specificity:', mean_specificity)
print('Mean MCC:', mean_mcc)
print('Mean AUC:', mean_auc_score)
