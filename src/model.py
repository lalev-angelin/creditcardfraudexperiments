##############################################################################
# This is the basic template we used to generate most of the experiments 
# Important constants: 
# model_name - Descriptive name of the architecture being tested. The software
# 	will create a directory with the same name and will place in it 
#       saved versions of the trained model every 100 or so eochs. The exact
#	interval is in model_train_epochs_steps. 
#       In addition, after the training completes, the program will write 
#       a file, named model.score, which will contain performance metrics of 
#       the model, computed on the test set. 
# model_train_epochs_steps - Number of epochs, after which a saved copy of 
#       the model is written on disk and a measurement of model performance
#       is taken.
# model_train_epochs_stop - Number of epochs after the training will stop and 
# 	the program will write model.score file with summary of the testing 
#	results.
# 

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

model_name = 'L1_DENSE_RELU_16_L2_DENSE_RELU_48_L3_DENSE_1_SIGMOID'
model_train_epochs_step=100
model_train_epochs_stop=1000

try:
    os.mkdir(model_name)
except:
    print("Error creating directory!")

trainData = pd.read_csv("train.csv")
# Do we need to shuffle our train set???
trainData = trainData.sample(frac=1)
trainX = trainData.drop(columns=["Class"])
trainY = trainData.Class

testData = pd.read_csv("test.csv")
testX = testData.drop(columns=["Class"])
testY = testData.Class

model = Sequential()
model.add(Dense(16, input_dim=30, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

combined = pd.DataFrame({"iterations":[], "auroc":[], "auprc": []})

for model_train_epochs in range(model_train_epochs_step, model_train_epochs_stop, model_train_epochs_step):
    model.fit(trainX.values, trainY.values, epochs=model_train_epochs_step, batch_size=10)
    model.save("./"+model_name+"/model_e"+str(model_train_epochs)+".h5")

    predictY = model.predict(testX)
    fpr, tpr, thresholds_roc = roc_curve(testY, predictY)
    roc = pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':thresholds_roc})
    roc.to_csv("./"+model_name+"/model_e"+str(model_train_epochs)+"_roc.csv")
    auc_roc_score=roc_auc_score(testY, predictY)

    precision, recall, thresholds_prc = precision_recall_curve(testY, predictY)
    prc = pd.DataFrame({'precision':precision,'recall':recall,'threshold':np.append(thresholds_prc, np.NaN)})
    prc.to_csv("./"+model_name+"/model_e"+str(model_train_epochs)+"_prc.csv")
    auc_prc_score= auc(recall, precision)

    # Pass the row elements as key value pairs to append() function 
    combined = combined.append({"iterations":model_train_epochs, "auroc":auc_roc_score, "auprc": auc_prc_score}, ignore_index=True)

combined.to_csv("./"+model_name+"/model_score.csv")
