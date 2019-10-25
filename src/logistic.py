import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from sklearn.linear_model import LogisticRegression

model_name = 'LOGISTIC_REGRESSION'

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

combined = pd.DataFrame({"iterations":[], "auroc":[], "auprc": []})

model = LogisticRegression().fit(testX, testY)
model.fit(trainX, trainY)

predictY = model.predict_proba(testX)[:,1]
fpr, tpr, thresholds_roc = roc_curve(testY, predictY)
roc = pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':thresholds_roc})
roc.to_csv("./"+model_name+"/model_roc.csv")
auc_roc_score=roc_auc_score(testY, predictY)

precision, recall, thresholds_prc = precision_recall_curve(testY, predictY)
prc = pd.DataFrame({'precision':precision,'recall':recall,'threshold':np.append(thresholds_prc, np.NaN)})
prc.to_csv("./"+model_name+"/model_prc.csv")
auc_prc_score= auc(recall, precision)

# Pass the row elements as key value pairs to append() function 
combined = combined.append({"iterations":0, "auroc":auc_roc_score, "auprc": auc_prc_score}, ignore_index=True)

combined.to_csv("./"+model_name+"/model_score.csv")
