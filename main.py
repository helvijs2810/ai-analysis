#%% matplotlib ipympl
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.utils import Bunch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC


#super()._check_params_vs_input(X, default_n_init=10)

tempFeatures = []
tempTargets = []

feature_labels = ['age', 'gender' , 'impluse' , 'pressurehight', 'pressurelow',	'glucose', 'kcm', 'troponin']
traget_labels = ['results']

file = open('Heart Attack.csv', 'r')

#df = pd.read_csv(file)

#feature_labels = df.keys()

reader = csv.reader(file)

next(reader, None)

#print(feature_labels)

for line in reader:
    x = line[:-1]
    x = [float(i) for i in x]
    y = line[-1:][0]
    if y == "negative":
        y = int(0)
    else:
        y = int(1)
        
    tempFeatures.append(x)
    tempTargets.append(y)   

features = np.array(tempFeatures)
targets = np.array(tempTargets)

data = Bunch(data=features, target=targets, feature_names = feature_labels, target_names = traget_labels)

x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target)

scaler = MinMaxScaler()

features = scaler.fit_transform(features)
#print(targets)

selector = SelectKBest(chi2, k=3)
selector.fit(x, y)
features = selector.fit_transform(features, targets)

newX = selector.transform(x)
vector_names = list(x.columns[selector.get_support(indices=True)])
print(" ".join(vector_names))

trainFeatures, testFeatures, trainTargets, testTargets = train_test_split(features, targets, test_size=0.2)  

#KNN Classifier
knn = KNeighborsClassifier()
knn.fit(trainFeatures, trainTargets)

predictions = knn.predict(testFeatures)
accuracy = accuracy_score(testTargets, predictions)

print("KNN accuracy: " +str(accuracy))

knum = math.sqrt(len(testTargets))

knn16 = KNeighborsClassifier(n_neighbors=16)
knn16.fit(trainFeatures, trainTargets)

predictions16 = knn16.predict(testFeatures)
accuracy2 = accuracy_score(testTargets, predictions16)

print("KNN 16K accuracy: " +str(accuracy2))

#Logistical Regression
LR = LogisticRegression(random_state=0, solver='lbfgs', max_iter=200)
LR.fit(trainFeatures, trainTargets)

probabilities = LR.predict_proba(testFeatures)

lrpredictions = LR.predict(testFeatures)

accuracy2 = accuracy_score(testTargets, lrpredictions)
print("LR accuracy: " +str(accuracy2))

clf = SVC(random_state=0)
clf.fit(trainFeatures, trainTargets)
confpred = clf.predict(testFeatures)

cm = confusion_matrix(testTargets, confpred)

disp = ConfusionMatrixDisplay(cm).plot()
# %%