#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:16:27 2019

@author: Gruppo PP
"""
import numpy as np, pandas as pd, seaborn as sn
from matplotlib import pyplot
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

""""""
feature = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]
feature_dummied = ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "breast", "breast-quad", "irradiat"]
dataset = pd.read_csv("breast-cancer.csv", sep=",", names=feature, dtype={'class':object, 'age':object, 'menopause':object, 'tumor-size':object, 'inv-nodes':object, 'node-caps':object, 'deg-malig':np.int32, 'breast': object, 'breast-quad':object, 'irradiat':object})
data_dummies = pd.get_dummies(dataset, columns=feature_dummied)
data_dummies = data_dummies.drop(["class"], axis=1)

X = data_dummies
y = pd.get_dummies(dataset["class"], columns=["class"])
y = y["recurrence-events"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 13)

error = []
# Calculating error for K values between 1 and 40
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
 
    
#Grafico che mostra l'errore medio nelle predizioni a seguito di una variazione del valore K(numero vicini)
"""plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize = 10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')
plt.show()"""



neigh = KNeighborsClassifier(n_neighbors = 2)
knn = neigh.fit(X_train, y_train) 

prediction = knn.predict(X_test)
accuracy = accuracy_score(prediction, y_test)

print ('\nClasification report:\n',classification_report(y_test, prediction))
print ('\nConfusion matrix:\n',confusion_matrix(y_test, prediction))

confusion_matrix = confusion_matrix(y_test, prediction)
df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "01"], columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


#train model with cv of 5 
cv_scores = cross_val_score(neigh, X, y, cv=5)


print('\ncv_scores mean:{}'.format(np.mean(cv_scores)))
print('\ncv_score variance:{}'.format(np.var(cv_scores)))
print('\ncv_score dev standard:{}'.format(np.std(cv_scores)))

data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig,axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)

"""
probs = knn.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
pyplot.xlabel('FP RATE')
pyplot.ylabel('TP RATE')
# show the plot
pyplot.show()

average_precision = average_precision_score(y_test, prediction)
precision, recall, _ = precision_recall_curve(y_test, prediction)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
"""

""" BILANCIANDO IL DATASET """

sm = SMOTE(random_state=0)

X1, y1 = sm.fit_sample(X, y.ravel())

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size = 0.75, random_state = 13)

knn1 = neigh.fit(X1_train, y1_train) 

prediction_res = knn1.predict(X1_test)

accuracy_res = accuracy_score(prediction_res, y1_test)

#train model with cv of 5 
cv_scores1 = cross_val_score(neigh, X1, y1, cv=5)

print('\nFollowing scores after balancing:')
print('\ncv_scores  mean:{}'.format(np.mean(cv_scores1)))
print('\ncv_score variance:{}'.format(np.var(cv_scores1)))
print('\ncv_score dev standard:{}'.format(np.std(cv_scores1)))
print('\n')

"""
probs = knn.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
pyplot.xlabel('FP RATE')
pyplot.ylabel('TP RATE')
# show the plot
pyplot.show()



average_precision2 = average_precision_score(y1_test, prediction_res)
precision2, recall2, _ = precision_recall_curve(y1_test, prediction_res)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall2, precision2, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall2, precision2, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision2))
plt.show()"""


f1= f1_score(y_test, prediction)

data = {'variance': np.var(cv_scores1), 'standard dev': np.std(cv_scores1)}
names = list(data.keys())
values = list(data.values())
fig,axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)



