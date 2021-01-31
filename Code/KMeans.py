#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:34:34 2019

@author: Gruppo PP
"""

# roc curve and auc
import numpy as np, pandas as pd
from sklearn.cluster import KMeans  
import seaborn as sns; sns.set()  # for plot styling
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from inspect import signature

feature = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]
feature_dummied = ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "breast", "breast-quad", "irradiat"]
dataset = pd.read_csv("breast-cancer.csv", sep=",", names=feature, dtype={'class':object, 'age':object, 'menopause':object, 'tumor-size':object, 'inv-nodes':object, 'node-caps':object, 'deg-malig':np.int32, 'breast': object, 'breast-quad':object, 'irradiat':object})
data_dummies = pd.get_dummies(dataset, columns=feature_dummied)
X =  data_dummies.drop(["class"], axis=1)
y = pd.get_dummies(data_dummies['class'], columns=['class'])
y = y.drop(["recurrence-events"], axis=1)


kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 2, n_init = 9, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_

print("\nEtichette:")  
print(kmeans.labels_)

print ('\nClasification report:\n',classification_report(y, y_kmeans))
print ('\nConfussion matrix:\n',confusion_matrix(y, y_kmeans))

average_precision = average_precision_score(y, y_kmeans)
precision, recall, _ = precision_recall_curve(y, y_kmeans)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

accuracy = accuracy_score(y_kmeans, y)
