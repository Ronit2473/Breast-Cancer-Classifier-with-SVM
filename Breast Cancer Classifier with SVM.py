#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sklearn
from sklearn import svm, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer()
print("Feature names:", cancer.feature_names)
print("Target names:", cancer.target_names)


x = cancer.data
y = cancer.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)


clf = svm.SVC(kernel='linear', C=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)


acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", acc)


# In[ ]:




