from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale

"""
Digits dataset that is in Python located in sklearn package is used for applying SVM model. In this dataset each row is vectorized image of a 
number between 0 and 9. we use classification here.Digits classification is the act of labeling every image with its corresponding digit. 
The response variable; as a result, is a number between 0 and 9 given by y.In this code, SVM model with 'kernel':['linear','rbf','poly'] is used.
"""
#load digits dataset in Python
digits = load_digits(return_X_y =True)
X=digits[0]
y=digits[1]
X = scale(X)
print(len(digits), len(X),len(y))
print(len(y==1))
print(np.unique(y))
#find that the data is balance or not so we should find the proportion of each class in dataset
for i in range(1,10):
    print(i, len(y[y==i]))
#plot the picture
plt.imshow(X[7,:].reshape(8,8))
#split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#use cross validation by using GridSearchCV function.
Hyperparameters ={'C':np.arange(0.01,11,1),'gamma':np.arange(0.001,0.01,0.001),'degree':range(5),'kernel':['linear','rbf','poly']}
svm_clf = svm.SVC()
cv_grid= GridSearchCV(estimator= svm_clf ,param_grid = Hyperparameters, scoring ='accuracy',cv = 10)
print(cv_grid.best_params_)
print(cv_grid.best_score_)
#create a dataframe of parametrs and score
data=pd.DataFrame(cv_grid.cv_results_['params'],cv_grid.cv_results_['mean_test_score'])
print(data)

print("best C for SVM is",cv_grid.best_estimator_)
cv_grid.best_estimator_.fit(X_train, y_train)
print("accuracy is for test data is: ",cv_grid.score(X_test,y_test))
y_pred=cv_grid.predict(X_test)
print("y_pred is: ",y_pred)
print("confusion_matrix is:\n ",confusion_matrix(y_test,y_pred))