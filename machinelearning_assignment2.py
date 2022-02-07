
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing or loading the dataset
data = pd.read_csv('auto-mpg.csv')
 
feature_cols = ['mpg','displacement', 'horsepower', 'weight','acceleration','model year']
X = data[feature_cols] # Features
y = data.cylinders # Target variable
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# performing preprocessing part
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA function on training
# and testing set of X component
from sklearn.decomposition import PCA
 
pca = PCA(n_components=2)
 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
 
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression To the training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print( accuracy_score(y_test, y_pred))

# # Fitting Logistic Regression To the training set
# from sklearn.linear_model import LogisticRegression 
 
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, y_train)

# # Predicting the test set result using
# # predict function under LogisticRegression
# y_pred = classifier.predict(X_test)

# # making confusion matrix between
# #  test set of Y and predicted value.
# from sklearn.metrics import confusion_matrix
 
# cm = confusion_matrix(y_test, y_pred)


# # Predicting the training set
# # result through scatter plot
# from matplotlib.colors import ListedColormap
 
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
#                      stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1,
#                      stop = X_set[:, 1].max() + 1, step = 0.01))
 
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
#              X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
#              cmap = ListedColormap(('yellow', 'white', 'aquamarine')))
 
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
 
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
 
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('PC1') # for Xlabel
# plt.ylabel('PC2') # for Ylabel
# plt.legend() # to show legend
 
# # show scatter plot
# plt.show()
