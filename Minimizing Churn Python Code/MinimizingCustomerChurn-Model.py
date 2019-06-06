# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:54:41 2019

@author: Syed Jafri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

dataset = pd.read_csv('my_new_churn_data.csv')

# Data Preperation
user_identifier = dataset['user']
dataset = dataset.drop(columns=['users'])

# One-hot encoding - for categorical variables
dataset.housing.value_counts()
dataset = pd.get_dummies(dataset)
dataset.columns

dataset = dataset.drop(columns=['housing_na', 'zodiac_sign_na', 'payment_type_na'])

# Train_test_split step

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns=['churn']),
                                                    dataset['churn'],
                                                    test_size = 0.2,
                                                    random_state=0)


y_train.value_counts()

pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index
    
random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes, ]
y_train = y_train[new_indexes]

# Feature Scaling - Normalizing the dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test.index = X_test.index.values
X_train = X_train2
X_test = X_test2


# Building our model - Logistic Regression Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)

# Now, making predictions
y_pred = model.predict(X_test)

# Evaluating model results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred)
precision_score(y_test, y_pred)
# As a reminder, the precision score is -> what is the actual number of
# trues vs. the number of trues predicted.

# For all the definitions, and explanations, please see the other projects

recall_score(y_test, y_pred)

# Visualizing results of confusion matrix
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# K-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model,
                             X = X_train,
                             y = y_train,
                             cv = 10)

accuracies.mean()


# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train.columns, columns = ["features"]),
           pd.DataFrame(np.transpose(model.coef_), columns = ["coef"])
           ],axis = 1)

    
# The reason we are analyzing the coefficients here is to know which 
# coefficients carry most weights and will affect the model in what ways.
# Now, we will perform feature selection based on the factors which 
# affect the model the most, positively or negatively.
    
# Feature Selection 

# First of all, we import RFE feature from sklearn library. This method 
# will rank the coefficients based on their weight values. For RFE method
# to work properly, we need to have created an array of coefficients earlier
# which we did above.
    
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

X_train.shape
rfe = RFE(classifier, 20)

# In the line above, we are only considering the 20 coefficients which
# affect our model the most in descending order. i.e. top 20 weights.

rfe = rfe.fit(X_train, y_train)

print(rfe.support_)
X_train.columns[rfe.support_]

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)

# Predictint Test Set
y_pred = classifier.predict(X_test[X_test.columns[rfe.support_]])

# Results Evaluation - These are all the same results we are already 
# familiar with. So we just copy and paste from above.

cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)


df_cm = pd.DataFrame(cm, index = (1, 0), columns = (1, 0))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train[X_train.columns[rfe.support_]],
                             y = y_train, cv = 10)
print("Logistic Regression Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))



# Analyzing Coefficients
# From the model performance we got above, it is clear that even with
# only 20 parameters, we got the same result as we did with 40 params
# therefore in order to make our model compact and efficient, we should
# only use our model with 20 params. 
# One big problem with this model is that if we try to do iterations on 
# our model, we will end up making some coefficients bigger than they
# actually should be.

# One solutions is to train the model all over again.

pd.concat([pd.DataFrame(X_train[X_train.columns[rfe.support_]].columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)



# Final Results
    
final_results = pd.concat([y_test, user_identifier], axis=1).dropna()
final_results['predicted_churn'] = y_pred
final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop=True)






