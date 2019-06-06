# -*- coding: utf-8 -*-
"""
Created on Thu May 30 00:25:56 2019

@author: Syed Jafri
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('Minimizing Churn Data\churn_data.csv')

dataset.head()

dataset.describe()

# Before we begin to further start to process our dataset,
# we should remove NaN values i.e. Not a Number values.
# This should be standard practice before we begin any
# processing on our dataset.
# However, we should be careful in using NaN. If we have 
# a lot of NaN values, we can jeopardize the legitimacy 
# of our model by removing all NaN values as it will shrink 
# the amount of data available for processing.

dataset.isna().any()

# .any() returns Bool if ANY value in the column is NaN.

dataset.isna().sum()

dataset = dataset[pd.notnull(dataset['age'])]

dataset.isna().sum()

dataset = dataset.drop(columns=['credit_score', 'rewards_earned'])

dataset.columns


# Dataset Visualization - Building Histograms

dataset2 = dataset.drop(columns=['user', 'churn'])

fig = plt.figure(figsize=(15,12))
plt.suptitle('Histogram of Numerical Columns', fontsize=20)

for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i-1])
    
    #vals = np.size(dataset2.iloc[:, i-1].unique())
    
    #plt.hist(dataset2.iloc[:, i-1], bins=vals, )
    sns.countplot(dataset2.iloc[:,i-1], color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# Building Pie Plots (Pie Charts)

# Pie Charts are important to understand the distribution of
# some of the variable which we didn't understand from the previous 
# bar plots.

# We are plotting pie charts of ONLY BINARY columns i.e. the 
# columns whose output is binary, because that's how we are going
# to get the best visual distribution.

# We are creating a new dataset to get the binary columns.
# These are total 17 columns.

dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Dustribution', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i-1])
    values = dataset2.iloc[:, i - 1].value_counts(normalize=True).values
    index = dataset2.iloc[:, i - 1].value_counts(normalize=True).index
    
    plt.pie(values, labels = index, autopct = '%1.1f%%')
    plt.axis('equal')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 


# Here, we are counting the total churn numbers for the customers 
# who are waiting for a loan to process.

# BONUS TIP: One quick way to read Pandas, SQL or any querry and 
# find out what is going on there, is to read it backwards. i.e.
# start reading from the end of the querry.
# In out example, we have the following order:

# 1) counting the values,
# 2) of churn,
# 3) where,
# 4) from dataset2, the value of waiting_4_loan is equal to 1,
# 5) from dataset

# The output should be of how "many" of each "values" are in the 
# churn column, i.e. how many 0s and how many 1s in our case, as this
# is a binary column.

dataset[dataset2.waiting_4_loan == 1].churn.value_counts()

# Now, we will do simalar processing for other variables which had 
# a very high domination of one factor in the pie chart plotted above.
# In the current dataset, 5 pie chart was dominated by one field overwhelmengly
# therefore we are analyzing those columns below.

dataset[dataset2.cancelled_loan == 1].churn.value_counts()

dataset[dataset2.received_loan == 1].churn.value_counts()

dataset[dataset2.rejected_loan == 1].churn.value_counts()

dataset[dataset2.left_for_one_month == 1].churn.value_counts()



# Correlation Plot 

# Correlation plots quickly tell us which variables are most important
# for the model and helps us in peforming Feature Engineering on the 
# model. Please note that domain knowledge is also very important in 
# performing a reasonable feature engineering.

# To create a correlation plot, we need to eliminate the categorical variables
# and keep all the numerical variables 

dataset.drop(columns=['churn',
                      'user',
                      'housing',
                      'payment_type',
                      'zodiac_sign']).corrwith(dataset.churn).plot.bar(
                        figsize=(20,10),
                        title = 'Correlation with Response Variable',
                        fontsize=12,
                        rot=45,
                        grid=True)


# Correlation MATRIX

# Correlation matrix tells us about the correlation between all of the 
# variables. This is an important step in feature engineering.

sns.set(style='white')

# Computing the correlation matrix
corr = dataset.drop(columns=['user','churn']).corr()

# Generating a mask for the upper right triangle

# The main reason we are masking the upper right hand triangle is because
# when correlation matrix is plotted, it will create two instances of the 
# same product, i.e. corr(age,deposits) and corr(deposits, age)
# As both of the above variables give the same information, we don't need
# to add more confusion the plotted figure. Therefore we are masking half
# of the square and masking it with zeros.

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Drawing the heatmap with the mask
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Removing Correlated Fields
# From the correlation chart, we can see that app_web_user is a correlated
# field with app_downloaded and web_user. Therefore we should remove
# this column from our analysis dataset.
# The main reason why we are doing this is that we dont want those fields/cols
# which are already depending on one another or are formed as a product
# of other columns.

dataset = dataset.drop(columns = ['app_web_user'])

## Note: Although there are somewhat correlated fields, they are not colinear
## These feature are not functions of each other, so they won't break the model
## But these feature won't help much either. Feature Selection should remove them.

dataset.to_csv('my_new_churn_data.csv', index = False)




















