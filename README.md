# Minimizing Churn Rate through Financial Habbit Analysis

This project analyzes financial behavior of users and estimates whether someone will chrun (leave our product/app) or not.

One major drawback of our project was that this analysis was performed on the assumption that if someone leaves the product, they would leave it, of if someone stays with the product, they stay forever.

This flaw was discovered after performing Exploratory Data Analysis on our dataset when I discovered no mention of time frame as a feature.

### Process

In this project, we are primarily analyzing the probability of someone staying with us or leaving our product.

### Why I chose legacy machine learning algorithm instead of deep learning?

We use classical or legacy machine learning models when we have a small dataset. 

If we have a large enough dataset, deep learning almost always outperforms any other shallow learning method.

The amount of data needed to solve a problem using shallow or deep learning models depends on the problems being solved. If e.g. we are modeling a consumer finance problem, then we need a LOT of data and methods to neutralize or eliminate racial or ethnic biases by preventing our deep learning model to learn from them and take them into consideration. Our best bet there is to use shallow-deep models.

Since we are given a very small dataset relative to the size of the problem, we are going to use our trusted lagacy machine learning model, LogisticRegression. We are using LogisticRegresstion because it is a simple binary classification problem.

### Methodology

I created a dataset with only binary columns i.e. the cols in which the value could be either 0 or 1. Then I created their correlation plot to see the correlation of all features on one another. This also creates a redundent set of dependent variables, which we should mask from the final image and only show half of the triangle with their correlations.

I then dropped all the columns which could have been dependent on one another in such a way that if any variable could be the result of the product of one or more other variables in the correlation plot, then we have to remove that variable. This is where all the domain knowledge comes in and this is where we need experts' opinion.

After the basic modeling was complete, I performed One-Hot-Encoding for categorical variables and then split the dataset into 80-20 - train test dataset.

The model was then trained using .fit method, and then its performance parameters like accuracy, recall, f1-score, were calculated. The test dataset, y_pred was then created using .predict method.

All the steps are detailed in the code .py files above.

## Conclusion

Because we had a little amount of data available, we do not have enough leverage to tune our model. We performed k-fold cross validation, which was as much as we could with classical machine learning models. This project would have performed much better with deep neural networks instead of simple LogisticRegression. 

By using deep learning, we have a LOT more leverage and levers to tune the model and further improve the model performance. 

