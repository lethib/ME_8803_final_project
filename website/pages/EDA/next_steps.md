---
layout: page
title: "Next Steps"
date: 2023-12-13 14:45:35 -0500
---

This notebook aims to answer the question 4 of the Homework.

## 1. Technical/Statistical purpose of each process model

Before training any model, a step of data cleaning is necessary to deal with missing values. There are several ways to deal with that:

- If a country contains lot of missing values for a given feature, we drop it from the dataset.
- If a country contains few missing values (data missing for 3 or 4 years), we impute the missing values with a hot-deck strategy (preserves the distribution) or linear interpolation (to have more accurate data).

Both problems (predicting CO2 emissions and the primary energy consumption per capita) are regression problems. So we naturally use regression models to answer those problems.

### Linear Regression

One of the most-used regression algorithms in Machine Learning. According to our analysis, we have continuous features that are well correlated with the response feature which means that a linear model could be a good choice and the Linear Regression is the simplest one.

The aim of this model is to find the best linear relationship between the input and output variables.

Then based on this cheatsheet, we can identify other models that could be interesting to use:

![cheatsheet](https://scikit-learn.org/stable/_static/ml_map.png)

According to our analysis, the path is the following one: _>50 samples_ **->** _Predicting a quantity_ **->** _<100k samples_ **->** _Few features should be important_

As the term few features is quite subjective, we will try one of each model that is proposed in the cheatsheet and compare the results.

### Elastic Net

Elastic Net is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods. These methods adresses the issue of overfitting in Linear Regression. Penalties are helpful to control the complexity of the model. It is useful when there are multiple features which are correlated with one another.

As we need to find a correct parameter (the penalty), we will use cross-validation to find the best one.

### Linear SVR

Support Vector Regression is a regression algorithm that uses the same principle as Support Vector Machine. It is a powerful model that can be used for linear or non-linear regression. As we seems to have a linear relationship between the input and output variables, we will use the linear version of the model (a linear kernel). It also add a penalty parameter to control the complexity of the model and may reduce the risk of overfitting introduced by the Linear Regression.
