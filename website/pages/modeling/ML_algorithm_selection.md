---
layout: page
title: "ML Algorithm Selection"
permalink: /modeling/ML_algorithm_selection
---

The aim of thie notebook is to provide information about the Machine Learning algorithms that are used in this project. A deeper understanding of the choice of the algorithms is provided in the [next steps notebook](../EDA/next_steps.ipynb).

## 1. Kind of ML algorithms

Still according to the [next steps notebook](../EDA/next_steps.ipynb), the kind of ML algorithms that are used in this project are linear regression algorithms. This has been motivated by the fact that all of our features are numerical (input and output) and that our features are well correlated with the output, which pushes me to use linear algorithms (see [data sampling](../EDA/dataset_sampling.ipynb) for more information).

To recall, the algorithms that are used are:

- Linear Regression (LR)
- Elastic Net
- Linear SVR

## 2. Algorithms parameters

### 2.1. Linear Regression

LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation. For a simple linear regression with one variable, the equation is typically represented as $$y = \beta_0 + \beta_1 x + \epsilon$$ where:

- $$y$$ is the output variable,
- $$x$$ is the input variable,
- $$\beta_0$$ is the intercept term,
- $$\beta_1$$ is the coefficient for the input variable $$x$$,
- $$\epsilon$$ represents the error term.

In this case, we have two parameters: $$\beta_0$$ and $$\beta_1$$. In the case of​ multiple variables in a multiple linear regression, each variable has its own coefficient, and the number of parameters increases accordingly.

So, for our case, we have 4 input variables, which means that we have 5 parameters to estimate: the intercept term and the coefficients for the 4 input variables.

### 2.2. Elastic Net

Elastic Net is a Linear Regression with combined L1 and L2 priors as regularizer.

The Elastic Net algorithm has 2 types of parameters:

1. **Coefficients for the input variables**:

   As Elastic Net performs a Linear Regression (with penalization), it has the same coefficient parameters as the Linear Regression algorithm.

2. **Regularization parameters**:

   Elastic Net has 2 regularization hyperparameters:

   - $$\alpha$$: the constant that multiplies the penalty terms. $$\alpha = 0$$ is equivalent to an ordinary least square regression. For $$\alpha > 0$$, the penalty terms are added to the loss function.

   - $$l1_{ratio}$$: the Elastic Net mixing parameter. With $$0 <= l1_{ratio} <= 1$$. For $$l1_{ratio} = 0$$ the penalty is an L2 penalty. For $$l1_{ratio} = 1$$ it is an L1 penalty. For $$0 < l1_{ratio} < 1$$, the penalty is a combination of L1 and L2.

So, the total number of parameters in Elastic Net includes both the coefficients for each independent variable and the hyperparameters. In our case, we will have 5 parameters to estimate and 2 hyperparameters to tune.

### 2.3. Linear SVR

Linear Support Vector Regression is a model based on Support Vector Machine (SVM) that supports regression. The term _Linear_ means that the Kernel used by the algorithm is a linear one. It is different from traditional linear regression methods as it finds a hyperplane that best fits the data points in a continuous space, instead of fitting a line to the data points.

Linear SVR equation is $$y = c.W^T X + b$$ where:

- $$y$$ is the output variable,
- $$X$$ is the vector of input variables,
- $$W$$ is the vector of coefficients for each variable,
- $$c$$ is the constant that multiplies the dot product of $$W$$ and $$X$$,
- $$b$$ is the intercept term.

So for our case, as we have 4 input variables, we will have 5 parameters to estimate: the intercept term and the coefficients for the 4 input variables.

Linear SVR has many hyperparameters:

- $$c$$: Controls the trade-off between achieving a low training error and a low testing error. A smaller value of C leads to a wider margin but more classification errors on the training set.
- The loss function: It is the loss function to be used. It can be one of "epsilon_insensitive" or "squared_epsilon_insensitive".
  - $$\epsilon$$: Epsilon in the epsilon-insensitive loss functions. For epsilon=0, this is equivalent to an ordinary Support Vector Regression.
- $$tol$$: Tolerance for stopping criterion.

So, the total number of parameters in Linear SVR includes both the coefficients for each independent variable and the hyperparameters. In our case, we will have 5 parameters to estimate and 3 or 4 hyperparameters to tune.

## 3. Best Algorithm at first sight

All of the models do not have a huge number of parameters. As our feature are highly correlated (meaning that a strong linear relationship exits between the input and the output), I think that the Linear Regression algorithm will be the best one as it is the simplest one. However, Elastic Net and Linear SVR can be good alternatives as they can be more robust to outliers (linear regression can easily lead to overfitting). They also are not that complicated to understand and tune. It would be interesting to compare the results of the 3 algorithms.
