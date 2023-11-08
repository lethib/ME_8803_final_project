# Personal notes about Statistical Modeling and Machine Learning

Machine Learning -> Statistical Modeling for big data using Computer Science

## What's bias and variance?

Bias and variance are components of reducible error. Reducing errors requires selecting models that have appropriate complexity and flexibility, as well as suitable training data.

### Bias

Bias is the amount that a model’s prediction differs from the target value, compared to the training data. Bias error results from simplifying the assumptions used in a model so the target functions are easier to approximate.

Bias can be introduced by model selection. Data scientists conduct resampling to repeat the model building process and derive the average of prediction values. Resampling data is the process of extracting new samples from a data set in order to get more accurate results. There are a variety of ways to resample data including:

- K fold resampling, in which a given data set is split into a K number of sections, or folds, where each fold is used as a testing set.
- Bootstrapping, which involves iteratively resampling a dataset with replacement.

### Variance

Variance describes how much a random variable differs from its expected value. Variance is based on a single training set. Variance measures the inconsistency of different predictions using different training sets — **it’s not a measure of overall accuracy**.

Variance can lead to overfitting, in which small fluctuations in the training set are magnified. A model with high-level variance may reflect random noise in the training data set instead of the target function.

### Bias-Variance Tradeoff

When building a supervised machine-learning algorithm, the goal is to achieve low bias and variance for the most accurate predictions. We must do this while keeping underfitting and overfitting in mind. A model that exhibits small variance and high bias will underfit the target, while a model with high variance and little bias will overfit the target.

A model with high variance may represent the data set accurately but could lead to overfitting to noisy or otherwise unrepresentative training data. In comparison, a model with high bias may underfit the training data due to a simpler model that overlooks regularities in the data.

The trade-off challenge depends on the type of model under consideration. A linear machine-learning algorithm will exhibit high bias but low variance. On the other hand, a non-linear algorithm will exhibit low bias but high variance. Using a linear model with a data set that is non-linear will introduce bias into the model. The model will underfit the target functions compared to the training data set. The reverse is true as well — if you use a non-linear model on a linear dataset, the non-linear model will overfit the target function.

## Why response features should represent a random distribution of a probability density ?

In statistical modeling, the response variable is a crucial component of the model as it represents the outcome or the variable of interest that you want to study or predict. So, it should ideally follow a random distribution for several reasons:

- **Assumption of independance**: Many statistical models, such as linear regression and generalized linear models, assume that the observations are independent. If the response variable does not follow a random distribution, it may violate this assumption and lead to incorrect model results. When the response variable follows a probability distribution, it's more likely that the observations are independent.

- **Model Interpretability**: Probability distributions have well-defined properties and characteristics, making it easier to interpret the results of a model. For example, in linear regression, assuming that the response variable follows a normal distribution allows you to make probabilistic statements about the relationships between predictor variables and the response.

- **Model Performance**: Statistical models often make assumptions about the distribution of the errors or residuals in the model. When the response variable follows a known probability distribution, it becomes easier to assess the goodness of fit and the model's performance. You can use statistical tests to check if the model's assumptions about the distribution of the residuals are met.

---

# Sources

- [Bias and Variance in Machine Learning](https://www.mastersindatascience.org/learning/difference-between-bias-and-variance/)
