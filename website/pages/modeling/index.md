---
layout: page
title: "Modeling"
permalink: modeling/
---

The modeling part of a machine learning project is a crucial phase where the actual machine learning algorithms are selected, trained, and evaluated.

Let's divide this part into 2 sections.

### 1. [ML Algorithms Selection](./ML_algorithm_selection.md)

The aim of this section is to provide information about the Machine Learning algorithms that are used in this project. A deeper understanding of the choice of the algorithms is provided in the [next steps section](../EDA/next_steps.md).

We will tackle the following questions:

- What kinds of ML algorithms are correct for the desired process model(s)? Why?

- Of the desired kinds, and considering the available samples (i.e., number of data entries), how many parameters/degrees of freedom are feasible? (There is not 1 fixed number, but rather, this is an order of magnitude).

- What are the best algorithms to evaluate, or is it clear that one algorithm rises above the rest at this point, without yet having trained or tested algorithms? Usually in statistical modeling, the "Best" is almost always the simplest model (i.e., fewest number of parameters) that sufficiently captures the statistics at play in the process model.

### 2. [Training and Testing](./training_testing.md)

The aim of this section is to train, test and evaluate the selected ML algorithms.

We will tackle the following questions:

#### Training

- How to divide the data into training vs. testing batches?

- How to address hyperparameter tuning? (e.g., cross-validation, etc.)

#### Testing

- Were bias, variance, and sensitivity to model parameterization (e.g., regularization, boot-strapping, etc. - for some simpler models, this analysis may not be necessary) all considered when evaluating the predictions? How?

- If residual - based methodologies like cross validation were used, were the residuals well - behaved? If not, how are they ill - behaved, why, and how could they be improved?
