---
layout: page
title: "EDA"
permalink: /EDA
---

Exploratory Data Analysis (EDA) is a critical step in the data analysis process where the main goal is to summarize the main characteristics of a dataset, often with the help of graphical representations. The primary purpose of EDA is to **understand the structure and nature of the data, identify patterns, relationships, anomalies, and derive insights** that can guide further analysis.

To do so, let's divide this part into 3 sections.

### 1. [Dataset Description](./dataset_description.md)

The aim of this section is to provide some description of the dataset that I will use for my project.

We will tackle the following questions:

- How were the data obtained? Experiments? Simulations? A mixture?

- What kinds of data are included? e.g., Scalar valued data? Time series? Spectral data? Image data? etc.

- What are all of the different features (potential factors, cofactors, and responses) of the dataset?

- What is the overall dimension of the data?

- Is the dataset dense, sparse, or are certain features dense while others are sparse?

### 2. [Dataset Sampling](./dataset_sampling.md)

The aim of this section is to provide information about features sampling in order to determine which response features are more suitable for the project.

We will tackle the following questions:

- Which of the features are well sampled (e.g., is it feasible that the samples of _response features_ represent random sampling of probability densities? How are the input features sampled? Is there sufficient variation to support statistical modeling?, etc.).

- Of the response features that are sufficiently sampled, what is the nature of the probability densities suggested by the samples? (continuous vs. discrete, is there an identifiable distribution like uniform vs. Gaussian vs. bimodal vs. lognormal vs. Weibull...? Or is the probability density in higher dimension? etc.)

- Are any of the data features cross-correlated (i.e., highly correlated)? Are any of them completely uncorrelated?

- If the factor-response relationships are defined, what are the relationships between the factors and each response? Are any of them obvious/well-defined by 2D analyses (pairplots, etc..)?

- Are higher dimensional relationships evident? (using a higher dimension visualization technique, in addition to low dimension techniques, to critically assess and probe these relationships).

### 3. [Next Steps](./next_steps.md)

This section aims to provide some information about the next steps of the project after the _Exploratory Data Analysis_.
