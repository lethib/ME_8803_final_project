---
layout: page
title: "Feature Engineering"
permalink: /feature_engineering
---

```python
# Standard imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## 1. Supervised or Unsupervised Learning?

As discussed in the [next_steps notebook](../EDA/next_steps.ipynb), the goal of this project is to predict the **Primary Energy consumption per Capita** using regression algorithms. Regressions algorithm are supervised Machine Learning algorithms.

Inputs and outputs features have already been determined in [this notebook](../EDA/next_steps.ipynb):

| Features                                                         | Factor | Cofactor | Response |
| ---------------------------------------------------------------- | :----: | :------: | :------: |
| Access to electricity (% of population)                          |   X    |    X     |          |
| Access to clean fuels for cooking                                |   X    |    X     |          |
| Renewable energy share in the total final energy consumption (%) |   X    |    X     |          |
| Primary energy consumption per capita (kWh/person)               |        |          |    X     |
| GDP per capita                                                   |   X    |    X     |          |

## 2. Features Details

Based on the Exploratory Data Analysis, I have saved 3 different datasets to train my regression models:

1. [`pe_dataset.csv`](../data/pe/pe_dataset.csv): This dataset contains all the rows of the main dataset but has only targetted features (inputs and outputs + country name and year).
2. [`normalized_pe_dataset.csv`](../data/pe/normalized_pe_dataset.csv): Same as `pe_dataset.csv` but with normalized numerical features.
3. [`normalized_pe_dataset_without_outliers.csv`](../data/pe/normalized_pe_dataset_without_outliers.csv): Same as `normalized_pe_dataset.csv` but without top outliers.

It will be interesting to watch how the models perform on each dataset. Data features were not collected from experiments or simulations but based on administratives sources, measurements and surveys. More details about that in the [dataset description notebook](../EDA/dataset_description.ipynb).

No Machine Learning models, no data augmentation and PCA have been used to derive the features. Daata features are the raw values from the main dataset that has only been normalized for the second and third datasets.

## 3. Features types

As discussed in the [`datset_sampling` notebook](../EDA/dataset_sampling.ipynb), all of the features (inputs and outputs) are modeled as continuous distributions (I don't take into account the country name and the year which are not relevant). Moreover, all of the features are numerical.

## 4. Data Cleaning

```python
df_pe = pd.read_csv("../data/pe/pe_dataset.csv")
df_pe.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3649 entries, 0 to 3648
    Data columns (total 7 columns):
     #   Column                                                            Non-Null Count  Dtype
    ---  ------                                                            --------------  -----
     0   Entity                                                            3649 non-null   object
     1   Year                                                              3649 non-null   int64
     2   Access to electricity (% of population)                           3639 non-null   float64
     3   Access to clean fuels for cooking                                 3480 non-null   float64
     4   Renewable energy share in the total final energy consumption (%)  3455 non-null   float64
     5   Primary energy consumption per capita (kWh/person)                3649 non-null   float64
     6   gdp_per_capita                                                    3367 non-null   float64
    dtypes: float64(5), int64(1), object(1)
    memory usage: 199.7+ KB

As the dataset is devided by countries, doing a hot deck imputation directly on all the dataset would be a mistake. For instance, data for _France_ and for _Yemen_ are not comparable. In one hand we have a developed country and in the other hand we have a developing country. Therefore, it is better to do a hot deck imputation for each country separately.

Due to the fact that we don't have a huge amount of missing values that will be imputed, I have decided to use an interpolation startegy for each country (it might not change a lot the distribution).

```python
countries = df_pe['Entity'].unique()
```

```python
# Interpolate missing values for each country
for country in countries:
    df_pe[df_pe['Entity'] == country] = df_pe[df_pe['Entity'] == country].interpolate(method='linear', limit_direction='forward', axis=0)
```

For other missing values, we will drop the rows that contain them. After this step, we still have more than 3,000 rows which is enough to train our models.

```python
df_pe = df_pe.dropna(axis=0)
df_pe.to_csv("../data/pe/cleaned/pe_dataset.csv", index=False)
df_pe.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3193 entries, 2 to 3648
    Data columns (total 7 columns):
     #   Column                                                            Non-Null Count  Dtype
    ---  ------                                                            --------------  -----
     0   Entity                                                            3193 non-null   object
     1   Year                                                              3193 non-null   int64
     2   Access to electricity (% of population)                           3193 non-null   float64
     3   Access to clean fuels for cooking                                 3193 non-null   float64
     4   Renewable energy share in the total final energy consumption (%)  3193 non-null   float64
     5   Primary energy consumption per capita (kWh/person)                3193 non-null   float64
     6   gdp_per_capita                                                    3193 non-null   float64
    dtypes: float64(5), int64(1), object(1)
    memory usage: 199.6+ KB

We will do the same cleaning process for the two remaining datasets !

```python
def clean_data(df, countries):
    for country in countries:
        df[df['Entity'] == country] = df[df['Entity'] == country].interpolate(method='linear', limit_direction='forward', axis=0)
    return df.dropna(axis=0)
```

```python
normalized_df_pe = pd.read_csv("../data/pe/normalized_pe_dataset.csv")
normalized_df_pe = clean_data(normalized_df_pe, countries)
normalized_df_pe.to_csv("../data/pe/cleaned/normalized_pe_dataset.csv", index=False)
normalized_df_pe.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3193 entries, 2 to 3648
    Data columns (total 7 columns):
     #   Column                                                            Non-Null Count  Dtype
    ---  ------                                                            --------------  -----
     0   Entity                                                            3193 non-null   object
     1   Year                                                              3193 non-null   int64
     2   Access to electricity (% of population)                           3193 non-null   float64
     3   Access to clean fuels for cooking                                 3193 non-null   float64
     4   Renewable energy share in the total final energy consumption (%)  3193 non-null   float64
     5   Primary energy consumption per capita (kWh/person)                3193 non-null   float64
     6   gdp_per_capita                                                    3193 non-null   float64
    dtypes: float64(5), int64(1), object(1)
    memory usage: 199.6+ KB

```python
normalized_df_pe_without_outliers = pd.read_csv("../data/pe/normalized_pe_dataset_without_outliers.csv")
normalized_df_pe_without_outliers = clean_data(normalized_df_pe_without_outliers, countries)
normalized_df_pe_without_outliers.to_csv("../data/pe/cleaned/normalized_pe_dataset_without_outliers.csv", index=False)
normalized_df_pe_without_outliers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3009 entries, 2 to 3465
    Data columns (total 7 columns):
     #   Column                                                            Non-Null Count  Dtype
    ---  ------                                                            --------------  -----
     0   Entity                                                            3009 non-null   object
     1   Year                                                              3009 non-null   int64
     2   Access to electricity (% of population)                           3009 non-null   float64
     3   Access to clean fuels for cooking                                 3009 non-null   float64
     4   Renewable energy share in the total final energy consumption (%)  3009 non-null   float64
     5   Primary energy consumption per capita (kWh/person)                3009 non-null   float64
     6   gdp_per_capita                                                    3009 non-null   float64
    dtypes: float64(5), int64(1), object(1)
    memory usage: 188.1+ KB

Saving those DataFrames into CSV files will allow us to use them in the next notebooks to train our models.
