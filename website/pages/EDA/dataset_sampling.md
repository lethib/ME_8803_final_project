---
layout: page
title: "Dataset sampling"
permalink: /EDA/dataset_sampling
---

The aim of this notebook is to provide information about features sampling in order to determine which response features are more suitable for the project.

```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

custom_style = {"grid.color": "black", "grid.linestyle": ":", "grid.linewidth": 0.3, "axes.edgecolor": "black", "ytick.left": True, "xtick.bottom": True}
sns.set_context("notebook")
sns.set_theme(style="whitegrid", rc=custom_style)
```

```python
# Read the dataset
df = pd.read_csv("../data/global-data-on-sustainable-energy.csv")
```

## Statistical Relationship

### 1. Feature Correlation

Correlation coefficients are indicators of the strength of the linear relationship between two different variables. A bigger circle means a higher correlation. The color of the circle indicates the sign of the correlation. A negative correlation (indicated by a blue color) means that the two variables move in opposite directions (when a variable is increasing, the other is decreasing). A positive correlation (indicated by a red color) means that the two variables move in the same direction (when a variable is increasing the other is also increasing).

```python
plt.figure(figsize=(20,20))

corr_matrix = df.iloc[:, 2:21].corr()

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation matrix of the dataset", fontsize=20, fontweight="bold")
plt.savefig("../img/correlation_matrix.png", dpi=200, bbox_inches="tight")
```

<p align="center">
  <img src="../../assets/img/dataset_sampling_5_0.png" />
</p>

We can clearly see that some features are highly corrolated to each other but other are not. For instance the `Financial flow to developing countries (US$)` has only poor correlation with all other features. It would be hard to predict this feature based on the other ones. This is the same for the `gdp_growth`, `Longitude`, `Energy intensity level of primary energy` features. It could have been interesting to work with these features but we will not use them for the project.

An interesting thing to note is that the `Latitude` feature is quite positively well corrolated with the `Access to electricity` and `Access to clean fuels for cooking` features. This means that the more you go to the north, the more you have access to electricity and clean fuels for cooking. This is not surprising since the north is more developed than the south (look also for the `gdp_per_capita` correlation score). This is also the case for the `Primary energy consumption par capita` feature. Northern countries consume more energy than southern ones. They tend to have a higher impact on climate change than southern countries.

Now, we can set an objective for the project. We would like to predict:

- The CO2 emissions
- Primary energy consumption per capita

These are mainly regression problems.

Predicting the percentage of Renewables energy as primary energy could have been also interesting to do but the response feature is so poorly filled that we can't use it as a response feature (cf the [dataset description notebook](./dataset_description.ipynb)).

### 2. Feature selection

#### Selection for the CO2 emissions prediction

Looking at the heatmap, we can clearly see that a linear relationship exists between some factors and the response (`Value_co2_emissions_kt_by_country`). Taking a threshold of 0.5 for the correlation score will make us takinng important feature to compute a regression after that.

```python
co2_cols_to_keep = [column for column in corr_matrix.columns if abs(corr_matrix.loc['Value_co2_emissions_kt_by_country', column]) > 0.5]
co2_cols_to_keep
```

    ['Electricity from fossil fuels (TWh)',
     'Electricity from nuclear (TWh)',
     'Electricity from renewables (TWh)',
     'Value_co2_emissions_kt_by_country',
     'Land Area(Km2)']

```python
df_co2 = df[['Entity', 'Year'] + co2_cols_to_keep]
df_co2.to_csv("../data/co2/co2_dataset.csv", index=False)
df_co2
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe smallest">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entity</th>
      <th>Year</th>
      <th>Electricity from fossil fuels (TWh)</th>
      <th>Electricity from nuclear (TWh)</th>
      <th>Electricity from renewables (TWh)</th>
      <th>Value_co2_emissions_kt_by_country</th>
      <th>Land Area(Km2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>0.16</td>
      <td>0.0</td>
      <td>0.31</td>
      <td>760.000000</td>
      <td>652230.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2001</td>
      <td>0.09</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>730.000000</td>
      <td>652230.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2002</td>
      <td>0.13</td>
      <td>0.0</td>
      <td>0.56</td>
      <td>1029.999971</td>
      <td>652230.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2003</td>
      <td>0.31</td>
      <td>0.0</td>
      <td>0.63</td>
      <td>1220.000029</td>
      <td>652230.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2004</td>
      <td>0.33</td>
      <td>0.0</td>
      <td>0.56</td>
      <td>1029.999971</td>
      <td>652230.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3644</th>
      <td>Zimbabwe</td>
      <td>2016</td>
      <td>3.50</td>
      <td>0.0</td>
      <td>3.32</td>
      <td>11020.000460</td>
      <td>390757.0</td>
    </tr>
    <tr>
      <th>3645</th>
      <td>Zimbabwe</td>
      <td>2017</td>
      <td>3.05</td>
      <td>0.0</td>
      <td>4.30</td>
      <td>10340.000150</td>
      <td>390757.0</td>
    </tr>
    <tr>
      <th>3646</th>
      <td>Zimbabwe</td>
      <td>2018</td>
      <td>3.73</td>
      <td>0.0</td>
      <td>5.46</td>
      <td>12380.000110</td>
      <td>390757.0</td>
    </tr>
    <tr>
      <th>3647</th>
      <td>Zimbabwe</td>
      <td>2019</td>
      <td>3.66</td>
      <td>0.0</td>
      <td>4.58</td>
      <td>11760.000230</td>
      <td>390757.0</td>
    </tr>
    <tr>
      <th>3648</th>
      <td>Zimbabwe</td>
      <td>2020</td>
      <td>3.40</td>
      <td>0.0</td>
      <td>4.19</td>
      <td>NaN</td>
      <td>390757.0</td>
    </tr>
  </tbody>
</table>
<p>3649 rows × 7 columns</p>
</div>

For this dataset, we will have:

| Features                            | Factors | Response |
| ----------------------------------- | :-----: | :------: |
| Year                                |         |          |
| Electricity from fossil fuels (TWh) |    X    |          |
| Electricity from nuclear (TWh)      |    X    |          |
| Electricity from renewables (TWh)   |    X    |          |
| Land Area (Km2)                     |    X    |          |
| Value_co2_emissions_kt_by_country   |         |    X     |

Remember also that according to the [dataset description](./dataset_description.ipynb), the `Value_co2_emissions_kt_by_country` has some missing values that we will have to deal with (drop rows or impute values).

#### Selection for the Primary energy consumption per capita

```python
pe_cols_to_keep = [column for column in corr_matrix.columns if abs(corr_matrix.loc['Primary energy consumption per capita (kWh/person)', column]) > 0.4]
pe_cols_to_keep
```

    ['Access to electricity (% of population)',
     'Access to clean fuels for cooking',
     'Renewable energy share in the total final energy consumption (%)',
     'Primary energy consumption per capita (kWh/person)',
     'gdp_per_capita']

```python
df_pe = df[['Entity', 'Year'] + pe_cols_to_keep]
df_pe.to_csv("../data/pe/pe_dataset.csv", index=False)
df_pe
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe smallest">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entity</th>
      <th>Year</th>
      <th>Access to electricity (% of population)</th>
      <th>Access to clean fuels for cooking</th>
      <th>Renewable energy share in the total final energy consumption (%)</th>
      <th>Primary energy consumption per capita (kWh/person)</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2000</td>
      <td>1.613591</td>
      <td>6.2</td>
      <td>44.99</td>
      <td>302.59482</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2001</td>
      <td>4.074574</td>
      <td>7.2</td>
      <td>45.60</td>
      <td>236.89185</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2002</td>
      <td>9.409158</td>
      <td>8.2</td>
      <td>37.83</td>
      <td>210.86215</td>
      <td>179.426579</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2003</td>
      <td>14.738506</td>
      <td>9.5</td>
      <td>36.66</td>
      <td>229.96822</td>
      <td>190.683814</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2004</td>
      <td>20.064968</td>
      <td>10.9</td>
      <td>44.24</td>
      <td>204.23125</td>
      <td>211.382074</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3644</th>
      <td>Zimbabwe</td>
      <td>2016</td>
      <td>42.561730</td>
      <td>29.8</td>
      <td>81.90</td>
      <td>3227.68020</td>
      <td>1464.588957</td>
    </tr>
    <tr>
      <th>3645</th>
      <td>Zimbabwe</td>
      <td>2017</td>
      <td>44.178635</td>
      <td>29.8</td>
      <td>82.46</td>
      <td>3068.01150</td>
      <td>1235.189032</td>
    </tr>
    <tr>
      <th>3646</th>
      <td>Zimbabwe</td>
      <td>2018</td>
      <td>45.572647</td>
      <td>29.9</td>
      <td>80.23</td>
      <td>3441.98580</td>
      <td>1254.642265</td>
    </tr>
    <tr>
      <th>3647</th>
      <td>Zimbabwe</td>
      <td>2019</td>
      <td>46.781475</td>
      <td>30.1</td>
      <td>81.50</td>
      <td>3003.65530</td>
      <td>1316.740657</td>
    </tr>
    <tr>
      <th>3648</th>
      <td>Zimbabwe</td>
      <td>2020</td>
      <td>52.747670</td>
      <td>30.4</td>
      <td>81.90</td>
      <td>2680.13180</td>
      <td>1214.509820</td>
    </tr>
  </tbody>
</table>
<p>3649 rows × 7 columns</p>
</div>

For this dataset, we will have:

| Features                                                         | Factors | Response |
| ---------------------------------------------------------------- | :-----: | :------: |
| Year                                                             |         |          |
| Access to electricity (% of population)                          |    X    |          |
| Access to clean fuels for cooking                                |    X    |          |
| Renewable energy share in the total final energy consumption (%) |    X    |          |
| Primary energy consumption per capita (kWh/person)               |         |    X     |
| GDP per capita                                                   |    X    |          |

### 3. Higher dimension relationship

#### CO2 emissions prediction

```python
# This prints the number of rows with at least one missing value
df_co2.shape[0] - df_co2.dropna().shape[0]
```

    548

This dataset does not contain a lot of missing values. If we drop rows that are containing we still have more than 3000 samples to train our model. As we have 5 features, this number if samples is sufficient to train a regression model. There is no need of finding higher dimensional relationship.

#### Renewable energy consumption prediction

```python
df_pe.shape[0] - df_pe.dropna().shape[0]
```

    622

Same result are the CO2 emission dataset: even though we drop all the rows contaning missing values, we still have around 3000 samples to train our model. This is sufficient for a regression model with 5 features.

## Feature Sampling

To have a better distribution, we'll normalize our data.

```python
def normalizeDataFrame(df):
    return (df - df.min()) / (df.max() - df.min())
```

```python
def saveDataset(df_to_save, original_df, path):
    df_to_save.insert(0, "Entity", original_df["Entity"])
    df_to_save.insert(1, "Year", original_df["Year"])
    df_to_save.to_csv(path, index=False)
```

### CO2 emissions prediction

```python
normalized_df_co2 = normalizeDataFrame(df_co2.iloc[:, 2:])

normalized_df_co2_to_save = normalized_df_co2.copy()
saveDataset(normalized_df_co2_to_save, df_co2, "../data/co2/normalized_co2_dataset.csv")

normalized_df_co2
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe smallest">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Electricity from fossil fuels (TWh)</th>
      <th>Electricity from nuclear (TWh)</th>
      <th>Electricity from renewables (TWh)</th>
      <th>Value_co2_emissions_kt_by_country</th>
      <th>Land Area(Km2)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000031</td>
      <td>0.0</td>
      <td>0.000142</td>
      <td>0.000070</td>
      <td>0.065321</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000017</td>
      <td>0.0</td>
      <td>0.000229</td>
      <td>0.000067</td>
      <td>0.065321</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000025</td>
      <td>0.0</td>
      <td>0.000256</td>
      <td>0.000095</td>
      <td>0.065321</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000060</td>
      <td>0.0</td>
      <td>0.000288</td>
      <td>0.000113</td>
      <td>0.065321</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000064</td>
      <td>0.0</td>
      <td>0.000256</td>
      <td>0.000095</td>
      <td>0.065321</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3644</th>
      <td>0.000675</td>
      <td>0.0</td>
      <td>0.001519</td>
      <td>0.001028</td>
      <td>0.039134</td>
    </tr>
    <tr>
      <th>3645</th>
      <td>0.000588</td>
      <td>0.0</td>
      <td>0.001968</td>
      <td>0.000965</td>
      <td>0.039134</td>
    </tr>
    <tr>
      <th>3646</th>
      <td>0.000720</td>
      <td>0.0</td>
      <td>0.002499</td>
      <td>0.001155</td>
      <td>0.039134</td>
    </tr>
    <tr>
      <th>3647</th>
      <td>0.000706</td>
      <td>0.0</td>
      <td>0.002096</td>
      <td>0.001097</td>
      <td>0.039134</td>
    </tr>
    <tr>
      <th>3648</th>
      <td>0.000656</td>
      <td>0.0</td>
      <td>0.001918</td>
      <td>NaN</td>
      <td>0.039134</td>
    </tr>
  </tbody>
</table>
<p>3649 rows × 5 columns</p>
</div>

```python
sns.pairplot(normalized_df_co2, height=4, kind="scatter", diag_kind="kde", diag_kws={"linewidth": 1.5, "color": "red"})
```

    <seaborn.axisgrid.PairGrid at 0x7fc490e0bf70>

<p align="center">
  <img src="../../assets/img/dataset_sampling_29_1.png" />
</p>

We can clearly see correlations on scatter plot between features. For feature distribution, is hard to see because countries that are emitting a lot of CO2 are so few compared to those who are not. Computing logarithmic values of the dataset could be a possibility but many samples have 0 as value for the `Electricity from nuclear (TWh)` feature. As $log(0)$ is not defined, we can't use this method. Let's try to remove some outliers.

```python
top_outliers_co2 = df_co2['Value_co2_emissions_kt_by_country'].quantile(0.95)

normalized_df_co2_2 = normalizeDataFrame(df_co2[df_co2['Value_co2_emissions_kt_by_country'] < top_outliers_co2].iloc[:, 2:])
normalized_df_co2_2_to_save = normalized_df_co2_2.copy()
saveDataset(normalized_df_co2_2_to_save, df_co2[df_co2['Value_co2_emissions_kt_by_country'] < top_outliers_co2], "../data/co2/normalized_co2_dataset_without_outliers.csv")

print(normalized_df_co2_2.shape)

sns.pairplot(normalized_df_co2_2, height=4, kind="scatter", diag_kind="kde", diag_kws={"linewidth": 1.5, "color": "red"})
plt.savefig("../img/pairplot_co2.png", dpi=200, bbox_inches="tight")
```

    (3059, 5)

<p align="center">
  <img src="../../assets/img/dataset_sampling_31_1.png" />
</p>

This is better ! We still have enough samples to train our model and tt is a bit easier to see that our data are quite randomly sampled. We have continuous samples and it might seems that the response feature is following a **Weibull** or **Log-normal** distribution.

There is also sufficient variation across all features to support statistical modeling.

### Primary energy consumption per capita

```python
normalized_df_pe = normalizeDataFrame(df_pe.iloc[:, 2:])

normalized_df_pe_to_save = normalized_df_pe.copy()
saveDataset(normalized_df_pe_to_save, df_pe, "../data/pe/normalized_pe_dataset.csv")

normalized_df_pe
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe smallest">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Access to electricity (% of population)</th>
      <th>Access to clean fuels for cooking</th>
      <th>Renewable energy share in the total final energy consumption (%)</th>
      <th>Primary energy consumption per capita (kWh/person)</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.003659</td>
      <td>0.062</td>
      <td>0.468451</td>
      <td>0.001152</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.028581</td>
      <td>0.072</td>
      <td>0.474802</td>
      <td>0.000902</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.082603</td>
      <td>0.082</td>
      <td>0.393898</td>
      <td>0.000803</td>
      <td>0.000547</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.136573</td>
      <td>0.095</td>
      <td>0.381716</td>
      <td>0.000876</td>
      <td>0.000638</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.190513</td>
      <td>0.109</td>
      <td>0.460641</td>
      <td>0.000778</td>
      <td>0.000806</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3644</th>
      <td>0.418333</td>
      <td>0.298</td>
      <td>0.852770</td>
      <td>0.012292</td>
      <td>0.010961</td>
    </tr>
    <tr>
      <th>3645</th>
      <td>0.434707</td>
      <td>0.298</td>
      <td>0.858601</td>
      <td>0.011684</td>
      <td>0.009102</td>
    </tr>
    <tr>
      <th>3646</th>
      <td>0.448824</td>
      <td>0.299</td>
      <td>0.835381</td>
      <td>0.013108</td>
      <td>0.009260</td>
    </tr>
    <tr>
      <th>3647</th>
      <td>0.461066</td>
      <td>0.301</td>
      <td>0.848605</td>
      <td>0.011439</td>
      <td>0.009763</td>
    </tr>
    <tr>
      <th>3648</th>
      <td>0.521484</td>
      <td>0.304</td>
      <td>0.852770</td>
      <td>0.010207</td>
      <td>0.008935</td>
    </tr>
  </tbody>
</table>
<p>3649 rows × 5 columns</p>
</div>

```python
sns.pairplot(normalized_df_pe, height=4, kind="scatter", diag_kind="kde", diag_kws={"linewidth": 1.5, "color": "red"})
```

    <seaborn.axisgrid.PairGrid at 0x7fc4a181e8e0>

<p align="center">
  <img src="../../assets/img/dataset_sampling_35_1.png" />
</p>

For this problem, it is easier to see that all the input features are well distributed: our data are more randomly sampled than for the CO2 problem. However, it is a bit more difficult to see that for the response feature. We can try to remove some outliers to have a better distribution.

```python
top_outliers_pe = df_pe['Primary energy consumption per capita (kWh/person)'].quantile(0.95)

normalized_df_pe_2 = normalizeDataFrame(df_pe[df_pe['Primary energy consumption per capita (kWh/person)'] < top_outliers_pe].iloc[:, 2:])
normalized_df_pe_2_to_save = normalized_df_pe_2.copy()
saveDataset(normalized_df_pe_2_to_save, df_pe[df_pe['Primary energy consumption per capita (kWh/person)'] < top_outliers_pe], "../data/pe/normalized_pe_dataset_without_outliers.csv")

print(normalized_df_pe_2.shape)

sns.pairplot(normalized_df_pe_2, height=4, kind="scatter", diag_kind="kde", diag_kws={"linewidth": 1.5, "color": "red"})
plt.savefig("../img/pairplot_pe.png", dpi=200, bbox_inches="tight")
```

    (3466, 5)

<p align="center">
  <img src="../../assets/img/dataset_sampling_37_1.png" />
</p>

Only by removing 5% of the outliers, we can see that the distribution of the continuous response feature is better ! Moreover, the shape of this distribution looks like a **Log-normal distribution**.

There is also more variation across all these features to support statistical modeling than for the CO2 emissions prediction problem, which is good.
