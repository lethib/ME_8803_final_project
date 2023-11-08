# List of all the mistakes I have made during this project

## 1. Not defining a clear goal during the data analysis

At first I wanted to llok at the feature distribution to pick the ones that were well sampled. But, ploting a 21x21 pair plot was totally unreadable.

**What have I learned?**

- Doing feature selection by ploting simple high dimension plots such as correlation matrix is necessary to select features that we want to study. (I could have also use PCA to reduce the dimensionality of the data). This will naturally lead to goals that I want to machine learn.
- Based on those plots, we can clearly identify the features that are correlated to each other and remove unnecessary features from the analysis.
- After getting a _new low dimension dataset_ with all the features we want, we can plot some low dimension figures to get a better understanding of the data (distributions, possible outliers, etc.).

## 2. Doing extrapolation with regression models

The questions that I wanted to solve at first were the following ones:

- Predicting the CO2 emissions by country for the next 5 years
- Predicting the Renewable energy consumption by country for the next 5 years

As I don't have the data for the next 5 years, I tried to extrapolate the data using regression models. But, this is not a good idea as the data is not stationary. The data is not stationary because the CO2 emissions and the Renewable energy consumption are increasing over time. So, the model will not be able to predict the future values.

**What have I learned?**

- Working with time series data, it is really hard to make predictions outside the time range of the data. I still have regression problems to solve, but I will not be able to predict the future values.
