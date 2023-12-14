# Primary Energy (PE) Consumption

The objective of this project is to assess the suitability of an engineering dataset for machine learning modeling, determine data features that should be used for machine learning, and then fit and critique the performance of ML model(s) using uncertainty quantification. This project is part of the Engineering Applications of Machine Learning course at Georgia Tech.

## Dataset used

The dataset is showcasing sustainable energy indicators and other useful factors across all countries from 2000 to 2020.

It can be obtained from [Kaggle](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy/data).

## Objectives

After some _Explanatory Data Analysis_ (see [EDA](./EDA/)), I have listed 2 possible ojectives for this project from the [dataset_sampling notebook](/EDA/dataset_sampling.ipynb):

- Predicting the CO2 emissions
- Predicting the primary energy consumption per capita

Many projects are predicting the CO2 emissions, so I will focus on the second objective. That's why in the modeling part of the project, I will focus on predicting the primary energy consumption per capita.

## Project Structure

- **EDA**: Contains the notebooks used for the _Exploratory Data Analysis_.

- **data**: Contains the dataset used for this project. It is structured like this:

  ```
  .
  └── data/
    ├── co2/
    ├── pe/
    │   └── cleaned/
    └── global-data-on-sustainable-energy.csv
  ```

  The `global-data-on-sustainable-energy.csv` is the original dataset.

  The [co2/](./data/co2/) and [pe/](./data/pe/) folders contain the data (after some normalization steps and features extraction from the original dataset) for the CO2 emissions prediction problem and the primary energy consumption per capita prediction problem respectively. The [cleaned/](./data/pe/cleaned/) folder contains those same dataset but after some cleaning steps.

- **feature_engineering**: Contains the notebook used for feature selection and data cleaning.

- **modeling**: Contains the notebooks used for the modeling part of the project.

- **website**: Contains all the files used to build the Github Pages of the project.

- `mistakes.md`: Contains the big mistakes where I have lost a lot of time during the project.

- `personal_notes.md`: Contains some personal notes that I have taken during the project to understand deeper some ML concepts that were not clear to me.

## Conclusion

### Performance against random guessing

Accuracies that I have obtained with all the models are way above 0.5, which is the accuracy that we would obtain if we were to randomly guess. This is a good sign, as it means that the models are learning something from the data.

### Modeling approach against other projects

On Kaggle, many notebooks are not that detailed. When I was looking what other users have done, many were answering the same problem with sometimes different models. Also not a lot of explanations were given. I have tried to be as detailed as possible in my approach, and I have tried to use different models to see which one would perform the best.

### Broader impacts of the ML modeling

This model can be used for other studies or anyone that would like to predict the Primary Energy Consumption per Capita of a country given some of its characteristics (GPD, Access to electricy, Renewable Energy Consumption, etc.). If data are within the same range as the data used for this project, the model should perform well (take the Elastic Net model if your data are part of outliers).

### Model improvement

The model could be improved by adding more data around the outliers that I have identified. By doing that, we will have better results for the global and normalized datasets. Doing some data augmentation can be useful for this task but won't really represent the reality of the world.

### Work extension

The work I have done here can be used to solve other problems. Obviously models couldn't be reused but the method is here. I have tried to be as detailed as possible in my approach, and I have tried to use different models to see which one would perform the best. This can be used as a template for other projects.

```

```
