# Auto-ML Report: Age Criteria Analysis
## Executive Summary
This report presents the findings of an automated machine learning (auto-ML) pipeline designed to analyze the age criteria of passengers who died. The goal of this project is to identify the age group with the highest mortality rate. The auto-ML pipeline analyzed a dataset containing 891 rows and 12 columns, with a mix of integer, float, and string data types.

## Data Insights
The data analysis revealed the following key findings:
* The mean age of passengers is 29.7 years, with a standard deviation of 14.53 years.
* The majority of passengers are in their 20s and 30s.
* There is a correlation between `Pclass` and `Fare`, with higher classes having higher fares.
* The `Survived` column is potentially correlated with `Age`, `Sex`, and `Pclass`, which could influence survival rates.
* The `Age` column has 177 missing values (19.87% of the data), which were not addressed in this analysis.

## Preprocessing Steps
The preprocessing steps involved:
* Selecting the relevant columns for analysis, resulting in a dataset with 9 columns.
* Vectorizing text columns (although none were present in this dataset).
* The resulting dataset had a shape of 891 rows and 9 columns.

## Model Performance
The auto-ML pipeline trained and evaluated several regression models, including:
* Linear Regression
* Ridge
* Lasso
* Decision Tree
* Random Forest
* Gradient Boosting
* KNN
* SVR

The results showed that **Gradient Boosting** was the best-performing model, with an **R-squared score of 0.248**. This indicates that the model can explain approximately 24.8% of the variance in the target variable `Age`. The model's performance metrics are:
* Mean Absolute Error (MAE): 8.4478
* Root Mean Squared Error (RMSE): 11.3143

## Conclusion & Recommendations
Based on the analysis, we can conclude that:
* The age group with the highest mortality rate is not explicitly identified in this report, as the target variable is `Age` and the goal is to predict the age of passengers who died.
* The best-performing model, Gradient Boosting, can be used to make predictions about the age of passengers who died.
* However, the model's performance can be improved by addressing the missing values in the `Age` column and exploring additional features that may be relevant to the problem.

Recommendations for future work include:
* Imputing the missing values in the `Age` column using techniques such as mean, median, or imputation using other features.
* Exploring additional features that may be relevant to the problem, such as socioeconomic status or family relationships.
* Tuning the hyperparameters of the Gradient Boosting model to improve its performance.
* Using the model to make predictions about the age of passengers who died and analyzing the results to identify the age group with the highest mortality rate.