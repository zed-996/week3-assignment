# Customer Intelligence System - Country Segmentation

## Overview

This project applies unsupervised machine learning to segment 167 countries using
socio-economic and health indicators. The main goal was to figure out which countries
need humanitarian aid most urgently, which are on a development trajectory, and which
are already stable economies.

Dataset: Unsupervised Learning on Country Data (Kaggle)
https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data


## Files

Customer_Intelligence_System.ipynb   - main notebook with full analysis
country_clusters.csv                 - output file with cluster labels for all 167 countries
data-dictionary.csv                  - feature descriptions from the dataset
Country-data.csv                     - raw dataset


## Features in the Dataset

child_mort   - child mortality rate per 1000 births
exports      - exports as % of GDP
health       - health spending as % of GDP
imports      - imports as % of GDP
income       - net income per person (USD)
inflation    - annual inflation rate (%)
life_expec   - average life expectancy (years)
total_fer    - total fertility rate (children per woman)
gdpp         - GDP per capita (USD)


## What Was Done

**EDA** - checked distributions, correlation heatmap, boxplots for outliers.
Strong correlations found: child_mort vs life_expec (-0.89), child_mort vs
total_fer (+0.85), income vs gdpp (+0.97).

**Preprocessing** - StandardScaler applied to all numeric features before
clustering. Features like gdpp (up to $105,000) would otherwise dominate
distance calculations over features like health (1.8–17.9).

**K-Means Clustering** - used Elbow method and Silhouette scores across
k=2 to 10. Settled on k=3 because the elbow flattens after that and k=3
produces a split that actually means something (Developed / Developing /
Underdeveloped). Silhouette score: ~0.47.

**DBSCAN** - run alongside K-Means with eps=1.5, min_samples=3. Found 139
countries in the main cluster, 3 in a small secondary cluster, and 25 outliers.
The outliers are either extreme poverty cases or oil-wealth anomalies.

**PCA Visualization** - reduced to 2 components for plotting. The three
K-Means clusters separate cleanly. Notable countries labeled on the scatter plot.

**Random Forest + XGBoost** - both trained on the K-Means labels to validate
cluster quality and rank feature importance. High accuracy (~95%) means the
clusters are well-separated, not that the models predict anything new.

**Feature Importance** - both models agree: child_mort is the most important
feature by a clear margin, followed by life_expec, gdpp, and income.


## Results

Three clusters:

Developed (36 countries)
  Child mortality: 5/1000 | Life expectancy: 80 yrs | Income: $45,672 | GDP: $42,494

Developing (84 countries)
  Child mortality: 22/1000 | Life expectancy: 73 yrs | Income: $12,306 | GDP: $6,487

Underdeveloped (47 countries)
  Child mortality: 93/1000 | Life expectancy: 59 yrs | Income: $3,942 | GDP: $1,922

The child mortality gap between Developed and Underdeveloped is 18x.


## Countries Most in Need of Aid

Ranked by child mortality within the Underdeveloped cluster:

1. Haiti              - 208/1000, life expectancy 32 years
2. Sierra Leone       - 160/1000
3. Chad               - 150/1000
4. Central African Republic - 149/1000
5. Mali               - 137/1000

Haiti is a significant outlier even within the worst cluster. DBSCAN flags it
separately from the rest.

Equatorial Guinea is worth noting: income of $33,700 and GDP of $17,100, yet
child mortality is 111. Oil revenue is clearly not reaching the population.
Standard GDP-based targeting would miss this country.


## Model Performance

| Model          | CV Accuracy | Test Accuracy |
|----------------|-------------|---------------|
| Random Forest  | ~95%        | ~95%          |
| XGBoost        | ~95%        | ~95%          |

Both models were tuned with GridSearchCV (5-fold CV). The accuracy reflects
cluster separability, not real-world predictive power. The feature importance
output is the more useful result here.


## How to Run

1. Make sure Country-data.csv and data-dictionary.csv are in the same folder
   as the notebook
2. Install dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
3. Run all cells top to bottom
4. country_clusters.csv will be saved in the same directory after the last cell


## Dependencies

Python 
pandas, numpy, matplotlib, seaborn
scikit-learn
xgboost
