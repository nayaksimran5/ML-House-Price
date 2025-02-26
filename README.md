# House Price Prediction System
## Project Description
The House Price Prediction System is a comprehensive machine learning project designed to predict the sale prices of residential properties based on various features. This project aims to leverage historical data to build a predictive model that can assist buyers, sellers, and real estate professionals in making informed decisions regarding property transactions.
## Objectives
- **Data Analysis**: To explore and analyze the dataset, identifying key features that influence house prices.
- **Data Preprocessing**: To clean and preprocess the data, handling missing values, encoding categorical variables, and normalizing numerical features.
- **Model Development**: To implement and compare various machine learning algorithms, including linear regression, decision trees, random forests, and gradient boosting methods.
- **Model Evaluation**: To evaluate the performance of different models using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared values.
- **Model Optimization**: To fine-tune the selected model using techniques like cross-validation and hyperparameter tuning for improved accuracy.
## Data Sources
The dataset used for this project is derived from the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) competition on Kaggle.
# Week 1 - Project Setup
## Tasks
**1. Setting up Git/GitHub**
- Initialize a Git repository for version control.
- Link the project to a GitHub repository for remote tracking and collaboration.

**2. Setting up Python, VSCode, and Jupyter Notebook**
- Install Python and configure it for the project.
- Set up VSCode as the primary IDE with required extensions.
- Install Jupyter Notebook for exploratory data analysis.

**3. Setting up AWS**
- Configure AWS services for data storage and computational resources.

**4. Setting up Virtual Environment using Pipenv**
- Install and initialize a virtual environment using `pipenv` to manage dependencies effectively.
- Add essential libraries and packages required for the project.

# Week 2 - Data Understanding and Data Processing
## Tasks
**1. Importing Libraries**
The first step was to import the essential libraries, such as `pandas`, `numpy`, `seaborn`, `matplotlib` for data analysis and visualization. 

**2. Loading the Dataset**
After importing the necessary libraries, I proceeded to load the datasets into the environment. This step was crucial to begin analyzing the data and performing necessary transformations.

**3. Understanding the data**

- **Data Overview:** I used `train_df.info()` to examine the number of features, data types, and missing values in the dataset.

- **Data Shape:** To understand the dimensions of the dataset, I used `train_df.shape` to check the number of rows and columns.

- **Summary Statistics:** I employed `train_df.describe()` to analyze summary statistics for numerical columns, providing insights into the distribution, central tendency, and spread of the data.

**4. Identifying Numeric and Categorical Features**
I classified features as either numeric or categorical based on their data types: `int64`, `float64`, and `object`.

- **Data Visualization (Numeric Type Variables)**: To better understand the distribution and relationships of numeric variables, I visualized them using histograms and boxplots. The histograms helped identify the frequency of specific values, while the boxplots revealed the spread of the data and highlighted potential outliers.

- **Understanding Numerical Columns**: The visualizations allowed for a deeper understanding of the numeric features and their distributions, helping to identify patterns, distributions, and anomalies.

- **Data Visualization (Object Type Variables)**: For categorical features, I used bar charts to visualize the frequency of each category, which helped in understanding the distribution of values across different categories.

- **Understanding Categorical Columns**: The bar charts provided insight into the distribution of categorical variables, highlighting imbalances or trends that could inform feature engineering or imputation strategies.

**5. Checking for Duplicate Values**

I used the `duplicated()` method from pandas to check for duplicate entries in the dataset. After running the check, I found that there were no duplicate values in the dataset, ensuring data consistency for model training.

**6. Checking for correlations**

A correlation matrix was generated to identify relationships between numerical features in the dataset. This matrix provided insights into which features are highly correlated with one another. Some notable highly correlated pairs include:

- **'GarageCars'** and **'GarageArea'**: Correlation of 0.88
- **'GrLivArea'** and **'TotRMsAbvGrd'**: Correlation of 0.83
- **'TotalBsmtSF'** and **'1stFlrSF'**: Correlation of 0.82

Additionally, I found that **'OverallQual'**, **'GrLivArea'**, and **'TotalBsmtSF'** are strongly correlated with the target variable, **'SalePrice'**. These correlations suggest that these features have a significant relationship with the target, making them potentially important predictors for modeling.


**7. Significance Test**

- Conducted ANOVA test to identify statistically significant variables.
- Retained only the significant features that impact the target variable (SalePrice).
- Dropped non-significant variables to enhance model performance.

**8. Handling Missing Values**
I first calculated the percentage of missing values for each feature to understand the extent of the gaps in the data.

- **Numeric Features**: Missing values in numeric columns were filled using KNN imputation, which predicts missing values based on the nearest neighbors in the dataset.

- **Categorical Features**: For categorical columns, missing values were imputed with the mode (the most frequent value) of each column.

- Dropped columns with a high percentage of missing value.

**9. Target Variable Exploration and Transformation**
- The **SalePrice** variable had a right-skewed distribution.
- Applied a log transformation to normalize the skewness.
- The transformation made the target variable more suitable for modeling.

**10. Removing or Dropping Unnecessary Features**

In this step, I removed or dropped features that were either unnecessary or did not add significant value to the model.

The **'MiscVal'** column was dropped as it mostly contained zero values, with only 3 observations having non-zero values. Due to its lack of variability, this column did not provide meaningful information and could potentially introduce noise into the analysis.

From the correlation matrix, I identified **'GarageArea', '1stFlrSF', 'TotRmsAbvGrd'**
variables that were highly correlated with others that were more valuable to the model. Instead of keeping these redundant features, I retained the more relevant variables and dropped the ones that were less useful.

The **'Id'** column was removed because it was merely an identifier and did not contribute any predictive value for the model.

# Week 3 - Feature Selection and Feature Engineering
## Tasks
New features were created to enhance model performance

**Total Square Footage Features:**
- TotalSF
- TotalPorchSF
- TotalOutdoorSF

**TotalBathrooms:** Total number of bathrooms, weighting half-baths as 0.

**Garage Features:**
- GarageSize: 
- GarageAge:

**Age of the House:**
- HouseAge:
- RemodelAge:

**TotalRooms:** Sum of total rooms above ground and bedrooms.

**Basement Presence**
- HasBasement 

**Garage Presence**
- HasGarage:

**Pool Presence**
- HasPool:

**Fireplace Presence**
- HasFireplace:

**2nd Floor Presence**
- Has2ndfloor:

**Feature Scaling and Encoding**

RobustScaler was applied to numerical features to handle outliers
One-Hot Encoding was applied to categorical features

**Feature Selection Using Random Forest**

A Random Forest Regressor was trained to determine feature importance
