# House Price Prediction System

## Dashboard
![Dashboard](<House Price Dashboard .png>)

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

# Project Setup
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

# Data Understanding and Data Processing
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

- The **'MiscVal'** column was dropped as it mostly contained zero values, with only 3 observations having non-zero values. Due to its lack of variability, this column did not provide meaningful information and could potentially introduce noise into the analysis.

- From the correlation matrix, I identified **'GarageArea', '1stFlrSF', 'TotRmsAbvGrd'**
variables that were highly correlated with others that were more valuable to the model. Instead of keeping these redundant features, I retained the more relevant variables and dropped the ones that were less useful.

- The **'Id'** column was removed because it was merely an identifier and did not contribute any predictive value for the model.

# Feature Selection and Feature Engineering
## Tasks
**1. Additional features were engineered to improve model performance.**

Engineered features for house price prediction: Total Square Footage, Total Bathrooms, Garage Size, Garage Age, House Age, and Total Rooms (including Bedrooms).

**2. Feature Scaling and Encoding**

RobustScaler was applied to numerical features to handle outliers
One-Hot Encoding was applied to categorical features

**3. Feature Selection Using Random Forest**

A Random Forest Regressor was trained to determine feature importance, and the top features were selected based on their contribution to predicting house prices.

**4. Feature Selection Using Lasso Regression (Optional)**

A Lasso Regression model was applied to shrink less important feature coefficients to zero, effectively selecting the most relevant predictors.

# Supervised Model Building
## Tasks

**1. Model Training**"

Each model was initialized and trained on the dataset:

A `Random Forest Regressor` was initialized with 500 estimators and trained.

A `Gradient Boosting Regressor` was initialized and trained.

A `XGBoost Regressor` was trained on the dataset.

A `Decision Tree Regressor` was trained on the dataset.

A `SVM Regressor` was trained on the dataset.

**2. Model Predictions**

Predictions were generated on the training set for each model.

**3. Model Evaluation**

The models were evaluated using Mean Squared Error (MSE) and R² Score, including cross-validation scores.

**1. Random Forest Regressor**

    Training Set Evaluation:

    MSE: 0.0027

    R² Score: 0.9830

    Adjusted R² Score: 0.9826

    **Cross-Validation:**

    Mean MSE: 0.0205

    R² Score: 0.8715

    MSE for each fold: [0.0183, 0.0243, 0.0206, 0.0165, 0.0227]

    R² for each fold: [0.8775, 0.8654, 0.8770, 0.8858, 0.8518]

**2. Gradient Boosting Regressor**

    Training Set Evaluation:

    MSE: 0.0018

    R² Score: 0.9887

    Cross-Validation:

    Mean MSE: 0.0200

    R² Score: 0.8875

    MSE for each fold: [0.0167, 0.0185, 0.0186, 0.0147, 0.0208]

    R² for each fold: [0.8879, 0.8976, 0.8893, 0.8985, 0.8643]

**3. XGBoost Regressor**

    Training Set Evaluation:

    MSE: 0.0024

    R² Score: 0.9851

    Cross-Validation:

    Mean MSE: 0.0200

    R² Score: 0.8856

    MSE for each fold: [0.0153, 0.0210, 0.0187, 0.0136, 0.0227]

    R² for each fold: [0.8974, 0.8840, 0.8888, 0.9063, 0.8515]

**4. Decision Tree Regressor**

    Training Set Evaluation:

    MSE: 0.0237

    R² Score: 0.8511

    Cross-Validation:

    Mean MSE: 0.0375

    R² Score: 0.7639

    MSE for each fold: [0.0348, 0.0456, 0.0339, 0.0370, 0.0364]

    R² for each fold: [0.7672, 0.7476, 0.7980, 0.7443, 0.7622]

**5. SVM Regressor**

    Training Set Evaluation:

    MSE: 0.0058

    R² Score: 0.9633

    Cross-Validation:

    Mean MSE: 0.0300

    R² Score: 0.8311

    MSE for each fold: [0.0241, 0.0367, 0.0261, 0.0230, 0.0253]

    R² for each fold: [0.8384, 0.7969, 0.8445, 0.8413, 0.8346]

**Results:**

- Among all models, the **Gradient Boosting Regressor** achieved the best performance, with an **R² Score of 0.9887** on the training set and a **cross-validated R² Score of 0.8875**, indicating strong generalization. The **XGBoost Regressor** followed closely, achieving a training **R² of 0.9851** and a **cross-validated R² of 0.8856**, making it another highly effective model.

- The **Random Forest Regressor** also performed well, with a **training R² of 0.9830** and a **cross-validated R² of 0.8715**, but it showed **slight overfitting**. The **SVM Regressor** achieved a **training R² of 0.9633** and a **cross-validated R² of 0.8311**, performing decently but lagging behind the boosting models.

- The **Decision Tree Regressor** had the lowest accuracy, with an **R² of 0.8511** on training data and a significantly lower **cross-validated R² of 0.7639**, indicating **strong overfitting**. It performed significantly better on the training set compared to cross-validation results, highlighting its poor generalization ability.

# Hyperparameter Tuning, Model Evaluation and Model Selection
## Tasks


1. **Hyperparameter Tuning**
Optimized the hyperparameters for Random Forest, Gradient Boosting, and XGBoost models using RandomizedSearchCV to find the best hyperparameters efficiently.

2. **Model Training and Evaluation**

    - **Gradient Boosting & XGBoost** demonstrated the best performance  with a Mean MSE of 0.017 and a Mean R² Score of 0.893.

    - Random Forest had slightly worse performance with a higher Mean MSE (0.021) and a lower Mean R² Score (0.871).

3. **Final Model Selection and Prediction**
    - Chose Gradient Boosting Regressor as the best-performing model.
    - Generated predictions using the Gradient Boosting Regressor.
    - Applied the exponential function (np.exp()) to revert the log transformation and obtain the original scale.
    - Saved the final predictions to final_predictions.csv.







