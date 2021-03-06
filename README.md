# Telco-Customer-Churn-Prediction

## Introduction:
Nowadays, telecommunication industries are extremely saturated, and service providers are experiencing high rates of churn. We seek to help telecom companies improve customer rentention rate by leveraging data mining and machine learning.

## Data Source:
- Data Set: Kaggle (Originally from IBM Watson analytics data)
- Dimension: 7043 rows x 20 columns
- Details: 3 numerical columns('tenure', 'MonthlyCharges', 'TotalCharges'), 17 categorical/binary columns, target column: 'Churn'
- Each row indicates a unique customer who is identified by the column 'customerID'


## Objective
1. Use various feature engineering techniques and train several classification models to select the most optimal ones with high predicting power
2. Discover the key factors that would have the biggest impact on customer’s churn rate
3. Provide specific business strategies that are easy to implement, from both a short-term and long-term perspective

## Tools used:
- EDA, Modeling: Python Jupyter Notebook
- Presentation: Microsoft Powerpoint

## Files explained:
1. **DMP Final Project - Step 1A_LabelEncoder Experiment.ipynb**
2. **DMP Final Project - Step 1B_GetDummies Experiment.ipynb**
3. **DMP Final Project - Step 2_Get Tweets.ipynb**
4. **DMP Final Project - Step 3_Assign Tweets to Original Data.ipynb**
5. **DMP Final Project - Step 4_Model Run with Augmented Data.ipynb**
6. **WA_Fn-UseC_-Telco-Customer-Churn.csv**: the original dataset
7. **TweetSentiment.csv**: tweeter sentiment data used in tweet sentiment analysis
8. **telco_with_polarity.csv**: the revised telco dataset where an additional column of tweet polarity is added to the original dataset

## Result:
1. Detected feature collinearity using pariwise correlation plots and conducted outlier analysis
2. Experimented different variable transformations, two grouping methods, dimension reduction with PCA, resampling using SMOTE(hybrid of regular and under), and used augmented data with tweet sentiments
3. Applied logistic regression, KNN, SVM, decision tree, and random forest ML models and used grid search for hyper-parameter tuning
4. Achieved 80% accuracy score with 0.63 F1 score for Churn (9% improvement) and 0.85 F1 score for Not Churn using logistic regression
5. Proposed 4 business strategies based on analysis of key factors extracted from feature importance charts:
    - Offer Long-Term Contracts if Possible
    - Provide Discounts and Incentives to High-Risk Customers
    - Be Proactive and Responsive to Customer’s Complains and Questions on Social Platforms
    - Invest in Innovation and Create Competitive Advantages Over Others
