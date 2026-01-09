# Customer-Churn-Analysis-for-a-Subscription-Based-Streaming-Platform
This project analyzes customer churn for a subscription-based streaming platform to unerstand why the customers stop using the service, the key churn drivers and data based recommendations to retain users


# Project Overview
Subscription based streaming platforms rely on recurring monthly subscription making customer retention the most important aspect for business performance. Customer churn occurs when users discontinue the subscription or stop using the product. Even a small increse in customer churn can have a huge impact on the revenue and business sustainability.
The dataset consists of customer level information including customer behaviour (watch hours, number of profiles),subscription types, customer demographics (age, gender, region). The analysis focuses on seeing comparing patterns based on these attributes for active and churned users to draw insights on key churn drivers and generated recommendation to retain customers and reduce churn rate.


# Dataset Description

The dataset contains customer-level information including:

Behavioral metrics (watch hours, average watch time per day, last login days)

Subscription and revenue attributes (subscription type, monthly fee, payment method)

Customer attributes (age, gender, region, device, favorite genre)

Target variable: churn status (Active / Churned)

# Exploratory Data Analysis (EDA)

Univariate analysis was used to understand distributions and characteristics of individual features.

Bivariate analysis compared churned vs active users to identify churn patterns.

Behavioral variables showed the strongest separation between churned and active customers.

# Predictive Modeling

Model used: Logistic Regression

Evaluation metric: Accuracy

Model accuracy achieved: ~89.7%

Key Predictors of Churn

Days since last login (strongest churn indicator)

Low watch hours and low daily engagement

Payment methods such as Gift Cards and Crypto

Lower subscription tiers (Basic plans)

Demographic variables such as age, gender, region, and device showed minimal impact.

# Key Insights
Days since inactivity and non-traditional payment methods are the key predictive metrics for user churn
Churned users show much lower watch hours and watch time per day
When users share an account they are more likely to continue with their subscription
Similarily, Users with higher tier plans are more likely to be retained compared to basic plan users
Age, Gender, Device and Region distrubution show negligible variance in user churn and active users meaning these are paradoxially not useful to determine which users could churn
Based on time related metrics, churned users show lower activity and engagement relative to active users

# Business Recommendations
Improve activity by pushing more curated notifications and emails based on user past preferred content
Coax users to shift from card and crypto payment methods to auto-renewing card and online payments
Promote upgrade to higher tiered subscription plans especially the premium plan
Promote sharing of accounts as opposed to single user accounts
focus more on time and engagement related metrics rather demographic categories like age, gender, device and region


# Tools & Technologies

Python (Pandas, NumPy)

Matplotlib & Seaborn

Scikit-learn

Jupyter Notebook

# Conclusion

Improving user inactivity and focusing on user in-app engagement and promoting auto renewing card payments are the most effective methods to retaing users and prevent churn

# Author

Adnan Khader
Aspiring Business & Data Analyst
