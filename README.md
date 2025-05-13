# test_bootcamp_final
Final for NYU Data Bootcamp
# Put your final project here.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# I trimmed the raw dataset imported from UCI’s Machine Learning Database, “adult,”  by dropping insignificant null values, and grouping unnecessarily independent categories for each categorical feature. I am then able to analyze dataset “adult” with little interference from missing data, and draw insights from the dataset more efficiently.
# Problem Statement
# For my final project, I aim to develop a predictive model that would accurately classify whether an individual earns more than $50,000 per year 
# based on demographic and employment information. Predicting income class can provide valuable insights for policymakers, businesses, and researchers 
# interested in understanding the factors associated with higher earnings. By analyzing these patterns, stakeholders can better target educational 
# resources, design effective social programs, and identify key predictors of economic success.
# For example, organizations can use these predictions to better understand the profiles of higher earners, while researchers can use the findings to 
# inform further studies on economic mobility and workforce diversity.
pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

adult=X.copy()
adult['income']=y

# Number of missing values per feature
adult.isna().sum()

# The number of missing values is much smaller than the total number of instances,
total_instances=len(adult)
print(len(adult))

# so it's reasonable to drop rows with missing data
adult_cleaned=adult.dropna()
adult_cleaned.info()

adult_cleaned.head()

# Proportion Analysis by Category: Categorical Variable Selection
categorical_cols = adult_cleaned.select_dtypes(include=['category', 'object']).columns.tolist()
height_per_category = 0.3

for col in categorical_cols:
    if col != 'income':
        proportions = pd.crosstab(adult_cleaned[col], adult_cleaned['income'], normalize='index')
        if '>50K' in proportions.columns:
            proportions = proportions['>50K'].sort_values()

            n_categories = adult_cleaned[col].nunique()
            total_height = height_per_category * n_categories

            fig, ax = plt.subplots(figsize=(12, total_height))
            proportions.plot(kind='barh', color='skyblue', ax=ax)
            ax.set_title(f"Proportion earning >50K by {col}")
            ax.set_xlabel("Proportion >50K")
            ax.set_xlim(0, 0.6)
            ax.tick_params(axis='y', labelsize=8)

# Workclass: drop unclear/low-value entries
adult_cleaned = adult_cleaned[~adult_cleaned['workclass'].isin(['?', 'Without-pay', 'Never-worked'])]

# Education: grouping mid-edu and lower-edu
mid_edu = ['Some-college', 'Assoc-acdm', 'Assoc-voc']
adult_cleaned['education'] = adult_cleaned['education'].replace({edu: 'Mid-edu' for edu in mid_edu})

low_edu = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']
adult_cleaned['education'] = adult_cleaned['education'].replace({edu: 'Low-edu' for edu in low_edu})

# Occupation: grouping manual labor and uncommon jobs
adult_cleaned = adult_cleaned[~(adult_cleaned['occupation']=='?')]

high_skill = ['Exec-managerial', 'Prof-specialty', 'Tech-support', 'Protective-serv']
mid_skill = ['Sales', 'Adm-clerical', 'Armed-Forces']
low_skill = ['Craft-repair', 'Machine-op-inspct', 'Manual-labor', 'Other-service', 'Transport-moving',
             'Farming-fishing', 'Handlers-cleaners', 'Priv-house-serv']
adult_cleaned['occupation'] = adult_cleaned['occupation'].replace(
    {occ: 'High-skill' for occ in high_skill} |
    {occ: 'Mid-skill' for occ in mid_skill} |
    {occ: 'Low-skill' for occ in low_skill}

)

# Relationship: grouping low-value relationship types
adult_cleaned['relationship'] = adult_cleaned['relationship'].replace({
    'Own-child': 'Dependent',
    'Other-relative': 'Dependent'
})

# Country: grouping by development level
adult_cleaned = adult_cleaned[
    ~adult_cleaned['native-country'].isin(['Outlying-US(Guam-USVI-etc)', '?'])
]
developed = ['Canada', 'England', 'Germany', 'France', 'Italy', 'Japan', 'Portugal', 'Ireland', 'Scotland', 'Holand-Netherlands',
             'Yugoslavia', 'Hungary', 'Greece', 'Poland']

developing = ['Mexico', 'India', 'China', 'Philippines', 'Vietnam', 'Columbia', 'Guatemala',
              'Dominican-Republic', 'Jamaica', 'El-Salvador', 'Honduras', 'Ecuador',
              'Peru', 'Nicaragua', 'Cambodia', 'Thailand', 'Laos', 'Taiwan', 'Iran',
              'South', 'Hong', 'Cuba', 'Puerto-Rico', 'Trinadad&Tobago', 'Haiti'
]

US = ['United-States']

adult_cleaned['native-country'] = adult_cleaned['native-country'].replace(
    {c: 'US' for c in US} |
    {c: 'Developed' for c in developed} |
    {c: 'Developing' for c in developing}
)

adult_cleaned['income'].str.strip().str.replace('.', '', regex=False)

adult_cleaned.info()

# Post-trimming
categorical_cols = adult_cleaned.select_dtypes(include=['category', 'object']).columns.tolist()
height_per_category = 0.3

adult_cleaned['income'] = adult_cleaned['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})

for col in categorical_cols:
    if col != 'income':
        proportions = pd.crosstab(adult_cleaned[col], adult_cleaned['income'], normalize='index')
        if '>50K' in proportions.columns:
            proportions = proportions['>50K'].sort_values()

            n_categories = adult_cleaned[col].nunique()
            total_height = height_per_category * n_categories

            fig, ax = plt.subplots(figsize=(12, total_height))
            proportions.plot(kind='barh', color='skyblue', ax=ax)
            ax.set_title(f"Proportion earning >50K by {col}")
            ax.set_xlabel("Proportion >50K")
            ax.set_xlim(0, 0.6)
            ax.tick_params(axis='y', labelsize=8)

# Response variable analysis
# As illustrated below, the distribution of income in the dataset is imbalanced, 
# as the majority of individuals earn <=50K.
adult_cleaned['income'].value_counts().plot(kind='bar')
adult_cleaned['income'].value_counts(normalize=True)

adult_cleaned.head()

# Individuals with higher education levels (higher education-num) are more likely to earn above $50K. 
# The median and upper-quartile education level for the >50K group is higher than for the <=50K group.
# This suggests that education is an effective predictor of higher income, 
sns.boxplot(x='income', y='education-num', data=adult_cleaned)
plt.title('Education Level by Income')

# The clear separation between the income groups for different marital statuses 
# indicates that this variable will likely be useful for modeling purposes
sns.countplot(x='marital-status', hue='income', data=adult_cleaned)
plt.title('Income by Marital Status')

# A much higher proportion of males earn >50K compared to females. 
# For the <=50K group, however, the counts for males and females are more balanced.
# Therefore, gender is a significant determinant of income.
sns.countplot(x='sex', hue='income', data=adult_cleaned)
plt.title('Income by Sex')

# Baseline Model
# I evaluated the success of each of my models by comparing its performance metrics, 
# such as the model's accuracy, against this baseline's accuracy. To get my baseline 
# value, I simply took the proportion of individuals who earn <=$50k

majority_class = adult_cleaned['income'].value_counts().idxmax()
baseline_predictions = [majority_class] * len(adult_cleaned)

baseline_accuracy = (adult_cleaned['income'] == majority_class).mean()

print(f'Baseline Accuracy: {baseline_accuracy:.2%}')

# Logistics Regression Model
# I built a logistic regression to predict income classification using the three features
# I analyzed through EDA: education-num, marital-status, and sex. 

# The logistic regression model achieved an accuracy of 81.25% on the test set. This performance 
# exceeds the baseline accuracy of always predicting an income <=$50K (75.2%).
# Therefore, the three features are promising predictors for income classfication.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

features = ['education-num', 'marital-status', 'sex'] 
X = adult_cleaned[features]
y = (adult_cleaned['income'] == '>50K').astype(int) 

# Dummying categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['marital-status', 'sex']),
        ('num', 'passthrough',['education-num'])
    ]
)

X_encoded = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.2%}')

# The permutation importance results show that education-num is the most influential 
# predictor of whether an individual earns more than $50K, followed by specific categories 
# of marital-status (such as being married or never married), and then sex. 
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['marital-status', 'sex'])
feature_names = list(cat_features) + ['education-num']
print(pd.DataFrame({'feature': feature_names, 'importance': result.importances_mean}).sort_values('importance', ascending=False))

# KNN Classifier
# I also trained a KNN model because it can capture non-linear patterns in the data and 
# doesn't make strong assumptions about feature distributions.
# My KNN classifier (n=9) has an accuracy of 80.63% on the test set. 
# While this performance is better than the baseline model, it is slightly lower than 
# the accuracy achieved by the logistic regression model (81.21%).
# In other words, the extra flexibility of KNN did not help—it may even have introduced 
# more noise or overfitting, leading to slightly worse performance.
# For instance, as KNN treats all features equally, it can underestimate the influence of 
# stronger predictors.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

# Best model and its accuracy
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best k: {grid.best_params_["n_neighbors"]}')
print(f'KNN Classifier Accuracy (GridSearch): {accuracy:.2%}')

# Decision Tree Classifier Model
# I chose Decision Tree Classifier because it can handle both numerical and categorical features 
# and offers a clear decision-making process. After experimenting with various depths, 
# I found a depth of 5 has the highest accuracy (81.77%), slightly outperforming both the logistic 
# regression and the KNN model.
# According to the feature importance scores, education-num and Married-civ-spouse are the most important 
# features. Moreover, the tree weighs fewer variables than the logistics regression
# model, focusing on the clearest distinctions in the data.
# Additionally, the tree's clear structure makes it easier to explain to a non-technical audience.
Though not complex in design, the tree worked well and provided a clear picture of the decision-making process — useful for explaining the model's output to non-technical stakeholders.
from sklearn.tree import DecisionTreeClassifier
train_scores = []
test_scores = []
depths = range(1, 20)

for d in depths:
    dtree = DecisionTreeClassifier(max_depth=d, random_state=42)
    dtree.fit(X_train, y_train)
    y_train_preds = dtree.predict(X_train)
    y_test_preds = dtree.predict(X_test)
    train_scores.append(accuracy_score(y_train, y_train_preds))
    test_scores.append(accuracy_score(y_test, y_test_preds))

plt.plot(depths, train_scores, '--o', label='train')
plt.plot(depths, test_scores, '--o', label='test')
plt.grid()
plt.legend()
plt.xticks(depths)
plt.xlabel('max tree depth')
plt.ylabel('accuracy')
plt.title('Decision Tree Depth vs. Test/Train Accuracy')

# Find the best depth
best_depth = depths[test_scores.index(max(test_scores))]
print(f'Optimal max_depth: {best_depth}, Test Accuracy: {max(test_scores):.2%}')

final_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_tree.fit(X_train, y_train)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': final_tree.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df)

# Summary of Findings
# All three of my models are better the baseline predictor, signifying their utility and significance. 
# The models ranked in terms of performance are as follows: Decision Tree Classifier, 
# KNN Classifier, and Logistics Regression.

# Key findings
# 1) Decision Tree Classifier achieved the highest test accuracy (81.77%), outperforming 
# both the Logistic Regression (81.25%) and KNN Classifier (80.63%). This is largely due to
# the model's focus on the most predictive features.

# 2) Education level (education-num) and marital status (particularly being Married-civ-spouse) 
# are the most promising predictors of income class. These features showed a strong and consistent 
# association with higher earnings, highlighting their real-world socioeconomic relevance.

# 3) While Logistic Regression accounted for various variables with balanced influence, the Decision 
# Tree focused on a few high-impact features. Meanwhile, KNN struggled slightly due to sensitivity to 
# less informative features and a lack of built-in feature weighting.

# Next Steps/Improvements
# 1) Not all learning happens at school. Beyond traditional education levels, 
# it would be interesting to account for mentorship exposure, upbringing environment, and 
# participation in training programs, certifications, or apprenticeships.
# These invisible forms of education often impact income, but aren't always reflected in 
# standard education data.

# 2) Who you know often matters as much as what you know. Including indicators of someone’s 
# professional or personal network—like industry connections, alumni groups, or community 
# involvement—might offer further insights into opportunity's influence on income

# 3) A person’s innate qualities might influence their decision-making habits, and how much 
# they earn as well. While hard to measure, surveys or self-reported preferences could help.
# Alternatively, neural activity monitoring could serve as innovative indicators of this factor.

# 4) Life events like moving, health issues, or caregiving responsibilities can disrupt career 
# progression. Adding information about personal stability or major life transitions could help 
# explain changes in income.

# By integrating these additional factors into the analysis, I would be able to refine these models 
# even more and obtain a more nuanced understanding of the multitude of factors that could influence
# income. This approach could potentially lead to more accurate predictions and actionable insights for 
# policymakers and researchers.
