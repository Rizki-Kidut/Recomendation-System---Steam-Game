# -*- coding: utf-8 -*-
"""Submission Predictive Analytics - Heart Disesase.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TPiMPYygLJRHTSTJTf4dSRv-ny2N7GAz

#**Predictive Analytics Submission - Heart Disease Prediction on UCI Heart Disease Data**

Welcome to my Predictive Analytics Submission, Heart Disease Prediction on UCI Heart Disease Data.

In this notebook, we will delve into the Heart Disease UCI dataset, which contains various clinical and demographic features of patients to predict the presence of heart disease. By exploring this dataset, we aim to gain insights into the factors that contribute to heart disease and develop a better understanding of the data.

##About Data

The UCI Heart Disease dataset contains a collection of features that are used to predict the presence of heart disease in patients. Each row in the dataset represents a different patient, and the columns represent various attributes related to their health and heart disease status.

##Context
This is a multivariate type of dataset which means providing or involving a variety of separate mathematical or statistical variables, multivariate numerical data analysis. It is composed of 14 attributes which are age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, oldpeak — ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels and Thalassemia. This database includes 76 attributes, but all published studies relate to the use of a subset of 14 of them. The Cleveland database is the only one used by ML researchers to date. One of the major tasks on this dataset is to predict based on the given attributes of a patient that whether that particular person has heart disease or not and other is the experimental task to diagnose and find out various insights from this dataset which could help in understanding the problem more.

##Column Descriptions:
1. **id**  : Unique id for each patient
2. **age** : Age of the patient in years
3. **sex** : Gender (Male/Female)
4. **dataset** : Place of study (Cleveland, Hungary, Switzerland, VA Long Beach)
5. **cp** : chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)
6. **trestbps** : resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
7. **chol** : serum cholesterol in mg/dl
8. **fbs** : if fasting blood sugar > 120 mg/dl (True/ Flase)
9. **restecg** : resting electrocardiographic results (normal, stt abnormality, lv hypertrophy)
10. **thalach** : maximum heart rate achieved
11. **exang** : exercise-induced angina (True/ False)
12. **oldpeak** : ST depression induced by exercise relative to rest
13. **slope** : the slope of the peak exercise ST segment (upsloping, flat, downsloping)
14. **ca** : number of major vessels (0-3) colored by fluoroscopy
15. **thal** : Thalassemia (normal; fixed defect; reversible defect)
16. **num**: the predicted attribute (0: no heart disease, 1,2,3,4: stage of heart disease)

##Acknowledgements
###Creators:
1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

###Relevant Papers:
- Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64,304--310.
- David W. Aha & Dennis Kibler. "Instance-based prediction of heart-disease presence with the Cleveland database."
- Gennari, J.H., Langley, P, & Fisher, D. (1989). Models of incremental concept formation. Artificial Intelligence, 40, 11--61.

###Citation Request:
The authors of the databases have requested that any publications resulting from the use of the data include the names of the principal investigator responsible for the data collection at each institution. They would be:

1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:Robert Detrano, M.D., Ph.D.

##1. Import Dataset

I used UCI heart disease data from Kaggle. ([Kaggle UCI Heart Disesase Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)). The Original dataset is from UCI. [Original UCI Heart Disease Data](https://archive.ics.uci.edu/dataset/45/heart+disease).

Because, I'm using data from Kaggle, I downloaded the data using Kaggle API.
"""

! pip install -q kaggle

from google.colab import files

files.upload()

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d redwankarimsony/heart-disease-data

import os
import zipfile
local_zip = '/content/heart-disease-data.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

"""##2. Load the Dataset

The data is loaded into heart_data Dataframe using Pandas dataframe.
"""

import pandas as pd

heart_data = pd.read_csv('/content/heart_disease_uci.csv')
heart_data

"""##3. Dataset Overview"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

heart_data.info()

"""* There are 920 rows, means the data of 920 human being.
* There are total 16 columns in the dataset, including id, dataset (location of the patient).
* The target feature num represents the ordinal numeric severity of the heart disease ([0, 1, 2, 3, 4]).
* There are 13 features or medical parameters (excluding id and dataset), which will be used to predict the target feature num (the intensity of the heart disease).
* There are some feature that doesn't have 920 data. It's indicate that feature has some missing values.

Because column 'id' and 'dataset' don't have connection to all column or feature, I decided to drop that columns.
"""

heart_data = heart_data.drop(['id', 'dataset'], axis=1)
heart_data

heart_data.describe()

"""From this Descriptive Statistics:
1. Age:

    a. Minimal Age of patient in this data is 28

    b. Average Age of patient is 54

    c. Maximal Age of patient is 77

2. Resting Blood Preasure ('tretbps'):

    Minimal value is 0, it is indicate either there are a null value or outliers in this feature

3. Cholestrol ('chol'):

    Minimal value is 0, it is indicate either there are a null value or outliers in this feature

4. ST Depression induced by exercise ('oldpeak'):

    Minimal value is -2.6, it is indicate an outliers in this feature







"""

# Check the shape of the data
print('Number of rows in the dataset : ',heart_data.shape[0])
print('Number of columns in the dataset : ',heart_data.shape[1])

"""##4. Handling Missing Values

###4.1 Inspect Missing Values

Inspect missing values in data using isnull() function.
"""

heart_data.isnull().sum()

"""Inspect the percentage of missing values in each feature."""

round((heart_data.isnull().sum()[heart_data.isnull().sum()>0]/len(heart_data)*100),1).sort_values(ascending=False)

"""From the observation:

1. There are 10 features with missing values.
2. There are 7 features with percentage of missing values is less than 10%.
3. There are 3 features with persentage of missing values is high (30%, 50%, 60%).

Because there are a lot of missing values, it won't be good too drop the missing values.

So, this missing values is handling with Iterative Imputer using Machine Learning Model, Random Forest Classifier, and Random Forest Regressor to predict the missing values.
"""

missing_data_cols = heart_data.isnull().sum()[heart_data.isnull().sum() > 0].index.tolist()
missing_data_cols

# find only categorical columns
cat_cols = heart_data.select_dtypes(include='object').columns.tolist()
# find only numerical columns
num_cols = heart_data.select_dtypes(exclude='object').columns.tolist()

print(f'Categorical Columns: {cat_cols}')
print(f'Numerical Columns: {num_cols}')

categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']
bool_cols = ['fbs', 'exang']
numeric_cols = ['oldpeak', 'thalch', 'chol', 'trestbps', 'age']

"""##4.2 Iterative Imputer"""

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error,r2_score


def impute_categorical_missing_data(passed_col):

    heart_data_null = heart_data[heart_data[passed_col].isnull()]
    heart_data_not_null = heart_data[heart_data[passed_col].notnull()]

    X = heart_data_not_null.drop(passed_col, axis=1)
    y = heart_data_not_null[passed_col]

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    if passed_col in bool_cols:
        y = label_encoder.fit_transform(y)

    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier()

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    print("The feature '"+ passed_col+ "' has been imputed with", round((acc_score * 100), 2), "accuracy\n")

    X = heart_data_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass

    if len(heart_data_null) > 0:
        heart_data_null[passed_col] = rf_classifier.predict(X)
        if passed_col in bool_cols:
            heart_data_null[passed_col] = heart_data_null[passed_col].map({0: False, 1: True})
        else:
            pass
    else:
        pass

    heart_data_combined = pd.concat([heart_data_not_null, heart_data_null])

    return heart_data_combined[passed_col]

def impute_continuous_missing_data(passed_col):

    heart_data_null = heart_data[heart_data[passed_col].isnull()]
    heart_data_not_null = heart_data[heart_data[passed_col].notnull()]

    X = heart_data_not_null.drop(passed_col, axis=1)
    y = heart_data_not_null[passed_col]

    other_missing_cols = [col for col in missing_data_cols if col != passed_col]

    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_regressor = RandomForestRegressor()

    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    print("MAE =", mean_absolute_error(y_test, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False), "\n")
    print("R2 =", r2_score(y_test, y_pred), "\n")

    X = heart_data_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
        else:
            pass

    if len(heart_data_null) > 0:
        heart_data_null[passed_col] = rf_regressor.predict(X)
    else:
        pass

    heart_data_combined = pd.concat([heart_data_not_null, heart_data_null])

    return heart_data_combined[passed_col]

# remove warning
import warnings
warnings.filterwarnings('ignore')

# impute missing values using our functions
for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((heart_data[col].isnull().sum() / len(heart_data)) * 100, 2))+"%")
    if col in categorical_cols:
        if col == '66.4':
            col = 'chol'
        heart_data[col] = impute_categorical_missing_data(col)
    elif col in numeric_cols:
        heart_data[col] = impute_continuous_missing_data(col)
    else:
        pass

"""Result of Iterative Imputer process:

1. **trestbps (Blood Pressure):**

* Missing Percentage: 6.41%
* MAE (Mean Absolute Error): 13.41
* RMSE (Root Mean Squared Error): 17.33
* R2 (R-squared): 0.06

* Inferences:

    a. The missing values in 'trestbps' were imputed successfully.

    b.The imputed values seem to have reasonable accuracy based on the provided MAE, RMSE, and R2 metrics.

2. **chol (Serum Cholesterol):**

* Missing Percentage: 3.26%
* MAE: 63.07
* RMSE: 86.67
* R2: 0.4

* Inferences:

    a. Imputation for 'chol' appears to be successful with relatively low MAE and RMSE.

    b. The R-squared value indicates a decent level of accuracy in imputing missing values.

3. **fbs (Fasting Blood Sugar):**

* Missing Percentage: 9.78%
* Imputation Accuracy: 78.92%

* Inferences:

    a. 'fbs' has been imputed with a high accuracy of 78.92%.
    b. The imputation method seems effective for this feature.

4. **restecg (Resting Electrocardiographic Results):**

* Missing Percentage: 0.22%
* Imputation Accuracy: 61.41%

* Inferences:

    a. 'restecg' has been imputed with an accuracy of 61.41%.
    b. The imputation method appears reasonable for this feature.

5. **thalch (Maximum Heart Rate Achieved):**

* Missing Percentage: 5.98%
* MAE: 17.22
* RMSE: 21.94
* Inferences:

    Imputation for 'thalch' seems to have moderate accuracy, as indicated by MAE and RMSE.

6. **exang (Exercise-Induced Angina):**

* Missing Percentage: 5.98%
* Imputation Accuracy: 75.72%
* Inferences:

    a. 'exang' has been imputed with a high accuracy of 75.72%.
    b. The imputation method seems effective for this feature.

7. **oldpeak (ST depression induced by exercise relative to rest):**

* Missing Percentage: 6.74%
* MAE: 0.57
* RMSE: 0.79
* Inferences:

    a. The missing values in 'oldpeak' were imputed successfully.

    b.The imputed values seem to have reasonable accuracy based on the provided MAE, RMSE.

8. **slope (The slope of the peak exercise ST segment):**

* Missing Percentage: 33.59%
* Imputation Accuracy: 65.04%
* Inferences:

    'slope' has a high percentage of missing values, but the imputation accuracy is decent at 65.04%.

9. **ca (Number of major vessels (0-3) colored by fluoroscopy):**

* Missing Percentage: 66.41%
* Imputation Accuracy: 65.04%
* Inferences:

    'ca' has a high percentage of missing values, but the imputation accuracy is moderate at 61.29%.

10. **thal (Thalassemia):**

* Missing Percentage: 52.83%
* Imputation Accuracy: 73.56%
* Inferences:

    'thal' has a high percentage of missing values, but the imputation accuracy is decent at 72.41%.

Overall, the imputation process appears to have performed well for most features, considering the provided metrics. It's important to note that the success of imputation may depend on the specific characteristics of the dataset and the imputation methods used. Always validate imputation results and consider the context of the data when drawing conclusions.

##4.3 Result

After Iterative Imputer process, there are no missing values in the dataset
"""

heart_data.isnull().sum()

heart_data

"""##5. Handling Outliers

After handling with missing values, the next step is to handling outliers. Because outliers can disturb model to find correlation within the dataset.

I will use IQR method and Boxplot to check outliers in this data

###5.1 Using IQR Methode
"""

Q1 = heart_data[num_cols].quantile(0.25)
Q3 = heart_data[num_cols].quantile(0.75)
IQR = Q3 - Q1
outliers_count_specified = ((heart_data[num_cols] < (Q1 - 1.5 * IQR)) | (heart_data[num_cols] > (Q3 + 1.5 * IQR))).sum()

outliers_count_specified

"""Upon identifying outliers for the specified continuous features, we found the following:

* **trestbps :** 26 outliers
* **chol :** 185 outliers
* **thalach :** 2 outlier
* **oldpeak :** 3 outliers
* **age :** No outliers
* **num :** No Outlirs
* **ca :** 20 Outliers

###5.2 Using Boxplot
"""

sns.boxplot(x=heart_data['trestbps'])

sns.boxplot(x=heart_data['chol'])

sns.boxplot(x=heart_data['thalch'])

sns.boxplot(x=heart_data['oldpeak'])

sns.boxplot(x=heart_data['ca'])

"""###5.3 Dealing the Outliers

Because there are a lot of outliers, I decided to handling outliers one by one.

####5.3.1 Tresbps Outliers Handling
"""

heart_data['trestbps'].describe()

"""There are some zero values in the column trestbps so Blood Pressure can never be 0. Therefore trestbps with 0 value can be drop."""

# remove rows with values less than 80 in the 'trestbps' column
heart_data = heart_data[heart_data['trestbps'] >= 80]

heart_data['trestbps'].describe()

"""After that, the minimal value of trestbps is 80."""

heart_data.info()

"""Dataframe size after trestbps outlier handling is 919 data.

####5.3.2 Thalach Outliers Handling

Using information from Boxplot, in thalch column values can be start 71 so removed less than 71
"""

# remove rows with values less than 71 in the 'thalch' column
heart_data = heart_data[heart_data['thalch'] >= 71]

heart_data['thalch'].describe()

"""After handling, the minimal value is 71."""

heart_data.info()

"""Dataframe size after trestbps outlier handling is 914 data.

####5.3.3 Oldpeak Outliers Handling

There are 3 outliers in oldpeak column.
The outtliers is handled using IQR method.
"""

# remove outliers in 'oldpeak' column
Q1 = heart_data['oldpeak'].quantile(0.25)
Q3 = heart_data['oldpeak'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
heart_data = heart_data[(heart_data['oldpeak'] >= lower_bound) & (heart_data['oldpeak'] <= upper_bound)]

"""####5.3.4 Chol Outliers Handling"""

heart_data['chol'].describe()

"""There are some zero values in the column cholestrol so cholestrol can never be 0."""

# print the row from df where chol value is 0
print("zero_counts :",(heart_data['chol'] == 0).sum())
# remove this row from data
heart_data = heart_data[heart_data['chol'] != 0]

"""There are 167 zero values in chol column.

From information on Boxplot, I decided to filter chol between 126 and 400.
"""

# remove values less than 126
heart_data = heart_data[heart_data['chol'] >= 126]
# remove values greater than 400
heart_data = heart_data[heart_data['chol'] <= 400]

heart_data['chol'].describe()

"""After handling, the minimal value is 126."""

heart_data.info()

"""Dataframe size after chol outlier handling is 728 data."""

heart_data.describe()

"""This is the final Statistical data after Handling Missing Values dan Handling Outliers Values.

##6. EDA (Exploratory Data Analysis)

###6.1 Univariate Analysis

####6.1.1 Categorical Features
"""

numerical_features = ['age','trestbps','chol','thalch','oldpeak']
categorical_features = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

feature = categorical_features[0]
count = heart_data[feature].value_counts()
percent = 100*heart_data[feature].value_counts(normalize=True)

df = pd.DataFrame({'Total Samples':count, 'Percentage':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""From graphic plot, this data set is dominated by Male patients with 75,7% percentage."""

feature = categorical_features[1]
count = heart_data[feature].value_counts()
percent = 100*heart_data[feature].value_counts(normalize=True)

df = pd.DataFrame({'Total Samples':count, 'Percentage':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""48,8% patients have asymptomatic chest pain, where 22,9% have non-anginal chest pain type, 22,7% have atypical anginan chest pain type, and the rest has typial angina"""

feature = categorical_features[2]
count = heart_data[feature].value_counts()
percent = 100*heart_data[feature].value_counts(normalize=True)

df = pd.DataFrame({'Total Samples':count, 'Percentage':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Most of the patients don't have fasting blood sugar > 120 mg/dl, with 83,7% percentage."""

feature = categorical_features[3]
count = heart_data[feature].value_counts()
percent = 100*heart_data[feature].value_counts(normalize=True)

df = pd.DataFrame({'Total Samples':count, 'Percentage':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Half of the patients have normal resting electrocardiographic results, where the others 23,6% of patients have Iv hypertrophy and the rest have st-t abnormality resting electrocardiographic results."""

feature = categorical_features[4]
count = heart_data[feature].value_counts()
percent = 100*heart_data[feature].value_counts(normalize=True)

df = pd.DataFrame({'Total Samples':count, 'Percentage':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""62,1% patients don't have exercise-induced angina, where the rest have it."""

feature = categorical_features[5]
count = heart_data[feature].value_counts()
percent = 100*heart_data[feature].value_counts(normalize=True)

df = pd.DataFrame({'Total Samples':count, 'Percentage':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""The majority of patients have flat and upsloping slope of the peak exercise ST segment with 47,9% and 45,5% percentage, where the rest have downsloping slope of the peak exercise ST segment."""

feature = categorical_features[6]
count = heart_data[feature].value_counts()
percent = 100*heart_data[feature].value_counts(normalize=True)

df = pd.DataFrame({'Total Samples':count, 'Percentage':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Most of the patients have 0 of major vessels with 70,7% percentage, where the rest 19% have 1, 7,7% have 2 and 2,6% have 3 of major vessels."""

feature = categorical_features[7]
count = heart_data[feature].value_counts()
percent = 100*heart_data[feature].value_counts(normalize=True)

df = pd.DataFrame({'Total Samples':count, 'Percentage':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Most of the patients have normal and reversible defect of Thalassemia with 48,5% and 45,9% percetage, where the rest have fixed defect of Thalassemia

####6.1.2 Numerical Features
"""

data = heart_data[numerical_features]

# Plot histograms with distribution lines for multiple numerical features
fig, axes = plt.subplots(nrows=1, ncols=len(data.columns), figsize=(25, 5))

for idx, column in enumerate(data.columns):
    ax = sns.histplot(data[column], kde=True, ax=axes[idx])
    sns.kdeplot(data[column], color='black', ax=ax)
    ax.set_title(f'Histogram of {column}')

plt.show()

"""From above plot can be concluded:

1. **Age  :**There is a peak in late 50's til 60's from the age data
2. **Trest bps (Blood Pressure) :** data is concentrated around 120-140 mmHg
3. **Chol (Serum Cholesterol)  :** Most patients has Cholestrol between 200 - 300
4. **Thalch (Maximum Heart Rate Achieved)  :** Majority of patients achieve a heart rate between 125 - 175 bpm during a test.
5. **Oldpeak (ST depression induced by exercise relative to rest)  :** Most of the values are concentrated towards 0, indicating that many individuals did not experience significant ST depression during exercise

###6.2 Bivariate Analysis

For this bivariate analysis on the dataset's features with respect to the target variable:

1.  For continuous data: I am going to use bar plots to showcase the average value of each feature for the different target classes, and KDE plots to understand the distribution of each feature across the target classes. This aids in discerning how each feature varies between the two target outcomes.

2. For categorical data : To show correlation between categorical values vs target values, I am going to use Chi-square test of independence.  This statistical test assesses whether there is a significant association between two categorical variables.

####6.2.1 Numerical features vs target
"""

# Set color palette
sns.set_palette(['#ff826e', 'red'])

# Create the subplots
fig, ax = plt.subplots(len(numerical_features), 2, figsize=(15,15), gridspec_kw={'width_ratios': [1, 2]})

# Loop through each continuous feature to create barplots and kde plots
for i, col in enumerate(numerical_features):
    # Barplot showing the mean value of the feature for each target category
    graph = sns.barplot(data=heart_data, x="num", y=col, ax=ax[i,0])

    # KDE plot showing the distribution of the feature for each target category
    sns.kdeplot(data=heart_data[heart_data["num"]==0], x=col, fill=True, linewidth=2, ax=ax[i,1], label='0')
    sns.kdeplot(data=heart_data[heart_data["num"]==1], x=col, fill=True, linewidth=2, ax=ax[i,1], label='1')
    ax[i,1].set_yticks([])
    ax[i,1].legend(title='Heart Disease', loc='upper right')

    # Add mean values to the barplot
    for cont in graph.containers:
        graph.bar_label(cont, fmt='         %.3g')

# Set the title for the entire figure
plt.suptitle('Continuous Features vs Target Distribution', fontsize=22)
plt.tight_layout()
plt.show()

"""**Inferences:**


1.   **Age (age)**: The distributions show a slight shift with patients having heart disease being a bit younger on average than those without. The mean age for patients without heart disease is higher.
2.  **Resting Blood Pressure (trestbps)**: Both categories display overlapping distributions in the KDE plot, with nearly identical mean values, indicating limited differentiating power for this feature.
3. **Serum Cholesterol (chol)**: The distributions of cholesterol levels for both categories are quite close, but the mean cholesterol level for patients with heart disease is slightly lower.
4. **Maximum Heart Rate Achieved (thalach)**: There's a noticeable difference in distributions. Patients with heart disease tend to achieve a higher maximum heart rate during stress tests compared to those without.
5. **ST Depression (oldpeak)**: The ST depression induced by exercise relative to rest is notably lower for patients with heart disease. Their distribution peaks near zero, whereas the non-disease category has a wider spread.
-----------
Based on the visual difference in distributions and mean values, M**aximum Heart Rate (thalach)** seems to have the most impact on the heart disease status, followed by **ST Depression (oldpeak)** and **Age (age)**.

####6.2.2 Correlation Matrix for Numeric Features
"""

plt.figure(figsize=(10, 8))
correlation_matrix = heart_data.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix for Numeric Features ", size=20)

"""**Inference:**

From correlation matrix, **ca (Number of major vessels (0-3) colored by fluoroscopy)** is the first feature that has strong correlation with target value, with **oldpeak (ST Depression)** in second place and **age** in the third place.

####6.2.3 Categorical features vs Target
"""

from scipy.stats import chi2_contingency

for cat_feature in categorical_features:
    observed = pd.crosstab(heart_data[cat_feature], heart_data['num'])

    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(observed)

    print("Chi-square statistic for {}: {}".format(cat_feature, chi2))
    print("p-value for {}: {}".format(cat_feature, p))

    # Visualize the observed counts
    sns.heatmap(observed, annot=True, cmap='Blues', fmt='d')
    plt.title('Observed Counts for {}'.format(cat_feature))
    plt.xlabel('Target')
    plt.ylabel(cat_feature)
    plt.show()

"""**Inferences:**
1. **Sex :** There are 170 Men patients with level 1 heart dissease, and 20 Men patients has level 4 heart dissease, while most of Men patients doesn't have heart dissease.
2. **Cp :** Most of patients with asymptomatic chest pain have heart dissease with 141 patients with level 1, 53 patients with level 2, 51 patients with level 3, and 17 patients with level 4 heart dissease. While most of patients with atypical angina don't have heart dissease.
3. **Fbs :** Majority of patients without fasting blood sugar > 120 mg/dl, don't have heart dissease,but most of patients with fasting blood sugar > 120 mg/dl, have heart dissease
4. **Restecg (Resting Electrocardiographic Results) :** Most of patients with normal restecg don't have heart dissease while majority patients with IV Hyperthropy and st-abnormality have heart dissease
5. **Exang (Exercise-Induced Angina) :** Majority of patients without excercise-induced angina don't have heart dissease  while most of patients with it have level 1 heart dissese (116 patients)
6. **Slope (The slope of the peak exercise ST segment) :** Majority patients with slope flat have heart dissease from level 1 (142 patients), level 2 & level 3 ( both 44 patients) and level 4 (11 patients), while most of patient with upslopping don't have heart dissease
7. **Ca (Number of major vessels (0-3) colored by fluoroscopy) :**
Majority of patient with 0 major vessels don't have heart dissease, while the most patients with 1,2,3 level have heart dissesase
8. **Thal (Thalassemia) :** Most of patients with reversible defect have heart dissease, while patients with normal thal don't have heart dissease

##7. Machine Learning Model

###7.1. Encoded Categorical Features

Encodinge Categorical Features using Label Encoder function
"""

# Identify the unique values in each categorical column.
for col in heart_data.columns:
    if heart_data[col].dtype == 'object' or heart_data[col].dtype == 'category':
        print(col, ":", heart_data[col].unique(), '\n')

categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg','fbs', 'cp', 'sex', 'num']

# Encode the categorical columns using label encoding.
label_encoder = LabelEncoder()

for col in heart_data.columns:
    if heart_data[col].dtype == 'object' or heart_data[col].dtype == 'category':
        heart_data[col] = label_encoder.fit_transform(heart_data[col])

heart_data

"""###7.2. Split Data

Data then split into train and test data using "train_test_split" function with ratio 20% data test.

"""

X = heart_data.drop(["num"],axis =1)
y = heart_data["num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""###7.3. Model

In this study I would like to use 5 models and select the best model based on accuracy.

Models:
1. Random Forest Calssifier
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Ada Boost, and
5. XG Boost

Models train using GridSearchCV, as Hyperparameter tuning
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define a list of tuples containing different classifiers and their hyperparameters
models = {
        'Random Forest': {
        'model'  : RandomForestClassifier(random_state=42),
        'params' :
        {'model__n_estimators': [50, 100, 200],
         'model__max_depth': [None, 10, 20]}
        },
        'K-Nearest Neighbors': {
        'model'  :KNeighborsClassifier(),
        'params' :
        {'model__n_neighbors': [3, 5]},
        },
        'GaussianNB': {
        'model'  : GaussianNB(),
        'params' :
        {},
        },
        'Ada Boost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.05, 0.1, 0.5]
        },
        },
        'XG Boost': {
        'model': XGBClassifier(random_state=42),
        'params':
        {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.05, 0.1, 0.5]
        },
        },
}

# print heading - for display purposes only
def heading(heading):
    print('-' * 50)
    print(heading)
    print('-' * 50)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Initialize a list to store model performance metrics
# model_scores = []
# # Initialize variables to keep track of the best classification model
# # Start with the worst possible accuracy
# best_accuracy = -float('inf')
# best_classifier = None
# 
# # Iterate over the configured models
# for name, model in models.items():
#     # Set up a pipeline with data scaling and the current model
#     pipeline = Pipeline([
#         ('scaler', QuantileTransformer(random_state=42, output_distribution='normal')),
#         ('model', model['model'])
#     ])
# 
#     # Create a GridSearchCV object to tune model hyperparameters
#     grid_search = GridSearchCV(
#         estimator=pipeline,
#         param_grid=model['params'],
#         cv=5,  # Number of cross-validation folds
#     )
# 
#     # Fit the GridSearchCV object to the training data
#     grid_search.fit(X_train, y_train)
#     # Predict on the test set using the best found model
#     y_pred = grid_search.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
# 
#     # Append performance metrics for the current model to the list
#     model_scores.append({
#         'Model': name,
#         'accuracy': accuracy,
#         'precision': precision_score(y_test, y_pred, average='weighted'),
#         'recall': recall_score(y_test, y_pred, average='weighted'),
#         'f1': f1_score(y_test, y_pred, average='weighted'),
#     })
# 
#     # Initialize variables to keep track of the best classifier model
#     # Check if this model has the best R2 score so far
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_classifier = grid_search.best_estimator_
# 
# # Sort the models based on the Model name (alphabetically)
# sorted_models = sorted(model_scores, key=lambda x: x['Model'], reverse=False)
# # Convert the sorted model performance list to a DataFrame for display
# metrics = pd.DataFrame(sorted_models)
# # Identify the best performing model based on the accuracy score
# best_class_model = max(sorted_models, key=lambda x: x['accuracy'])
# 
# # Use a custom function 'heading' to display the heading
# heading("Multiclass Classifier Models Performance")
# # Display the metrics DataFrame with rounded values for readability
# metrics.round(2)
#

# Print the best model's performance metrics
heading("BEST MULTICLASS CLASSIFIER MODEL PERFORMANCE")

print(f"Model: {best_class_model['Model']}")
print(f"Accuracy: {best_class_model['accuracy']:.2f}")
print(f"Precision: {best_class_model['precision']:.2f}")
print(f"Recall: {best_class_model['recall']:.2f}")
print(f"F1: {best_class_model['f1']:.2f}")

"""The best model for this data is Random Forest Classifier with 76% accuracy"""

# Make bar plots for the model performance metrics
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
sns.barplot(data=metrics, x='Model', y='accuracy', color='#ff826e')
plt.xticks(rotation=90)
plt.title('Accuracy')
plt.subplot(1, 4, 2)
sns.barplot(data=metrics, x='Model', y='precision', color='#ff826e')
plt.xticks(rotation=90)
plt.title('Precision')
plt.subplot(1, 4, 3)
sns.barplot(data=metrics, x='Model', y='recall', color='#ff826e')
plt.xticks(rotation=90)
plt.title('Recall')
plt.subplot(1, 4, 4)
sns.barplot(data=metrics, x='Model', y='f1', color='#ff826e')
plt.xticks(rotation=90)
plt.title('F1')
plt.show()