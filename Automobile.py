#!/usr/bin/env python
# coding: utf-8

# **Problem Description:**
# There is an automobile company XYZ from Japan which aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
# They want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Japanese market. Essentially, the company wants to know:
# 
# Which variables are significant in predicting the price of a car?
# How well those variables describe the price of a car?
# Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the American market.
# 
# **Business Objectives:**
# You as a Data scientist are required to apply some data science techniques for the price of cars with the available independent variables. That should help the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels.

# **Data field description**
# 
# - car_ID : Unique id of each observation (Interger)		
# - symboling :Its assigned insurance risk rating, A value of +3 indicates that the auto              is risky, -3 that it is probably pretty safe.(Categorical) 		
# - CarName :Name of car company (Categorical)		
# - fueltype : Car fuel type i.e gas or diesel (Categorical)		
# - aspiration : Aspiration used in a car (Categorical)		
# - doornumber : Number of doors in a car (Categorical)		
# - carbody :	body of car (Categorical)		
# - drivewheel : type of drive wheel (Categorical)		
# - enginelocation : Location of car engine (Categorical)		
# - wheelbase : Weelbase of car (Numeric)		
# - carlength : Length of car (Numeric)		
# - carwidth : Width of car (Numeric)		
# - carheight	: height of car (Numeric)		
# - curbweight : The weight of a car without occupants or baggage. (Numeric)	
# - enginetype : Type of engine. (Categorical)		
# - cylindernumber : cylinder placed in the car (Categorical)		
# - enginesize :	Size of car (Numeric)		
# - fuelsystem : 	Fuel system of car (Categorical)		
# - boreratio : Boreratio of car (Numeric)		
# - stroke : Stroke or volume inside the engine (Numeric)		
# - compressionratio : compression ratio of car (Numeric)		
# - horsepower :			Horsepower (Numeric)		
# - peakrpm :		car peak rpm (Numeric)		
# - citympg :		Mileage in city (Numeric)		
# - highwaympg :			Mileage on highway (Numeric)		
# - price**(Dependent variable)**	:		Price of car (Numeric)		

# In[1]:


# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


cars = pd.read_excel("/Users/aaronp/Downloads/Automobile 1 -Aaron parlekar/Data Worksheet.xlsx")


# # Data Understanding and Exploration

# In[3]:


cars.head(5)


# In[4]:


cars.shape


# In[5]:


cars.info()


# In[6]:


cars.columns


# In[7]:


# Categorical variables

cars.select_dtypes(include=['object']).columns.tolist()


# In[8]:


# Numeric variables

cars.select_dtypes(exclude=['object']).columns.tolist()


# In[9]:


# Checking missing values

cars.isna().sum()


# In[10]:


cars.describe()


# In[11]:


cars['symboling'].value_counts() # symboling: -2(least risky) - +3(most risky)


# In[12]:


sns.histplot(data=cars,x='symboling',bins=6,kde=True)


# 
# **Most cars are 0,1,2**

# In[13]:


cars['aspiration'].value_counts() 

# aspiration: An (internal combustion) engine property showing 
# whether the oxygen intake is through standard (atmospheric pressure)
# or through turbocharging (pressurised oxygen intake)


# **About 82% of cars in the American market have a standard engine**
# 

# In[14]:


cars['carbody'].value_counts()


# In[15]:


sns.histplot(data=cars,x='carbody',color="yellow")


# **Most of the cars manufactured are either sedans(47%) or hatchbacks(34%)**

# In[16]:


cars['fueltype'].value_counts()


# In[17]:


cars['fuelsystem'].value_counts()


# In[18]:


cars['enginelocation'].value_counts()


# In[19]:


sns.distplot(cars['wheelbase'])
plt.show()


# In[20]:


sns.distplot(cars['horsepower'])
plt.show()


# In[21]:


sns.distplot(cars['peakrpm'])
plt.show()


# In[22]:


sns.distplot(cars['citympg'])
plt.show()


# In[23]:


# Target variable
sns.distplot(cars['price'])
plt.show()


# In[24]:


# all numeric (float and int) variables in the dataset

cars_numeric = cars.select_dtypes(include=['float64', 'int64'])
cars_numeric.head()


# In[25]:


# dropping symboling and car_ID 

cars_numeric = cars_numeric.drop(['symboling', 'car_ID'], axis=1)
cars_numeric.head()


# # Understand each col
# # how to read distplot. Also use the paramters from the yt video in ur analysis
# # skewness? dummy variables?
# #

# In[26]:


#  Correlations on a heatmap

plt.figure(figsize=(16,8))

sns.heatmap(cars_numeric.corr(), annot=True)
plt.show()


# # Data Cleaning

# In[27]:


cars.rename(columns={'price':'price($)'},inplace=True)
cars


# In[28]:


# we need to extract the company name from the column CarName.

cars['CarName'][:15]


# In[29]:


# Extracting only the car name

carnames = cars['CarName'].apply(lambda x: x.split(" ")[0])
carnames[:20]

# The lambda keyword is used to create small anonymous functions.
# A lambda function can take any number of arguments, but can only have one expression.


# In[30]:


# New column 
cars['CarCompany'] = cars['CarName'].apply(lambda x: x.split(" ")[0])


# In[31]:


cars['CarCompany'].value_counts()


# In[32]:


# Correcting and replacing the mispelled CarCompany names

# toyota
cars.loc[cars['CarCompany'] == "toyouta", 'CarCompany'] = 'toyota'

# volkswagen
cars.loc[(cars['CarCompany'] == "vw") | 
         (cars['CarCompany'] == "vokswagen")
         , 'CarCompany'] = 'volkswagen'

# mazda
cars.loc[cars['CarCompany'] == "maxda", 'CarCompany'] = 'mazda'

# porsche
cars.loc[cars['CarCompany'] == "porcshce", 'CarCompany'] = 'porsche'

# nissan
cars.loc[cars['CarCompany'] == "Nissan", 'CarCompany'] = 'nissan'


# In[33]:


cars.head(5)


# In[34]:


cars['CarCompany'].value_counts()


# In[35]:


cars.drop('CarName', axis=1)


# In[36]:


# converting symboling to categorical

cars['symboling']=cars['symboling'].astype('object')


# In[37]:


cars.info()


# In[38]:


# Using IQR to check the presence of outliers.

for col in cars.select_dtypes(include=np.number):
    counter=0
    q1= cars[col].quantile(0.25)
    q3= cars[col].quantile(0.75)
    iqr= q3-q1
    lower_fence= iqr-1.5*q1
    upper_fence= iqr+1.5*q3
    for i, row in cars.iterrows():
        if cars.at[i,col]<lower_fence and cars.at[i,col]>upper_fence:
            counter+=1
    print(col,counter)    


# **No outliers in the dataset**

# #  Data Preparation
# 

# In[76]:


# Splitting independent variables(X) and target variable(y)

X = cars.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'CarCompany']]

y = cars['price($)']


# In[77]:


# creating dummy variables for categorical variables

# subset all categorical variables
cars_categorical = X.select_dtypes(include=['object'])
cars_categorical.head()


# In[78]:


# convert into dummies
cars_dummies = pd.get_dummies(cars_categorical, drop_first=True)
cars_dummies.head()


# In[79]:


# drop categorical variables 
X = X.drop(list(cars_categorical.columns), axis=1)

# concat dummy variables with X
X = pd.concat([X, cars_dummies], axis=1)


# In[80]:


# scaling the features
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns

# train and test split
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                    train_size=0.8,
#                                                    test_size = 0.2, random_state=100)


# In[81]:


# Train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                  train_size=0.8,
                                                    test_size = 0.2, random_state=100)


# # Model Building

# In[82]:


# Building the first model with all the features

# instantiate
from sklearn.linear_model import LinearRegression
lrm = LinearRegression()

# fit
lrm.fit(X_train, y_train)


# In[83]:


# print coefficients and intercept
print(lrm.coef_)
print(lrm.intercept_)


# In[84]:


# predict 
y_pred = lrm.predict(X_test)

# metrics
from sklearn.metrics import r2_score

print(r2_score(y_true=y_test, y_pred=y_pred))


# we are getting approx. 83% r-squared with all the variables. Let's see how much we can get with lesser features.

# **Model Building Using RFE** (recursive feature elimination) to select features

# In[85]:


# RFE with 15 features

from sklearn.feature_selection import RFE

# RFE with 15 features
lrm = LinearRegression()
rfe_15 = RFE(estimator=lrm, n_features_to_select=15)

# fit with 15 features
rfe_15.fit(X_train, y_train)

# Printing the boolean results
print(rfe_15.support_)           
print(rfe_15.ranking_)  


# In[86]:


# making predictions using rfe model
y_pred = rfe_15.predict(X_test)

# r-squared
print(r2_score(y_test, y_pred))


# In[87]:


# RFE with 10 features
from sklearn.feature_selection import RFE

# RFE with 10 features
lrm = LinearRegression()
rfe_10 = RFE(estimator=lrm, n_features_to_select=10)

# fit with features
rfe_10.fit(X_train, y_train)

# predict
y_pred = rfe_10.predict(X_test)

# r-squared
print(r2_score(y_test, y_pred))


# Note that RFE with 6 features is giving about 88% r-squared, compared to 89% with 15 features. Should we then choose more features for slightly better performance?
# 
# A better metric to look at is adjusted r-squared, which penalises a model for having more features, and thus weighs both the goodness of fit and model complexity. Let's use statsmodels library for this.

# # Evaluation

# In[88]:


# import statsmodels
import statsmodels.api as sm  

# subset the features selected by rfe_15
col_15 = X_train.columns[rfe_15.support_]

# subsetting training data for 15 selected columns
X_train_rfe_15 = X_train[col_15]

# add a constant to the model
X_train_rfe_15 = sm.add_constant(X_train_rfe_15)
X_train_rfe_15.head()


# In[89]:


# fitting the model with 15 variables
lrm_15 = sm.OLS(y_train, X_train_rfe_15).fit()   # OLS(Ordinary Least Square)
print(lrm_15.summary())


# In[90]:


# making predictions using rfe_15 sm model
X_test_rfe_15 = X_test[col_15]


# # Adding a constant variable 
X_test_rfe_15 = sm.add_constant(X_test_rfe_15, has_constant='add')
X_test_rfe_15.info()


# # Making predictions
y_pred = lrm_15.predict(X_test_rfe_15)


# In[91]:


# r-squared
r2_score(y_test, y_pred)


# Thus, the test r-squared of model with 15 features is about 85.4%, while training is about 93%. Let's compare the same for the model with 10 features

# In[93]:


# subset the features selected by rfe_10
col_10 = X_train.columns[rfe_10.support_]

# subsetting training data for 10 selected columns
X_train_rfe_10 = X_train[col_10]

# add a constant to the model
X_train_rfe_10 = sm.add_constant(X_train_rfe_10)


# fitting the model with 10 variables
lrm_10 = sm.OLS(y_train, X_train_rfe_10).fit()   
print(lrm_10.summary())


# In[94]:


# making predictions using rfe_10 sm model
X_test_rfe_10 = X_test[col_10]


# Adding a constant  
X_test_rfe_10 = sm.add_constant(X_test_rfe_10, has_constant='add')
X_test_rfe_10.info()


# # Making predictions
y_pred = lrm_10.predict(X_test_rfe_10)


# In[95]:


# r2_score for 10 variables
r2_score(y_test, y_pred)


# Thus, for the model with 6 variables, the r-squared on training and test data is about 93.3% and 88.1% respectively. The adjusted r-squared is about 91.1%.

# **Choosing the optimal number of features**
# 
# Now, we have seen that the adjusted r-squared varies from about 93.3% to 91.1% as we go from 15 to 6 features, one way to choose the optimal number of features is to make a plot between n_features and adjusted r-squared, and then choose the value of n_features.

# In[104]:


n_features_list = list(range(5, 20))
adjusted_r2 = []
r2 = []
test_r2 = []

for n_features in range(5, 20):

    # RFE with n features
    lrm = LinearRegression()

    # specify number of features
    rfe_n = RFE(estimator=lrm,n_features_to_select=n_features)

    # fit with n features
    rfe_n.fit(X_train, y_train)

    # subset the features selected by rfe_n
    col_n = X_train.columns[rfe_n.support_]

    # subsetting training data for n selected columns
    X_train_rfe_n = X_train[col_n]

    # add a constant to the model
    X_train_rfe_n = sm.add_constant(X_train_rfe_n)


    # fitting the model with 6 variables
    lrm_n = sm.OLS(y_train, X_train_rfe_n).fit()
    adjusted_r2.append(lrm_n.rsquared_adj)
    r2.append(lrm_n.rsquared)
    
    
    # making predictions using rfe_15 sm model
    X_test_rfe_n = X_test[col_n]


    # # Adding a constant variable 
    X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')



    # # Making predictions
    y_pred = lrm_n.predict(X_test_rfe_n)
    
    test_r2.append(r2_score(y_test, y_pred))


# In[106]:


# plotting adjusted_r2 against n_features
plt.figure(figsize=(10, 8))
plt.plot(n_features_list, adjusted_r2, label="adjusted_r2")
plt.plot(n_features_list, r2, label="train_r2")
plt.plot(n_features_list, test_r2, label="test_r2")
plt.legend(loc='upper left')
plt.grid()
plt.show()


# Based on the plot, we can choose the number of features considering the r2_score we are looking for. Note that there are a few caveats in this approach, and there are more sopisticated techniques to choose the optimal number of features:
# 
# - Cross-validation: In this case, we have considered only one train-test split of the dataset; the values of r-squared and adjusted r-squared will vary with train-test split. Thus, cross-validation is a more commonly used technique (you divide the data into multiple train-test splits into 'folds', and then compute average metrics such as r-squared across the 'folds'
# 
# - The values of r-squared and adjusted r-squared are computed based on the training set, though we must always look at metrics computed on the test set. For e.g. in this case, the test r2 actually goes down with increasing n - this phenomenon is called 'overfitting', where the performance on training set is good because the model has in some way 'memorised' the dataset, and thus the performance on test set is worse.
# 
# Thus, we can choose anything between 5 and 12 features, since beyond 12, the test r2 goes down; and at lesser than 5, the r2_score is less.
# 
# In fact, the test_r2 score doesn't increase much anyway from n=8 to n=12. It is thus wiser to choose a simpler model, and so let's choose n=8.

# # Final Model
# 
# Building the final model using 8 features

# In[110]:


# RFE with n features
lm = LinearRegression()

n_features = 8

# specify number of features
rfe_n = RFE(estimator=lm,n_features_to_select= n_features)

# fit with n features
rfe_n.fit(X_train, y_train)

# subset the features selected by rfe_6
col_n = X_train.columns[rfe_n.support_]

# subsetting training data for 6 selected columns
X_train_rfe_n = X_train[col_n]

# add a constant to the model
X_train_rfe_n = sm.add_constant(X_train_rfe_n)


# fitting the model with 6 variables
lm_n = sm.OLS(y_train, X_train_rfe_n).fit()
adjusted_r2.append(lm_n.rsquared_adj)
r2.append(lm_n.rsquared)


# making predictions using rfe_15 sm model
X_test_rfe_n = X_test[col_n]


# # Adding a constant variable 
X_test_rfe_n = sm.add_constant(X_test_rfe_n, has_constant='add')



# # Making predictions
y_pred = lm_n.predict(X_test_rfe_n)

test_r2.append(r2_score(y_test, y_pred))


# In[111]:


lm_n.summary()


# In[112]:


# results 
r2_score(y_test, y_pred)


# **Final Model Evaluation**
# 
# Let's now evaluate the model in terms of its assumptions. We should test that:
# 
# - The error terms are normally distributed with mean approximately 0
# - There is little correlation between the predictors
# - Homoscedasticity, i.e. the 'spread' or 'variance' of the error term (y_true-y_pred) is constant

# In[114]:


# Error terms
c = [i for i in range(len(y_pred))]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('ytest-ypred', fontsize=16)                # Y-label
plt.grid()
plt.show()


# In[116]:


# Plotting the error terms to understand the distribution.
fig = plt.figure()
sns.distplot((y_test-y_pred),bins=50,color='r')
fig.suptitle('Error Terms', fontsize=20)                  # Plot heading 
plt.xlabel('y_test-y_pred', fontsize=18)                  # X-label
plt.ylabel('Index', fontsize=16)                          # Y-label
plt.show()


# In[117]:


# mean
np.mean(y_test-y_pred)


# Now it may look like that the mean is not 0, though compared to the scale of 'price', 207 is not such a big number (see distribution below).

# In[124]:


sns.displot(cars['price($)'],bins=20,kde=True)
plt.show()


# In[126]:


# multicollinearity
predictors = ['carwidth', 'curbweight', 'enginesize', 'boreratio',
             'enginelocation_rear','enginetype_rotor', 'CarCompany_bmw', 'CarCompany_porsche']

cors = X.loc[:, list(predictors)].corr()
sns.heatmap(cors, annot=True)
plt.show()


# In[ ]:




