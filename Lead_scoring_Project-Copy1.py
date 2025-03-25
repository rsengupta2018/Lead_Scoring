#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the warnings.

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#importing required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve


# In[3]:


#importing the dataset from csv

df= pd.read_csv(r'C:\Users\tuli\Desktop\data analysis\Upgrad\Lead_Scoring_assignment\Lead Scoring Assignment\Leads.csv')
df


# **Step 2: Inspecting the Dataframe**

# In[4]:


#checking for null values 
df.info()


# In[5]:


df.shape


# In[6]:


#To check out info of each column
df.info()


# **1. The presence of few categorical values are visible for which we have to create dummy variables.**
# **2. Null values are also present in the dataset which has to be treated either by dropping or by computation**

# In[7]:


df.isnull().sum()


# In[8]:


#to find the percentage of null value 
((df.isnull().sum())/df.shape[0])*100


# In[9]:


#Inspecting the lead quality column 
df['Lead Quality'].value_counts()


# **Exploratory Data Analysis**

# In[10]:


#filling the missing value for Lead quality with unknown 
df['Lead Quality']=df['Lead Quality'].fillna('unknown')


# In[11]:


df['Asymmetrique Activity Index'].value_counts()


# In[12]:


#Mapping the three categories to numeric value
df['Asymmetrique Activity Index']=df['Asymmetrique Activity Index'].map({'02.Medium ':2 ,'01.High ':1,'03.Low ':3})


# In[13]:


#filling the missing value for Asymmetrique Activity Index with na 
df['Asymmetrique Activity Index']=df['Asymmetrique Activity Index'].fillna(4)


# In[14]:


df['Asymmetrique Activity Index'].isnull().sum()


# In[15]:


# Dropping all the columns with >=40%

cols=df.columns

for i in cols:
    if((100*(df[i].isnull().sum()/len(df.index))) >= 35):
        df.drop(i, axis=1, inplace = True)


# In[16]:


#checking percentage of null values in each column after dropping columns with more than 40% missing values

round((df.isnull().sum()/len(df.index)*100), 2)


# In[17]:


#Columns remaining after dropping the columns with >=45% null values 
print(cols)
print(df.columns)


# In[18]:


#columns remaining after dropping. we have dropped 4 columns 
df.shape


# In[19]:


df.drop(['Prospect ID'],axis=1)


# In[20]:


#get the list of object cols. and convert to numerical 
objectcols=df.select_dtypes(include='object').columns
objectcols


# In[21]:


# Visualzing variables for imbalancing
fig, axs = plt.subplots(3,4,figsize = (20,12))
sns.countplot(x = "Search", hue = "Converted", data = df, ax = axs[0,0],palette = 'Pastel1')
sns.countplot(x = "Magazine", hue = "Converted", data = df, ax = axs[0,1],palette = 'Pastel1')
sns.countplot(x = "Newspaper Article", hue = "Converted", data = df, ax = axs[0,2],palette = 'Pastel1')
sns.countplot(x = "X Education Forums", hue = "Converted", data = df, ax = axs[0,3],palette = 'Pastel1')
sns.countplot(x = "Newspaper", hue = "Converted", data = df, ax = axs[1,0],palette = 'Pastel1')
sns.countplot(x = "Digital Advertisement", hue = "Converted", data = df, ax = axs[1,1],palette = 'Pastel1')
sns.countplot(x = "Through Recommendations", hue = "Converted", data = df, ax = axs[1,2],palette = 'Pastel1')
sns.countplot(x = "Receive More Updates About Our Courses", hue = "Converted", data = df, ax = axs[1,3],palette = 'Pastel1')
sns.countplot(x = "Update me on Supply Chain Content", hue = "Converted", data =df, ax = axs[2,0],palette = 'Pastel1')
sns.countplot(x = "Get updates on DM Content", hue = "Converted", data = df, ax = axs[2,1],palette = 'Pastel1')
sns.countplot(x = "I agree to pay the amount through cheque", hue = "Converted", data = df, ax = axs[2,2],palette = 'Pastel1')
sns.countplot(x = "A free copy of Mastering The Interview", hue = "Converted", data = df, ax = axs[2,3],palette ='Pastel1')
plt.show()


# **Inference**
# #Magazine , receive more updates about our cources, update me on supply chain content, get updates on dm content, I agree to pay the amount through cheque has no conversion rate 
# #A free copy of Mastering the interview has high data instability

# In[22]:


# A list of columns to be dropped as they are either redunt or have no conversion 

cols_to_drop=(['Magazine','Newspaper Article','X Education Forums',
                'Through Recommendations','Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content',
                 'Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview'])


# In[23]:


#checking value counts of Lead Source column

df['Lead Source'].value_counts(dropna=False)


# In[24]:


#replacing Nan Value with Google as it has the heighest value content 
df['Lead Source'] = df['Lead Source'].replace(np.nan,'Google')

#'Lead Source' is having same label name 'Google' but in different format i.e 'google', So converting google to Google
df['Lead Source'] = df['Lead Source'].replace('google','Google')


# In[25]:


#combining low frequency values to Others

df['Lead Source'] = df['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM','Live Chat','Social Media'] ,'Others')


# In[26]:


plt.figure(figsize=(15,5))

# Sorting Lead Source by total count in descending order
sorted_order = df['Lead Source'].value_counts().index

# Using a different palette (e.g., "coolwarm")
df1 = sns.countplot(x='Lead Source', hue='Converted', data=df, palette="coolwarm", order=sorted_order)

# Rotate x-axis labels
df1.set_xticklabels(df1.get_xticklabels(), rotation=45)

plt.show()


# #Google has the maximum generated leads 
# 
# #Conversion rate of Reference leads and Welinkgak Website leads is very high.

# In[27]:


#For country column the missing values are very high putting the missing values as na
df['Country'] = df['Country'].replace(np.nan,'not provided')


# In[28]:


# Visualizing Country variable after imputation
plt.figure(figsize=(15,5))
df1=sns.countplot(x= 'Country', hue='Converted' , data =df , palette = 'coolwarm')
df1.set_xticklabels(df1.get_xticklabels(),rotation=90)
plt.show()


# #Majority of the leads are from India or the contry name is not provided. Hence we will drop this variable as there is no relevent data provide 
# 

# In[29]:


cols_to_drop.append('Country')


# In[30]:


#checking value counts of 'What is your current occupation' column
df['What is your current occupation'].value_counts(dropna=False)


# In[31]:


#Creating new category 'Not provided' for NaN

df['What is your current occupation'] = df['What is your current occupation'].replace(np.nan, 'Not provided')


# In[32]:


#visualizing count of Variable based on Converted value
sorted_order = df['What is your current occupation'].value_counts().index

df1=sns.countplot(x='What is your current occupation', hue='Converted' , data = df , palette = 'coolwarm')
df1.set_xticklabels(df1.get_xticklabels(),rotation=45)
plt.show()


# #Maximum leads generated are unemployed and their conversion rate is more than 50%.
# #Conversion rate of working professionals is very high.

# In[33]:


df['What matters most to you in choosing a course'].value_counts(dropna=False)


# #Better career prospect is the highest hence we will map the NaN value to Better Career Prospects 

# In[34]:


df['What matters most to you in choosing a course'] = df['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[35]:


#visualizing count of Variable based on Converted value
sorted_order = df['What is your current occupation'].value_counts().index

df1=sns.countplot(x= 'What matters most to you in choosing a course', hue='Converted' , data = df , palette = 'coolwarm')
df1.set_xticklabels(df1.get_xticklabels(),rotation=45)
plt.show()


# #We can drop this column as well as there as it is not adding and important inference to our calculation. We can directly understant that Better career prospect is the reason why people are opting for the course

# In[36]:


cols_to_drop.append('What matters most to you in choosing a course')
cols_to_drop


# In[37]:


df['Last Activity'].value_counts(dropna=False)


# In[38]:


#replacing Nan Values with "Email Opened" as it has the heighest frequency 

df['Last Activity'] = df['Last Activity'].replace(np.nan,'Email Opened')


# In[39]:


#clubing values with low frequency to Others
df['Last Activity'] = df['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                       'Had a Phone Conversation',
                                                       'Approached upfront',
                                                        'View in browser link Clicked',
                                                        'Email Marked Spam',
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[40]:


#visualizing count of Last Activity Variable

plt.figure(figsize=(15,5))
df1=sns.countplot(x='Last Activity', hue='Converted' , data = df , palette = 'coolwarm')
df1.set_xticklabels(df1.get_xticklabels(),rotation=90)
plt.show()


# #The maximum leads are generated having last activity as 'Email opened. However, conversion rate of email opened is not too good.
# #SMS sent as last acitivity has high conversion rate but it is the second heighest last activity 

# In[41]:


#Checking for lead origine 
df1=sns.countplot(x='Lead Origin', hue='Converted' , data = df , palette = 'coolwarm')
df1.set_xticklabels(df1.get_xticklabels(),rotation=90)
plt.show()


# In[42]:



fig, axs = plt.subplots(1,2,figsize = (15,7.5))
sns.countplot(x = "Do Not Email", hue = "Converted", data = df, ax = axs[0],palette = 'coolwarm')
sns.countplot(x = "Do Not Call", hue = "Converted", data = df, ax = axs[1],palette = 'coolwarm')
plt.show()


# In[43]:


# Adding 'Do Not Call' to the cols_to_drop List
cols_to_drop.append('Do Not Call')

#checking updated list for columns to be dropped
cols_to_drop


# In[44]:


#checking value counts of last Notable Activity
df['Last Notable Activity'].value_counts()


# In[45]:


#clubbing lower frequency values

df['Last Notable Activity'] = df['Last Notable Activity'].replace(['Had a Phone Conversation',
                                                                       'Email Marked Spam',
                                                                         'Unreachable',
                                                                         'Unsubscribed',
                                                                         'Email Bounced',
                                                                       'Resubscribed to emails',
                                                                       'View in browser link Clicked',
                                                                       'Approached upfront',
                                                                       'Form Submitted on Website',
                                                                       'Email Received'],'Others')


# In[46]:


#visualizing count of Variable based on Converted value

plt.figure(figsize = (14,5))
ax1=sns.countplot(x = "Last Notable Activity", hue = "Converted", data = df , palette = 'Set2')
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
plt.show()


# In[47]:


# Append 'Last Notable Activity'to the cols_to_drop List as this is a sales team generated data
cols_to_drop.append('Last Notable Activity')


# In[48]:


# checking final list of columns to be dropped
cols_to_drop


# In[49]:


#dropping columns
df = df.drop(cols_to_drop,axis=1)

#checking info of dataset for remaining columns
df.info()


# In[50]:


df['Lead Source'].value_counts(dropna=False)


# **Statistical Analysis**

# In[51]:


# Finding out conversion rate
Converted = round((sum(df['Converted'])/len(df['Converted'].index))*100,2)
Converted


# #The total conversion rate is 38.54%. 'Converted' is the target variable

# In[52]:


#checking the correlation between the numerical variables only
# Size of the figure
plt.figure(figsize=(10, 8))

# Heatmap with red color (for numeric columns only)
sns.heatmap(df.select_dtypes(include=['number']).corr(), cmap="Reds", annot=True)

# Show plot
plt.show()


# #The top 5 variables that has the heighest correlation with converted (target variable) are Total Time Spent on Website(0.36), Lead Origin(0.22),Lead Source(0.19),Total Visits(0.03),Page Views Per Visit(-0.0033)

# In[53]:


#Visualising Total time spent on website w.r.t converted as it has the heighest correlation 

sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = df)
plt.show()


# #Leads that converted (1) spent more time on the website. Non-converted leads (0) spent less time on the website.There is also presence of outliers. Some users spent an extreme amount of time on the website but didn’t convert.Some converted users also spent significantly longer than others.

# In[54]:


#Visualising Total time spent on website w.r.t converted as it has the heighest correlation 

sns.boxplot(y = 'Lead Origin', x = 'Converted', data = df)
plt.show()


# #The distribution of Lead Origin is almost the same for both converted and non-converted leads.This suggests that Lead Origin alone is not a strong differentiator between conversions and non-conversions.

# In[55]:


#Visualising Total time spent on website w.r.t converted as it has the heighest correlation 

sns.boxplot(y = 'Lead Source', x = 'Converted', data = df)
plt.show()


# Some sources could be performing slightly better, but the difference is not significant. More detailed analysis
# is needed.

# In[56]:


#visualizing spread of variable 'Page Views Per Visit'
sns.boxplot(y =df['Page Views Per Visit'])
plt.show()


# There is presence of outlier and hence we will do outlier treatment 

# In[57]:


#Outlier Treatment: capping the outliers to 95% value for analysis
percentiles = df['Page Views Per Visit'].quantile([0.05,0.95]).values
df['Page Views Per Visit'][df['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
df['Page Views Per Visit'][df['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]

#visualizing variable after outlier treatment
sns.boxplot(y=df['Page Views Per Visit'])
plt.show()


# In[58]:


#visualizing 'Page Views Per Visit' w.r.t Target variable 'Converted'
sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = df)
plt.show()


# For the converted and not converted leads, the median is almost same. Nothing conclusive can be said on the basis of Page Views Per Visit.

# In[59]:


# List of variables to map

varlist =  ['Do Not Email']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
df[varlist] = df[varlist].apply(binary_map)


# In[60]:


df.head()


# In[61]:


#List of categorical columns foy creating dummy

cat_cols= df.select_dtypes(include=['object']).columns
cat_cols


# In[62]:




# Create dummy variables
dummy = pd.get_dummies(df['Lead Source'], prefix='Lead Source')

# Check available columns before dropping
print("Available columns:", dummy.columns)

# Drop only if the column exists
if 'Lead Source_Others' in dummy.columns:
    dummy = dummy.drop(['Lead Source_Others'], axis=1)

# Concatenate back to the DataFrame
df = pd.concat([df, dummy], axis=1)

# Display first few rows
print(df.head())


# In[63]:


#dropping the original columns after dummy variable creation

df.drop(cat_cols,axis= 1,inplace = True)


# In[64]:


df.head()


# In[66]:


# Defining variables to X
X=df.drop('Converted', axis=1)

#checking head of X
X.head()


# In[69]:


# Defining response variable to y
y = df['Converted']

#checking head of y
y.head()


# In[70]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[71]:


#importing library for feature scaling
from sklearn.preprocessing import StandardScaler


# In[72]:


#scaling of features
scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

#checking X-train dataset after scaling
X_train.head()


# In[73]:


## Checking the conversion rate from 'converted' column as it denotes the target variable

(sum(y)/len(y.index))*100


# In[79]:


#removing nan and large values 
import numpy as np

print("NaN values in X_train:", np.isnan(X_train).sum())
print("Inf values in X_train:", np.isinf(X_train).sum())


# In[80]:


#Replace with Mean
X_train['TotalVisits'].fillna(X_train['TotalVisits'].mean(), inplace=True)
X_train['Page Views Per Visit'].fillna(X_train['Page Views Per Visit'].mean(), inplace=True)


# In[81]:


print(X_train.isna().sum())


# In[82]:


logreg = LogisticRegression()
rfe = RFE(logreg,n_features_to_select= 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[83]:


rfe.support_


# In[84]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[85]:


#list of RFE supported columns
col = X_train.columns[rfe.support_]
col


# In[86]:


X_train.columns[~rfe.support_]


# #Model Building

# In[87]:


X_train_sm = sm.add_constant(pd.get_dummies(X_train[col].astype(float), drop_first=True))
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[171]:


X_train_sm=sm.add_constant(X_train)


# #dropping varibles with high p value 
# 

# In[89]:


cols_to_drop = [
    'Lead Number', 
    'Page Views Per Visit', 
    'Lead Source_Facebook', 
    'Lead Source_Google', 
    'Lead Source_Olark Chat', 
    'Lead Source_Organic Search'
]

X_train = X_train.drop(columns=cols_to_drop)

# Rerun the model
X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial()).fit()
print(logm2.summary())


# In[93]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Create a DataFrame for VIF values
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train.columns  # Features only (exclude target)
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# Print sorted VIF values
print(vif_data.sort_values(by="VIF", ascending=False))


# In[179]:


# Create an empty DataFrame for VIF and find the most important parameter 
vif = pd.DataFrame()
vif["Features"] = X_train.columns  # Assign feature names
vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif["VIF"] = vif["VIF"].round(2)
vif = vif.sort_values(by="VIF", ascending=False)
# Display the VIF DataFrame
print(vif)


# In[94]:


#for dropping highly correlated features 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF and drop features with VIF > 10
def remove_high_vif_features(X, threshold=10):
    while True:
        vif = pd.DataFrame()
        vif["Features"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif = vif.sort_values(by="VIF", ascending=False)

        # Stop if all VIFs are below threshold
        if vif["VIF"].iloc[0] < threshold:
            break
        
        # Drop the feature with highest VIF
        feature_to_drop = vif.iloc[0]["Features"]
        print(f"Dropping '{feature_to_drop}' with VIF: {vif.iloc[0]['VIF']}")
        X = X.drop(columns=[feature_to_drop])

    return X

X_train_cleaned = remove_high_vif_features(X_train)


# In[186]:


#predicting on test set 
X_test_cleaned = X_test[X_train_cleaned.columns]  # Keep only selected columns
X_test_final = sm.add_constant(X_test_cleaned)  # Ensure intercept column is present
y_pred_prob = result.predict(X_test_final)  # Get predicted probabilities
y_pred = (y_pred_prob >= 0.5).astype(int)


# In[187]:


from sklearn.metrics import accuracy_score, classification_report

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[97]:


# Get predicted probabilities
y_pred_probs = result.predict(X_train_final)  

# Convert probabilities to binary classes (threshold = 0.5)
y_pred = (y_pred_probs >= 0.5).astype(int)

