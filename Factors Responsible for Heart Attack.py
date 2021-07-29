#!/usr/bin/env python
# coding: utf-8

# # Project : Examining Factors Responsible for Heart Attacks
# 
# DESCRIPTION
# 
# Cardiovascular diseases are one of the leading causes of deaths globally. To identify the causes and develop a system to predict potential heart attacks in an effective manner is necessary. The data presented has all the information about relevant factors that might have an impact on cardiovascular health. The data needs to be studied in detail for further analysis.
# 
# There is one dataset data that has 14 attributes with more than 4000 data points.
# 
# You are required to determine and examine the factors that play a significant role in increasing the rate of heart attacks. Also, use the findings to create and predict a model.
About this dataset:
variable	description
age         age in years
sex 	    (1 = male; 0 = female)
cp 	        chest pain type
trestbps 	resting blood pressure (in mm Hg on admission to the hospital)
chol 	    serum cholestoral in mg/dl
fbs 	    (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg 	 resting electrocardiographic results
thalach 	 maximum heart rate achieved
exang 	     exercise induced angina (1 = yes; 0 = no)
oldpeak 	 ST depression induced by exercise relative to rest
slope 	     the slope of the peak exercise ST segment
ca 	         number of major vessels (0-3) colored by flourosopy
thal 	     1 = normal; 2 = fixed defect; 3 = reversable defect
target 	     1 or 0

# # Import the Packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # 1. Importing, Understanding, and Inspecting Data :

# ### 1.1 Perform preliminary data inspection and report the findings as the structure of the data, missing values, duplicates, etc.

# #### Importing data

# In[2]:


data= pd.read_csv("Cardiovascular diseases.csv")
details = pd.read_csv("Variable Description.csv")
data.head()


# In[3]:


data.columns


# In[4]:


data.rename(columns={'cp':'chest_pain_type','trestbps':'resting_blood_pressure','chol':'cholestoral',
                     'fbs':'fasting_blood_sugar','restecg':'resting_electrocardiographic_results',
                     'thalach':'maximum_heart_rate_achieved','exang':'exercise_induced_angina',
                     'oldpeak':'ST.depression(exercise/rest)','ca':'no_of_major_vessels',
                     'thal':'thalassemia' },inplace=True)


# In[5]:


data.head()


# In[6]:


data.shape


# #### Checking for null values

# In[7]:


data.isna().sum()


# #### Checking for duplicate values

# In[8]:


data.duplicated().sum()


# ### 1.2 Based on the findings from the previous question, remove duplicates (if any) and treat missing values using an appropriate strategy.

# #### Removing duplicate values

# In[9]:


data =data.drop_duplicates()


# #### Treating null values

# In[10]:


data.isna().any().value_counts()


# Looks like perfect data set,as there are no missing values.

# ### 1.3 Get a preliminary statistical summary of the data. Explore the measures of central tendencies and the spread of the data overall.

# #### Statistical summary of the data

# In[11]:


data.describe()


# In[12]:


data.info()


# In[13]:


#Segregating numeric and categorical values for calculations


# In[14]:


list(enumerate(data))


# In[15]:


numeric_data = data.iloc[:,[0,3,4,7,9]]
numeric_data.head(2)


# In[16]:


categorical_data = data.iloc[:,[1,2,5,6,8,10,11,12,13]]
categorical_data.head(2)


# #### Measures of central tendencies

# In[17]:


numeric_data.mean()


# In[18]:


numeric_data.median()


# #### Spread of the data 

# In[19]:


numeric_data.hist(figsize=(12,8))
plt.show()


# From above graphs, we observe that:
# 1. ST.depression(exercise/rest) is right skewed.
# 2. Maximum hear rate achieved is left skewed.
# 3. Age,Cholestrol,Resting Blood Pressure is normally distributed.

# # 2. Performing EDA and Modeling:

# ### 2.1 Identify the data variables which might be categorical in nature. Describe and explore these variables using appropriate tools. For example: count plot.

# In[20]:


list(enumerate(categorical_data))


# In[21]:


plt.figure(figsize=(18,18))
for i in enumerate(categorical_data):
    plt.subplot(3,3,i[0]+1)
    sns.countplot(i[1], data =categorical_data)


# ### 2.2 Study the occurrence of CVD across different ages.

# In[22]:


df = data[data.target==1]


# In[23]:


df.target.value_counts()


# In[24]:


plt.figure(figsize = (15,6))
sns.countplot(x ="age",data= df)
plt.title("Occurence of CVD across different ages")
plt.show()


#  We can observe that occurence of disease is more in the age group between 40 to 60, though people of age 50-60 are at more risk.

# ### 2.3 Can we detect heart attack based on anomalies in resting blood pressure of the patient?

# In[25]:


sns.boxplot(y= "resting_blood_pressure", x="target", data=data)
plt.show()


# From the above observation, there are people who does not got heart attack also have high blood pressure. Therefore, we can not detect heart attack based on resting blood pressure.

# ### 2.4 Study the composition of overall patients w.r.t . gender.

# In[26]:


sns.countplot(x="sex", data=data, hue = "target")
plt.title("Sex distribution according to Target")
plt.xlabel("Sex: 0 = Female, 1 = Male")
plt.show()


# From the above graph it can be concluded that male patients are more prone to the Cardiovascular disease.
# Target = 0 represent Don't have disease, 1 represent have Disease

# ### 2.5 Describe the relationship between cholesterol levels and our target variable.

# In[27]:


data.cholestoral.corr(data.target)


# In[28]:


sns.jointplot("cholestoral","target",data)
plt.show()


# Cholestoral and target variables have weak correlation.

# ### 2.6 What can be concluded about the relationship between peak exercising and occurrence of heart attack? 

# In[29]:


data.slope.corr(data.target)


# In[30]:


sns.countplot(x="slope",data= data, hue= "target")


# People with Downsloping(2) have more people prone to heart attack. Peak exercising is poitively correlated to the target variable.

# ### 2.7 Is thalassemia a major cause of CVD? 

# In[31]:


data.thalassemia.corr(data.target)


# In[32]:


sns.countplot(x="thalassemia",data= data, hue= "target")


# Thalassemia--0=Null, 1= Normal, 2= Fixed Defect, 3=Reversable defect. People with fixed defect are at higher risk of CVD

# ### 2.8 How are the other factors determining the occurrence of CVD?

# In[33]:


new_data = data.drop(columns =["thalassemia","cholestoral","slope"])


# In[34]:


plt.figure(figsize = (15,10))
sns.heatmap(new_data.corr(), annot = True)
plt.show()


# Chest pain type and maximum heart rate achieved are positively correlated to target, and they are the causes of heart attack, there are no major causes as such.

# ### 2.9 Use a pair plot to understand the relationship between all the given variables.

# In[36]:


plt.figure(figsize=(10,8))
sns.pairplot(data)
plt.show()


# ### 2.9 Perform logistic regression, predict the outcome for test data, and validate the results by using the confusion matrix. 

# In[37]:


df = data.copy()


# In[38]:


df.head(2)


# In[39]:


X = df.drop(["target"],axis=1)
y = df["target"]


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1234)


# In[41]:


X_train.shape , X_test.shape , y_train.shape , y_test.shape


# For training we have 241 data points and for testing we have 61

# In[42]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()


# In[43]:


log.fit(X_train,y_train)


# In[44]:


y_pred = log.predict(X_test)
y_pred


# In[45]:


from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)


# In[46]:


conf_mat


# In[47]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# Accuracy for the logistic regression model is 77%.

# # 3. Dashboarding

# ### 3.1 Visualize the variables using Tableau to create an understanding for attributes of a Diseased vs. a Healthy person.

# https://public.tableau.com/app/profile/payal.bhargava/viz/FactorsaffectingHeartAttack/Story1

# ### 3.2 Demonstrate the variables associated with each other and factors to build a dashboard

# https://public.tableau.com/app/profile/payal.bhargava/viz/Correlation-FactorsaffectingHeartAttack/Sheet1

# In[ ]:




