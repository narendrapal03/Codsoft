#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# ### OBJECTIVES
# In this Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not.
# 
# ![image-2.png](attachment:image-2.png)

# In[2]:


#Important libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv('tested.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.size


# In[9]:


df['Survived'].value_counts()


# * Here 0 means non survived 1 means survived that is low passengers are survived.

# In[10]:


## Visualisation
sns.countplot(x=df['Survived'], hue =df['Pclass'])


# In[11]:


df['Sex']


# * String values does not help for predict model, So I have to convert into integer values.

# In[12]:


sns.countplot(x=df['Sex'], hue =df['Survived'])


# * Here we can see male person has NOT survived as per comparison female passengers.

# In[14]:


df.groupby('Sex')[['Survived']].mean()


# In[15]:


df['Sex'].unique()


# In[16]:


from sklearn.preprocessing import LabelEncoder
Label =LabelEncoder()


# In[17]:


df['Sex']= Label.fit_transform(df['Sex'])
df.head()


# * I have converted male & female into numeric values(0= Female and 1= Male).

# In[18]:


df['Sex'],df['Survived']


# In[19]:


sns.countplot(x=df['Sex'], hue =df['Survived'])


# In[20]:


df.isna().sum() #Check null values.


# * Here we can check null values into Age column that I'll drop in this datset.

# In[21]:


df=df.drop(['Age'],axis = 1) # drop null values.


# In[22]:


df_final = df
df_final.head()


# In[23]:


# We have need only two column for prediction Passenger class and gender column that is enough.
x=df[['Pclass','Sex']]
y=df['Survived']


# In[24]:


# splits are dataset into two portion 80% of training and 20% of testing.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y, train_size = 0.2, random_state = 0)


# ## Logistic Regression
# Logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary).
# Like all regression analyses, the logistic regression is a predictive analysis.
# Logistic regression is used to describe the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables
# The name "logistic regression" is derived from the concept of the logistic function that it uses. The logistic function is also known as the sigmoid function. The value of this logistic function lies between zero and one.
# ![image.png](attachment:image.png)

# In[25]:


from sklearn.linear_model import LogisticRegression
logit=LogisticRegression(random_state = 0)
logit.fit(x_train, y_train)


# ### Model Prediction

# In[27]:


pred=print(logit.predict(x_test))


# In[28]:


print(y_test)


# In[29]:


import warnings
warnings.filterwarnings ('ignore')


# In[97]:


res=logit.predict([[2,0]])
if (res==0):
        print('Not Survived')
else:
        print('Survived')


# In[30]:


res=logit.predict([[2,1]])
if (res==0):
        print('Not Survived')
else:
        print('Survived')


# ### Conclusion
# - Yes! Female passengers are survived.
# 
