#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## OBJECTIVE
# Sales prediction involves forecasting the amount of a product that
# customers will purchase, taking into account various factors such as
# advertising expenditure, target audience segmentation, and
# advertising platform selection.
# 
# ### Problem Statement
# * Build a model which predicts sales based on the money spent on different platforms for marketing.
# 
# ![image.png](attachment:image.png)

# In[1]:


# load important library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Load Dataset
df= pd.read_csv('Sales.csv')


# In[3]:


df.head()


# In[4]:


# About information
df.info()


# In[5]:


# know about rows and columns
df.shape # There are 4 columns and 200 rows in this dataset.


# In[6]:


# about the dataset variables
df.describe() # There are 4 variables.


# ####  Statistical Observation
# - Average investment for expense on TV advertisment.
# - Average investment for expense on Radio advertisment.
# - Maximum sales is 27 and minimum sales is 1.60.
# 

# In[7]:


df.isnull().sum() # No null values in this data.


# In[8]:


df.nunique()


# In[9]:


# Visualization
df.hist(bins=50,figsize=(20,15))


# In[10]:


sns.pairplot(df, x_vars=['TV','Radio','Newspaper'],y_vars = 'Sales', kind='hist')
plt.show()


# According pairplot observation when advertisment cost increases on TV ads then sales will increase as well.while newspaper and radio is bit unpredictable.

# In[11]:


# Histogram
df['TV'].plot.hist(bins=12)
plt.show()


# In[12]:


df['Radio'].plot.hist(bins=12)
plt.show()


# In[13]:


df['Newspaper'].plot.hist(bins=12)
plt.show()


# - As per observastion by histogram low advertisment cost of Newspaper.

# In[14]:


# Coorelation 
sns.heatmap(df.corr(), annot = True, cmap = 'Purples', linewidths = 0.1)
plt.show()


# - Sales is the highly correlated with TV.

# In[15]:


# --- Splitting Dataset 70:30 and Model selection.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size = 0.2, random_state = 0)


# In[16]:


x_train.head()


# In[17]:


y_train.head()


# In[18]:


x_test.head()


# In[19]:


y_test.head()


# In[20]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# # Linear Regression
# Linear Regression a data plot that graphs the Linear Relationship between an independent and a dependent varibles. It is typically used to visually show the Strength of relationship, the dispersion of results.

# In[21]:


from sklearn.metrics import accuracy_score


# In[22]:


from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score


# In[23]:


model = LinearRegression()
model.fit(x_train, y_train)
linear_pred = model.predict(x_test)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)


# In[24]:


from sklearn.tree import DecisionTreeRegressor


# In[25]:


tree_model = DecisionTreeRegressor()
tree_model.fit(x_train, y_train)
tree_pred = tree_model.predict(x_test)
tree_mse = mean_squared_error(y_test, tree_pred)
tree_r2 = r2_score(y_test, tree_pred)


# In[26]:


print('Linear Regression - R-squared:',linear_r2*100)
#print('DecisionTreeRegressor - R-squared:',tree_r2*100)


# In[27]:


# I have selected 2 models for check accuracy in which Linear Regression's accuracy is 80%.


# In[28]:


res = model.predict(x_train)
print(res)


# In[29]:


model.coef_


# In[30]:


model.intercept_


# 
#  ![Screenshot%202023-09-28%20202449.png](attachment:Screenshot%202023-09-28%20202449.png)

# In[31]:


0.0544343*36.9+7.16227597 # 36.9 values taken by TV's 1st column. 


# In[32]:


plt.plot(res)


# In[36]:


from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.scatter(x_test,y_test)
plt.plot(x_test, 7.16227597+0.0544343 * x_test, 'r')
plt.show()


# ### Conclusion
# - I have applied supervised machine learning models like Linear Regression and Decision tree.In which I have find of LR model of accuracy 80%. It's means accuracy is not high.
# - Sales is the highly correlated with TV.
# - As per observastion by histogram low advertisment cost of Newspaper.
# - I have predicted that TV is the best platform for advertising of sales.
# 
