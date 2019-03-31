
# coding: utf-8

# 
# ___
# # Ecommerce customers Project 
# 
# 

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

import matplotlib.pyplot as plt


# In[3]:

import seaborn as sns


# In[4]:

get_ipython().magic('matplotlib inline')


# In[ ]:




# ## Get the Data
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# 
# ** Read in the Ecommerce Customers csv file as a DataFrame called customers.**

# In[5]:

df = pd.read_csv('Ecommerce Customers')


# In[6]:

df.head()


# In[7]:

df.info()


# In[8]:

df.describe()


# In[ ]:




# In[ ]:




# In[ ]:




# In[9]:

df.info()


# ## Exploratory Data Analysis
# 
# **Let's explore the data!**
# 
# For the rest of the exercise we'll only be using the numerical data of the csv file.
# ___
# **Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[10]:

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df)


# In[ ]:




# In[ ]:




# In[ ]:




# 

# In[35]:

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)


# In[36]:

sns.jointplot(x='Time on App',y='Length of Membership',kind = 'hex',data=df)


# In[ ]:




# 

# In[ ]:




# In[37]:

sns.pairplot(df)


# In[ ]:




# In[ ]:




# In[15]:

#length of membership


# In[ ]:




# In[17]:

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df,size=6)


# In[ ]:




# In[ ]:




# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# 

# In[18]:

df


# In[ ]:




# In[19]:

X= df[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]


# In[21]:

y=df['Yearly Amount Spent']


# In[ ]:




# In[ ]:




# In[ ]:




# 

# In[22]:

from sklearn.model_selection import train_test_split


# In[23]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =101)


# In[ ]:




# In[ ]:




# In[ ]:




# ## Training the Model
# 
# Now its time to train our model on our training data!
# 
# ** Import LinearRegression from sklearn.linear_model **

# In[24]:

from sklearn.linear_model import LinearRegression


# In[25]:

lm = LinearRegression()


# In[ ]:




# In[ ]:




# In[ ]:




# 

# In[ ]:




# ** Train/fit lm on the training data.**

# In[26]:

lm.fit(X_train,y_train)


# In[27]:

lm.coef_


# In[ ]:




# In[ ]:




# 

# In[ ]:




# ## Predicting Test Data
# 
# 

# In[28]:

predictedval = lm.predict(X_test)


# In[29]:

plt.scatter(y_test,predictedval)


# In[ ]:




# 

# In[ ]:




# ## Evaluating the Model
# 
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

# In[30]:

from sklearn import metrics


# In[31]:

mae = metrics.mean_absolute_error(y_test,predictedval)
mae


# In[32]:

mrse = np.sqrt(metrics.mean_squared_error(y_test,predictedval))
mrse


# In[ ]:




# ## Residuals
# 
# 

# In[33]:

sns.distplot(y_test-predictedval,bins=50)


# In[ ]:




# In[ ]:




# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.
# 
# ** Recreate the dataframe below. **

# In[34]:

finalresult = pd.DataFrame(lm.coef_,X_train.columns)
finalresult.columns=['coeffecients']
                     
finalresult


# In[ ]:




# In[ ]:




# 

# 

# 

# 

# 
