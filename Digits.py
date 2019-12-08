#!/usr/bin/env python
# coding: utf-8

# In[1]:


# loading liberaries
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits,california_housing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split_test_split


# In[2]:


digits = load_digits()# inilizating the digits to a object


# In[3]:


dir(digits)  # ditails of the data set


# In[4]:


print(digits.DESCR) # Data Set Characteristics


# In[14]:


# displaying the digits of the data sets
[plt.matshow(digits.images[i]) for i in range(10)]
plt.show()


# In[7]:


# for i in range(10):
#     plt.matshow(digits.images[i])


# In[8]:


df = pd.DataFrame(data = digits.data)  # making a data frame of the digits data


# In[9]:


df    # printing the dataframe


# In[15]:


df["target"] = digits.target  # adding the target varbile to the data frame


# In[16]:


df


# In[17]:





# In[18]:


# spliting the data data into test and train
x_train,x_test,y_train,y_test = train_test_split(df.drop(["target"],axis = "columns"),digits.target,test_size = 0.2,random_state=4)


# In[19]:


x_train # train data of x 


# In[28]:


y_train   # train data of y


# #  DecisionTreeClassifier

# In[23]:


from sklearn.tree import DecisionTreeClassifier


# In[24]:


dt = DecisionTreeClassifier()


# In[27]:


dt.fit(x_train,y_train)  # fitting the DecisionTreeClassifier model


# In[29]:


pre = dt.predict(x_test) # predicting on the test variable


# In[30]:


dt.score(x_test,y_test) # getting the score


# In[31]:


from sklearn.metrics import accuracy_score


# In[32]:


accuracy_score(y_test,pre) # getting the accuracy score from sklearn.metrics 


# # ramdom forest

# In[34]:


from sklearn.ensemble import RandomForestClassifier


# In[35]:


rf = RandomForestClassifier()


# In[36]:


rf.fit(x_train,y_train) # fittning the model


# In[37]:


pre_rf = rf.predict(x_test)  # predicting the model


# In[ ]:





# In[38]:


accuracy_score(y_test,pre_rf)  # check the accuracy


# # bagging

# In[40]:


from sklearn.ensemble import BaggingClassifier


# In[41]:


bg = BaggingClassifier(RandomForestClassifier(),n_estimators=20,max_features=1.0,max_samples=0.5)
bg.fit(x_train,y_train)  # fitting the model


# In[42]:


pre_bag = bg.predict(x_test) # predicting the results


# In[43]:


accuracy_score(y_test,pre_bag)


# # ada boosting

# In[44]:


from sklearn.ensemble import AdaBoostClassifier


# In[45]:


ad = AdaBoostClassifier(RandomForestClassifier(),n_estimators=10,learning_rate=0.5)
ad.fit(x_train,y_train)  # fitting the model


# In[46]:


pre_ad = ad.predict(x_test) # predicting the model


# In[47]:


accuracy_score(pre_ad,y_test)


# In[ ]:




