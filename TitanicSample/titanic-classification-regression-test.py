#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[4]:


import pickle as pkl
rf=pkl.load(open("./model.pkl",'rb'))


# In[5]:


test = pd.read_csv('./test.csv')


# In[6]:


test.drop('Cabin',axis=1,inplace=True)


# In[7]:


test['Fare'].fillna(test['Fare'].median(), inplace=True)


# In[8]:


test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)


# In[9]:


sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test= pd.get_dummies(test['Embarked'],drop_first=True)


# In[10]:


test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[11]:


test = pd.concat([test,sex_test,embark_test],axis=1)


# In[12]:


test_prediction=rf.predict(test)


# In[13]:


test_pred = pd.DataFrame(test_prediction, columns= ['Survived'])


# In[14]:


new_test = pd.concat([test, test_pred], axis=1, join='inner')


# In[15]:


df= new_test[['PassengerId' ,'Survived']]


# In[16]:


df.to_csv('predictions.csv' , index=False)

