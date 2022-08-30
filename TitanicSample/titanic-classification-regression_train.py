#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('./train.csv')


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


# Now apply that function!

# In[4]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[5]:


train['Embarked'] = train['Embarked'].fillna('S')


# In[6]:


train.drop('Cabin',axis=1,inplace=True)


# In[7]:


train.dropna(inplace=True)


# In[8]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[9]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[10]:


train = pd.concat([train,sex,embark],axis=1)


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'],axis=1), 
                                                    train['Survived'], test_size=0.10, 
                                                    random_state=101)


# In[13]:


from sklearn.ensemble import RandomForestClassifier


# In[14]:


rf= RandomForestClassifier(n_estimators=500)
rf.fit(X_train,y_train)


# In[15]:


import pickle as pkl
pkl.dump(rf,open("model.pkl","wb"))


# In[16]:


rf_pre=rf.predict(X_test)

