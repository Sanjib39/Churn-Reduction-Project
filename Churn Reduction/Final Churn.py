
# coding: utf-8

# In[29]:


import os
import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


# In[67]:


os.chdir("D:/PROJECT")


# In[68]:


train=pd.read_csv("Train_data.csv")


# In[69]:


test=pd.read_csv("Test_data.csv")


# In[70]:


train=train.drop(['state','account length','area code','phone number'],axis=1)


# In[71]:


test=test.drop(['state','account length','area code','phone number'],axis=1)


# In[72]:


def cat_to_num(df):
    for i in range(0, df.shape[1]):
        #print(i)
        if(df.iloc[:,i].dtypes == 'object'):
            df.iloc[:,i] = pd.Categorical(df.iloc[:,i])
            df.iloc[:,i] = df.iloc[:,i].cat.codes
            df.iloc[:,i] = df.iloc[:,i].astype('object')
    return df


# In[73]:


train = cat_to_num(train)


# In[74]:


test= cat_to_num(test)


# In[75]:


train_class_0 = train[train['Churn'] == 0]
train_class_1 = train[train['Churn'] == 1]


# In[76]:


count_class_0, count_class_1 = train.Churn.value_counts()


# In[77]:


train_class_1_over = train_class_1.sample(count_class_0,replace=True)
train = pd.concat([train_class_0,train_class_1_over], axis=0)


# In[78]:


train.Churn.value_counts()


# In[79]:


cnames=['number vmail messages',
       'total day minutes', 'total day calls', 'total day charge',
       'total eve minutes', 'total eve calls', 'total eve charge',
       'total night minutes', 'total night calls', 'total night charge',
       'total intl minutes', 'total intl calls', 'total intl charge',
       'number customer service calls']


# In[81]:


train_corr=train.loc[:,cnames]


# In[46]:


f,ax=plt.subplots(figsize=(7,5))
corr=train_corr.corr()
sn.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sn.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)


# In[83]:


train=train.drop(['total day minutes','total eve minutes','total night minutes','total intl minutes', ],axis=1)


# In[84]:


test=test.drop(['total day minutes','total eve minutes','total night minutes','total intl minutes', ],axis=1)


# In[85]:


train.columns


# In[86]:


X_train=train.iloc[:,0:12]


# In[87]:


Y_train=train.iloc[:,12]


# In[88]:


X_test=test.iloc[:,0:12]


# In[89]:


Y_test=test.iloc[:,12]


# In[90]:


X_train=X_train.astype('int')


# In[91]:


Y_train=Y_train.astype('int')


# In[92]:


rf=RandomForestClassifier(n_estimators=100).fit(X_train,Y_train)


# In[93]:


prediction=rf.predict(X_test)


# In[94]:


from sklearn.metrics import confusion_matrix


# In[95]:


cm=pd.crosstab(Y_test,prediction)


# In[96]:


cm


# In[97]:


TN=cm.iloc[0,0]
FN=cm.iloc[1,0]
TP=cm.iloc[1,1]
FP=cm.iloc[0,1]


# In[98]:


Accuracy=((TP+TN)*100)/(TP+TN+FP+FN)


# In[99]:


Accuracy


# In[100]:


(FN*100)/(FN+TP)


# In[101]:


TN/(TN+FP)


# In[102]:


TP/(TP+FN)

