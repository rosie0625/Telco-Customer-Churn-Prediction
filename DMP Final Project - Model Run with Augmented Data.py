#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install imbalanced-learn
#pip install delayed


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings("ignore")


# In[4]:


# please read the csv directly for next step
df = pd.read_csv('telco_with_polarity.csv')


# In[5]:


df.head()


# In[7]:


x = df.drop('Churn',axis=1)


# In[17]:


y = df[['Churn']]


# In[18]:


SEED = 4
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, stratify=y, random_state=SEED)


# ### Step 1: Resample Imbalanced Training Data Using Smote and Oversampling (Hybrid)

# In[26]:


y_train[y_train['Churn']=='No'].shape


# In[27]:


y_train[y_train['Churn']=='Yes'].shape


# In[31]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

over = SMOTE(sampling_strategy=0.6) # First oversample Yes to be 60% of the original No
under = RandomUnderSampler(sampling_strategy=0.8) # Then undersample No so that Yes:No ration is 4:5

steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)


# In[33]:


x_train,y_train = pipeline.fit_resample(x_train,y_train)


# In[34]:


y_train[y_train['Churn']=='No'].shape


# In[36]:


y_train[y_train['Churn']=='Yes'].shape


# ### Step 2: Run Various Model to Test Accuracy and F1 scores

# In[ ]:




