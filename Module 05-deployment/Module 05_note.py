#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## Module Overview: Deploying Machine Learning Models
# 
# This module covers how to transition trained machine learning models out of experimental Jupyter notebooks and turn them into production-ready web services.
# 
# ### Key Objectives & Workflow:
# 
# * **Model Serialization:** Save trained models and pre-processing pipelines to disk using **Pickle** (`pickle.dump` / `pickle.load`) so they can be loaded independently in scripts.
# * **Web Services & APIs:** Build lightweight web services using Python frameworks like **Flask** or **FastAPI** to handle incoming HTTP requests (GET, POST).
# * **Prediction Endpoints:** Set up application routes to parse input JSON payloads, pass them through the serialized ML model, and return prediction probabilities.
# * **Environment Management:** Use dependency managers and virtual environments (like pipenv or conda) to lock down package versions for reliable execution.
# * **Containerization with Docker:** Package the model service, dependencies, and code into a portable **Docker container** to ensure consistent behavior across any environment.

# In[2]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# load data
df = pd.read_csv('/workspaces/ML-Zoomcamp/Module 03-classification/data-week-3.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_' )
categorical_columns = list(df.dtypes[df.dtypes == 'string'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_' )

# Convert totalcharges to numeric
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[4]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[5]:


#Numerical and categorical variables

numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[6]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient = 'records')

    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter= 1000)
    model.fit(X_train, y_train)

    return dv, model


# In[7]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient = 'records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]


    return y_pred




# In[8]:


C = 1.0
n_splits = 5


# In[9]:


kfold = KFold(n_splits= n_splits, shuffle= True, random_state= 1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred_val = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred_val)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[10]:


scores


# In[11]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# **Save the model**

# In[12]:


import pickle


# In[13]:


output_file = f'model_C={C}.bin'

f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


# In[14]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# **Loading the model**

# In[1]:


import pickle


# In[2]:


with open('model_C=1.0.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[3]:


dv,model


# In[12]:


customer = { 
    'gender': 'female', 'seniorcitizen': 0, 'partner': 'yes', 'dependents': 'no', 'phoneservice': 'no', 'multiplelines': 'no_phone_service', 'internetservice': 'dsl', 'onlinesecurity': 'no', 'onlinebackup': 'yes', 'deviceprotection': 'no', 'techsupport': 'no', 'streamingtv': 'no', 'streamingmovies': 'no', 'contract': 'month-to-month', 'paperlessbilling': 'yes', 'paymentmethod': 'electronic_check', 'tenure': 1, 'monthlycharges': 29.85, 'totalcharges': 29.85 
    }


# In[13]:


X = dv.transform([customer])


# In[14]:


y_pred = model.predict_proba(X)[0,1]


# In[15]:


print(y_pred)

