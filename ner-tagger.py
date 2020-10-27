
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from utils import data_transform, data_process, data_vectorise, classify, evaluate


# In[2]:


print("Reading original file...")
original_file = "reuters-train.en"
training_file = data_transform(original_file)


# In[3]:


print("Preprocessing data...")
processed_file = data_process(training_file)


# In[ ]:


print("Vectorising features...")
X, y = data_vectorise(processed_file)


# In[ ]:


print("Splitting data...")
X_train,X_test,y_train,y_test = train_test_split(X[:30000],y[:30000],test_size=0.3)


# In[ ]:


print("Training classifier...")
y_pred = classify(X_train, y_train, X_test )


# In[ ]:


print("Evaluating results...")
label_list = np.unique(y)
evaluate(y_test, y_pred, label_list)

