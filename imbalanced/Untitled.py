#!/usr/bin/env python
# coding: utf-8

# ## Carregando os dados

# In[1]:


import pandas as pd

df = pd.read_csv('initial_dataset.csv', sep=';', encoding='utf-8').sample(30000)
df.info()


# ### Desbalanceamento
# 
# Dataset extremamente desbalanceado, com **99,4%** dos dados pertencendo a uma classe.

# In[2]:


df.target.value_counts() / df.shape[0]


# In[3]:


(df.isna().sum() / df.shape[0]).sort_values(ascending=False)[:10]


# ## Abordagem Class Weight

# ### Distribuição em 2D

# In[4]:


from sklearn.decomposition import TruncatedSVD
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

y = df.target
X = df.drop('target', axis=1).fillna(-1)
del df

X2 = TruncatedSVD(n_components=2).fit_transform(X)

counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = np.where(y == label)[0]
    plt.scatter(X2[row_ix, 0], X2[row_ix, 1], label=str(label))
plt.legend()
plt.show()


# ### Treinamento

# In[5]:


from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=42, stratify=y)

cls = RandomForestClassifier(class_weight='balanced', max_depth=6)
cls.fit(X_train, y_train)

y_pred = cls.predict(X_test)
print('ROC_AUC ', metrics.roc_auc_score(y_test, y_pred))
print('Recall ', metrics.recall_score(y_test, y_pred))
print('Precision ', metrics.precision_score(y_test, y_pred))
print('F1 ', metrics.f1_score(y_test, y_pred))
metrics.plot_confusion_matrix(cls, X_test, y_test, normalize='true');


# ## Abordagem Random Over Sample

# ### Ajustando balanceamento

# In[6]:


from imblearn import over_sampling as over

oversample = over.RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)

y_over.value_counts()


# In[7]:


X_over.shape[0] == y_over.shape[0]


# ### Distribuição em 2D
# 
# *Não há alteração perceptível porque os data points se sobrepõem*

# In[8]:


from sklearn.decomposition import TruncatedSVD
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

X2 = TruncatedSVD(n_components=2).fit_transform(X_over)

counter = Counter(y_over)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = np.where(y_over == label)[0]
    plt.scatter(X2[row_ix, 0], X2[row_ix, 1], label=str(label))
plt.legend()
plt.show()


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, train_size=.7, random_state=42, stratify=y_over)


# In[10]:


from sklearn.ensemble import RandomForestClassifier

cls = RandomForestClassifier(max_depth=25)
cls.fit(X_train, y_train)


# In[11]:


from sklearn import metrics

y_pred = cls.predict(X_test)
print('ROC_AUC ', metrics.roc_auc_score(y_test, y_pred))
print('Recall ', metrics.recall_score(y_test, y_pred))
print('Precision ', metrics.precision_score(y_test, y_pred))
print('F1 ', metrics.f1_score(y_test, y_pred))


metrics.plot_confusion_matrix(cls, X_test, y_test, normalize='true');


# ## Abordagem Over and Under Random Sample

# In[12]:


from imblearn import over_sampling as over
from imblearn import under_sampling as under


over = over.RandomOverSampler(sampling_strategy=0.1)
X_over, y_over = over.fit_resample(X, y)
print(y_over.value_counts())
under = under.RandomUnderSampler(sampling_strategy=0.5)
X_ou, y_ou = under.fit_resample(X_over, y_over)
print(y_ou.value_counts())


# ### Distribuição 2D
# 
# *Redução dos data points azuis e data points laranja ainda se sobrepondo*

# In[13]:


from sklearn.decomposition import TruncatedSVD
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

X2 = TruncatedSVD(n_components=2).fit_transform(X_ou)

counter = Counter(y_ou)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = np.where(y_ou == label)[0]
    plt.scatter(X2[row_ix, 0], X2[row_ix, 1], label=str(label))
plt.legend()
plt.show()


# ### Treinamento

# In[14]:


from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


X_train, X_test, y_train, y_test = train_test_split(X_ou, y_ou, train_size=.7, random_state=42, stratify=y_ou)

cls = RandomForestClassifier(max_depth=25)
cls.fit(X_train, y_train)

y_pred = cls.predict(X_test)
print('ROC_AUC ', metrics.roc_auc_score(y_test, y_pred))
print('Recall ', metrics.recall_score(y_test, y_pred))
print('Precision ', metrics.precision_score(y_test, y_pred))
print('F1 ', metrics.f1_score(y_test, y_pred))
metrics.plot_confusion_matrix(cls, X_test, y_test, normalize='true');


# ## Abordagem SMOTE

# In[15]:


from imblearn import over_sampling as over

smote = over.SMOTE()
X_over, y_over = smote.fit_resample(X, y)

y_over.value_counts()


# ### Distrinuição em 2D
# 
# *Claro preenchimento do campo 2D por novos data points laranja*

# In[16]:


from sklearn.decomposition import TruncatedSVD
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

X2 = TruncatedSVD(n_components=2).fit_transform(X_over)

counter = Counter(y_over)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = np.where(y_over == label)[0]
    plt.scatter(X2[row_ix, 0], X2[row_ix, 1], label=str(label))
plt.legend()
plt.show()


# In[17]:


from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, train_size=.7, random_state=42, stratify=y_over)

cls = RandomForestClassifier(max_depth=25)
cls.fit(X_train, y_train)

y_pred = cls.predict(X_test)
print('ROC_AUC ', metrics.roc_auc_score(y_test, y_pred))
print('Recall ', metrics.recall_score(y_test, y_pred))
print('Precision ', metrics.precision_score(y_test, y_pred))
print('F1 ', metrics.f1_score(y_test, y_pred))
metrics.plot_confusion_matrix(cls, X_test, y_test, normalize='true');


# ## Abordagem SMOTE com variação de centroides

# In[18]:


from sklearn import metrics
from imblearn import over_sampling as over
from sklearn.ensemble import RandomForestClassifier

for k in range(1, 9):
    smote = over.SMOTE(k_neighbors=k)
    X_over, y_over = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, train_size=.7, random_state=42, stratify=y_over)
    cls = RandomForestClassifier(max_depth=25)
    cls.fit(X_train, y_train)

    y_pred = cls.predict(X_test)
    print('-----'*5)
    print('K = %s' % k)
    print('ROC_AUC ', metrics.roc_auc_score(y_test, y_pred))
    print('Recall ', metrics.recall_score(y_test, y_pred))
    print('Precision ', metrics.precision_score(y_test, y_pred))
    print('F1 ', metrics.f1_score(y_test, y_pred))


# ### Melhores resultados encontrados com 2 centroides

# In[20]:


from sklearn.decomposition import TruncatedSVD
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

smote = over.SMOTE(k_neighbors=2)
X_over, y_over = smote.fit_resample(X, y)
X2 = TruncatedSVD(n_components=2).fit_transform(X_over)

counter = Counter(y_over)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = np.where(y_over == label)[0]
    plt.scatter(X2[row_ix, 0], X2[row_ix, 1], label=str(label))
plt.legend()
plt.show()


# In[21]:


y_pred = cls.predict(X_test)
print('ROC_AUC ', metrics.roc_auc_score(y_test, y_pred))
print('Recall ', metrics.recall_score(y_test, y_pred))
print('Precision ', metrics.precision_score(y_test, y_pred))
print('F1 ', metrics.f1_score(y_test, y_pred))
metrics.plot_confusion_matrix(cls, X_test, y_test, normalize='true');


# In[ ]:




