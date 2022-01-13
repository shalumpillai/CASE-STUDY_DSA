#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ### 1. Read the dataset to the python environment.

# In[3]:


data = pd.read_excel('iris.xls')
data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.dtypes


# In[7]:


data.describe()


# In[8]:


# To check the null values
data.isna().sum()


# ### 2. Do necessary pre-processing steps.

# In[9]:


# Distribution plot of SL
fig,ax =plt.subplots(figsize=(8,4))
sns.distplot(data.SL)


# In[10]:


# filling the missing values with median
data['SL'] = data['SL'].fillna(data['SL'].median())


# In[11]:


# Distribution plot of SW
fig,ax =plt.subplots(figsize=(8,4))
sns.distplot(data.SW)


# In[12]:


# filling the missing values with median
data['SW'] = data['SW'].fillna(data['SW'].median())


# In[13]:


# Distribution plot of PL
fig,ax =plt.subplots(figsize=(8,4))
sns.distplot(data.PL)


# In[14]:


# filling the missing values with mode
data['PL'] = data['PL'].fillna(data['PL'].median())


# In[15]:


data.isna().sum()


# In[16]:


## checking outliers
for i in ['SW','SL', 'PW','PL']:
    #plt.title(i)
    sns.boxplot(x=data[i])
    plt.show() 


# Ouliers found in SW feature and remove the outliers using percentile method.

# In[17]:


# Handling outlier using percentile
import numpy as np
q1 = np.percentile(data['SW'],25,interpolation='midpoint')
q3 = np.percentile(data['SW'],75,interpolation='midpoint')
IQR = q3-q1
low_limit=q1-1.5*IQR
high_limit=q3+1.5*IQR
index=data['SW'][(data['SW']<low_limit)|(data['SW']>high_limit)].index
data.drop(index,inplace=True)
sns.boxplot(x=data['SW'])


# In[18]:


data.Classification.value_counts()


# ### 3. Find out which classification model gives the best result to predict iris species.(also do random forest algorithm)

# #### Label encoding to classification feature

# In[19]:


#import label encoder
from sklearn import preprocessing

#label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

#Encode labels in column 'Classification'.
data['Classification']= label_encoder.fit_transform(data['Classification'])

data['Classification'].unique()


# In[20]:


data.head()


# #### Logistic Regression

# In[21]:


# Splitting the dataset
x = data.drop(['Classification'],axis =1)
y = data['Classification']


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42,test_size=0.25)


# In[23]:


from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression()
logit_model.fit(x_train,y_train)
y_pred = logit_model.predict(x_test)


# In[24]:


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score


# In[25]:


print('Accuracy is:', accuracy_score(y_test,y_pred))
print('precision is:', precision_score(y_test,y_pred,pos_label = 'positive',average='macro'))
print('Recallscore is:',recall_score(y_test,y_pred,pos_label = 'positive',average='macro'))
print('f1_score:',f1_score(y_test,y_pred,pos_label = 'positive',average='macro'))


# In[26]:


temp=[]
temp.append(['Logistic Regression',round(f1_score(y_test,y_pred,average='macro'),5),round(accuracy_score(y_test,y_pred),5),
             round(precision_score(y_test,y_pred,pos_label = 'positive',average='macro'),5),
             round(recall_score(y_test,y_pred,pos_label = 'positive',average='macro'),5)])


# In[27]:


confusion_matrix(y_test,y_pred)


# #### K - nearest neighbour model

# In[28]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
acc_val=[]
neighbors=np.arange(3,15)
for k in neighbors:
    classifier=KNeighborsClassifier(n_neighbors=k,metric='minkowski')
    classifier.fit(x_train,y_train)
    y_pred=classifier.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    acc_val.append(acc)


# In[29]:


plt.plot(neighbors,acc_val)
plt.xlabel('k values')
plt.ylabel("accuracy")


# In[30]:


#finding best k
best_k = neighbors[acc_val.index(max(acc_val))]
print("The optimal number of neighbors is %d." % best_k)


# In[31]:


classifier=KNeighborsClassifier(n_neighbors=3,metric='minkowski')
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)


# In[32]:


print('Accuracy is:', accuracy_score(y_test,y_pred))
print('precision is:', precision_score(y_test,y_pred,pos_label = 'positive',average='macro'))
print('Recallscore is:', recall_score(y_test,y_pred,pos_label = 'positive',average='macro'))
print('f1_score:',f1_score(y_test,y_pred,pos_label = 'positive',average='macro'))


# In[33]:


temp.append(['KNN',round(f1_score(y_test,y_pred,average='macro'),5),round(accuracy_score(y_test,y_pred),5),
             round(precision_score(y_test,y_pred,pos_label = 'positive',average='macro'),5),
             round(recall_score(y_test,y_pred,pos_label = 'positive',average='macro'),5)])


# #### SVM

# In[34]:


#SVM
from sklearn.svm import SVC
svm_linear=SVC(kernel='linear')
svm_linear.fit(x_train,y_train)
y_pred=svm_linear.predict(x_test)


# In[35]:


print("accuracy is :",accuracy_score(y_test,y_pred))
print("precision is :",precision_score(y_test,y_pred,average='macro'))
print("recall is :",recall_score(y_test,y_pred,average='macro'))
print("F1 score is :",f1_score(y_test,y_pred,average='macro'))


# In[36]:


temp.append(['Linear Svm',round(f1_score(y_test,y_pred,average='macro'),5),round(accuracy_score(y_test,y_pred),5),
             round(precision_score(y_test,y_pred,pos_label = 'positive',average='macro'),5),
             round(recall_score(y_test,y_pred,pos_label = 'positive',average='macro'),5)])


# In[37]:


## polynomial svm
svm_poly=SVC(kernel='poly',degree=3)
svm_poly.fit(x_train,y_train)
y_pred=svm_poly.predict(x_test)


# In[38]:


print("accuracy is :",accuracy_score(y_test,y_pred))
print("precision is :",precision_score(y_test,y_pred,average='macro'))
print("recall is :",recall_score(y_test,y_pred,average='macro'))
print("F1 score is :",f1_score(y_test,y_pred,average='macro'))


# In[39]:


temp.append(['polynomial svm',round(f1_score(y_test,y_pred,average='macro'),5),round(accuracy_score(y_test,y_pred),5),
             round(precision_score(y_test,y_pred,pos_label = 'positive',average='macro'),5),
             round(recall_score(y_test,y_pred,pos_label = 'positive',average='macro'),5)])


# In[40]:


## radial svm
svm_radial=SVC(kernel='rbf')
svm_radial.fit(x_train,y_train)
y_pred=svm_radial.predict(x_test)


# In[41]:


print("accuracy is :",accuracy_score(y_test,y_pred))
print("precision is :",precision_score(y_test,y_pred,average='macro'))
print("recall is :",recall_score(y_test,y_pred,average='macro'))
print("F1 score is :",f1_score(y_test,y_pred,average='macro'))


# In[42]:


temp.append(['Radial Svm',round(f1_score(y_test,y_pred,average='macro'),5),round(accuracy_score(y_test,y_pred),5),
             round(precision_score(y_test,y_pred,pos_label = 'positive',average='macro'),5),
             round(recall_score(y_test,y_pred,pos_label = 'positive',average='macro'),5)])


# #### RANDOM FOREST

# In[43]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier
rm=RandomForestClassifier()
rm.fit(x_train,y_train)
y_pred=rm.predict(x_test)


# In[44]:


print("accuracy is :",accuracy_score(y_test,y_pred))
print("precision is :",precision_score(y_test,y_pred,average='macro'))
print("recall is :",recall_score(y_test,y_pred,average='macro'))
print("F1 score is :",f1_score(y_test,y_pred,average='macro'))


# In[45]:


temp.append(['Random forest',round(f1_score(y_test,y_pred,average='macro'),5),round(accuracy_score(y_test,y_pred),5),
             round(precision_score(y_test,y_pred,pos_label = 'positive',average='macro'),5),
             round(recall_score(y_test,y_pred,pos_label = 'positive',average='macro'),5)])


# In[46]:


df1 = pd.DataFrame(temp, columns = ['Algorithms', 'F1-Score','accuracy','precision','recall'])
df1


#  ####  Random Forest has highest F1 score (0.93915) and accuracy(0.94595) than other algorithms.Also polynomial Svm has F1 - score (0.9312) and accuracy(0.94595).

# In[ ]:




