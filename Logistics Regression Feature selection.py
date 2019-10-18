#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Implementation of Logistics Regression on Titanic Datset with many feature engg techniques.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv("train_titanic.xls")


# In[3]:


train.head()


# In[4]:


sns.heatmap(train.isnull())


# In[5]:


train.head()


# In[6]:


sns.countplot(x='Survived',data=train)


# In[7]:


sns.countplot(x='Survived',hue='Sex',data=train)


# In[8]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[9]:


train.head()


# In[10]:


train.replace(('male','female'), (1, 0), inplace=True)


# In[11]:


train['Embarked'].unique()


# In[12]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[24]:


train.isna().sum()


# In[23]:


train["Age"].fillna(train["Age"].median(skipna=True), inplace=True)


# In[14]:


train.head()


# In[111]:


x= train[['Age','Sex','PassengerId','Pclass','SibSp','Parch','Fare']]
y= train[['Survived']]


# In[76]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, train['Survived'],test_size=0.30,random_state=101)


# In[77]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()


# In[78]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.30, 
                                                    random_state=101)


# In[79]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[80]:


logmodel.predict(X_test.iloc[2:3])
X_test.iloc[2:3]


# In[81]:


#predictions = logmodel.predict(X_test.iloc[2:3])
#predictions


# In[82]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)*100


# ## selectKBest

# In[33]:


X= train[['Age','Sex','Pclass','Fare']]
Y= train[['Survived']]


# In[109]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, train['Survived'],test_size=0.30,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.30, 
                                                    random_state=101)


# In[37]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[38]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)*100


# In[39]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func= chi2, k=4)
fit = test.fit(X, Y)


# In[40]:


fit


# In[41]:


np.set_printoptions(precision=3)
print(fit.scores_)
#['Age','Sex','PassengerId','Pclass','SibSp','Parch','Fare']


# In[42]:


features = fit.transform(X)
# summarize selected features
print(features[0:,:])


# ## RFE

# In[43]:


X1= train[['Age','Sex','PassengerId','Pclass','SibSp','Parch','Fare']]
Y1= train[['Survived']]


# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, Y1,test_size=0.30,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1,Y1, test_size=0.30, 
                                                    random_state=101)


# In[115]:


# feature extraction
from sklearn.feature_selection import RFE
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 4)
fit = rfe.fit(X1, Y1)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


# In[116]:


#no of features
nof_list=np.arange(1,13)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)
    model = LogisticRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
print(fit.ranking_)


# In[120]:


## here we are fetching the feature name 

cols = list(x.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 6) ## as above we got 6 feature             
#Transforming data using RFE
X_rfe = rfe.fit_transform(x,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# ## Extra trees classifier

# In[92]:


# feature extraction
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X1, Y1)
print(model.feature_importances_)


# ## Correlation

# In[94]:


plt.figure(figsize=(12,10))
cor = train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[105]:


#Correlation with output variable
cor_target = abs(cor["Survived"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)


# ## Backward Elimination using OLS Method.

# In[112]:


#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(x)
#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
model.pvalues


# In[113]:


#Backward Elimination
cols = list(x.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = x[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# ## Embedded Method

# In[122]:


reg = LassoCV()
reg.fit(x, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(x,y))
coef = pd.Series(reg.coef_, index = x.columns)


# In[126]:


coef


# In[127]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

##looking below we can eliminate 'PassengerID' and 'Fare'


# In[ ]:




