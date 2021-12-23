#!/usr/bin/env python
# coding: utf-8

# # LOAN ELIGIBILITY PREDICTION.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset=pd.read_csv('loan-train.csv')


# # Let's explore the "dataset"

# In[3]:


dataset.head(2)


# In[4]:


dataset.shape


# In[5]:


dataset.info()


# In[6]:


dataset.describe()


# # Let's check how a credit history affected the loan eligibility
# 

# In[7]:


pd.crosstab(dataset['Credit_History'],dataset['Loan_Status'],margins=True)


# In[38]:


dataset.boxplot(column=['ApplicantIncome'])


# Here , both applicant income and Coapplicant income are "RIGHT SKEWED" SO WE Have to "NORMALIZE THESE"

# In[9]:


dataset['ApplicantIncome'].hist(bins=20)


# In[10]:


dataset['CoapplicantIncome'].hist()


# In[11]:


#lETS CHECK ON THE BASIS GRADUATION QUALIFICATION

dataset.boxplot(column='ApplicantIncome',by='Education')#Here, you can see the graduate income is higher than un graduate.


# In[12]:


dataset.boxplot(column=['LoanAmount'])


# In[13]:


dataset['LoanAmount'].hist(bins=10) #little bit right skewed


# # Let's Normalize and Scaling for implementation and further implementation.

# In[14]:


dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=10)#lets check after normalization


# In[15]:


#Let's check missing values here.
dataset.isnull().sum()


# In[16]:


# Let's fill the missing values right here,
dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)

dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)

dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)

dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)

#It's a numerical so mean will be the rulers

dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(),inplace=True)
dataset['LoanAmount_log'].fillna(dataset['LoanAmount_log'].mean(),inplace=True)

#Loan amount has some of categorical value so we will be using Mode() over here
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)

#So in this either 0 or 1 so agin it's an categorical vale so, we will be using mode() over here
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)


# In[17]:


# Let's check missing values handled or not
dataset.isnull().sum()


# So we have already seen that Applicant income and Co-Applicant income was right skewed so let's normalize now,

# In[18]:


#Firstly, let's combine Applicant and co applicant income

dataset['TotalIncome']=dataset['ApplicantIncome']+dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])


# In[39]:


dataset['TotalIncome_log'].hist(bins=10) #Completely normalize now


# In[20]:


dataset.head()


# # Now we have normalized and handled the missing values and now let's divide our data set into an independent and dependent variables

# In[21]:


x=dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y=dataset.iloc[:,12].values


# In[22]:


x


# In[23]:


y


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)#Random state to be zero becoz i doesn't want our data change in every cycle.


# In[25]:


x_train #Here you have categorical values at a certain point so we have to convert them into simple numerical like yes or no=1 or 0


# In[26]:


#For That we will be using LABEL ENCODER()

from sklearn.preprocessing import LabelEncoder

labelencoder_x=LabelEncoder()


# In[42]:


#Let's create for loop to apply on each of the index which we will be want to convert into a numeric format from textual format.

for i in range(0,5):
    x_train[:,i]=labelencoder_x.fit_transform(x_train[:,i])
    


# In[43]:


x_train[:,7]=labelencoder_x.fit_transform(x_train[:,7])


# In[44]:


x_train


# In[45]:


#Let's create another instance on a y_train


labelencoder_y=LabelEncoder()
y_train=labelencoder_y.fit_transform(y_train)


# In[46]:


y_train


# In[47]:


# similary like x_train apply on x_test
for i in range(0,5):
    x_test[:,i]=labelencoder_x.fit_transform(x_test[:,i])
    
    x_test[:,7]=labelencoder_x.fit_transform(x_test[:,7])


# In[48]:


x_test


# In[49]:


# And for y test as well 
y_test=labelencoder_y.fit_transform(y_test)


# In[50]:


y_test


# # Let's quickly scale our data

# In[51]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()


# In[52]:


x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# # Now, it's time to apply algorithm

# # 1):Decision Classifier

# In[53]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc.fit(x_train,y_train)


# In[55]:


y_pred=dtc.predict(x_test)


# In[56]:


from sklearn.metrics import accuracy_score


# In[61]:


print("The accuracy of decision tree is: ",accuracy_score(y_pred,y_test))


# But this accuracy is not that much effective so let's apply another alogorith called "Naive Bayes"

# In[62]:


from sklearn.naive_bayes import GaussianNB
nbclassifier=GaussianNB()
nbclassifier.fit(x_train,y_train)


# In[63]:


y_pred2=nbclassifier.predict(x_test)


# In[65]:


print("The Accuracy of the NB classifier",accuracy_score(y_pred2,y_test))#It's much better than before


# In[68]:


testdata=pd.read_csv("loan-test.csv")


# In this testdataset our status is  not present becoz we want from our model predict whether a person is eligible or not that we have made before!!!
# 
# We will be using NaiveBayes to perdict this

# In[71]:


testdata.head()


# Before that let's explore the dataset

# In[72]:


testdata.isnull().sum()


# In[97]:


#Let's handle these missing values
testdata['Gender'].fillna(testdata['Gender'].mode()[0],inplace=True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0],inplace=True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0],inplace=True)
testdata['LoanAmount'].fillna(testdata['LoanAmount'].mean(),inplace=True)
dataset['LoanAmount_log'].fillna(dataset['LoanAmount_log'].mean(),inplace=True)

testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0],inplace=True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0],inplace=True)


# In[98]:


dataset['LoanAmount_Log']=np.log(dataset['LoanAmount'])


# In[100]:


testdata.isnull().sum()


# In[101]:


#Handling outliers here
testdata['Total_Income']=testdata['ApplicantIncome']+testdata['CoapplicantIncome']
testdata['Total_Income_log']=np.log(testdata['Total_Income'])


# In[107]:


#Handling outliers here 
testdata.LoanAmount=testdata.LoanAmount.fillna(testdata.LoanAmount.mean())
testdata['LoanAmount_log']=np.log(testdata['LoanAmount'])


# In[110]:


testdata.head()


# In[111]:


test=testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[113]:


for i in range(0,5):
    test[:,i]=labelencoder_x.fit_transform(test[:,i])


# In[114]:


test[:,7]=labelencoder_x.fit_transform(test[:,7])


# In[115]:


test


# Scale our data

# In[116]:


test=ss.fit_transform(test)


# In[117]:


pred=nbclassifier.predict(test)


# In[120]:


pred #these are the our prediction done by NAIVE BAYES ALOGORITHM


# # Project Ended
