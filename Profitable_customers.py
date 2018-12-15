
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy = True')


# In[2]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mt
from sklearn.preprocessing import Imputer


# In[3]:


#==============================================================================
# Importing the dataset
#==============================================================================
dataset_new = pd.read_csv('big_case_train.csv')

dataset_test = pd.read_csv('puzzle_train_dataset.csv')

#selectingh the reorganized csv file to save time
#Use Celll Reorganize data to get this file.
dataset = pd.read_csv('reorganized.csv')

#dataset_test = pd.read_csv('puzzle_test_dataset.csv')


# #=============================================================================
# #Reorganize data
# #==============================================================================
# 
# imputer_mode = Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
# dataset.iloc[:,[1]]=imputer_mode.fit_transform(dataset.iloc[:,[1]])
# sz = len(dataset)
# defaulters = range(0,sz)
# for i in defaulters:
#     if dataset['default'][i] == True:
#         dataset = dataset.append(dataset.iloc[i,:],ignore_index=True)
# for i in log_progress(defaulters):
#     if dataset['default'][i] == True:
#         dataset.drop(i,inplace=True)
# #Save data to csv for reuse
# dataset.to_csv('reorganized.csv', sep=',', index=False)
# 

# In[4]:


#==============================================================================
#Calculating the total profit
#==============================================================================
profit = dataset_new['spends']*0.05 + dataset_new['revolving_balance']*0.17 - dataset_new['card_request']*10 - dataset_new['minutes_cs']*2.6

#Calculating the absolute profit after considering the inflation costs
dataset_new['absolute_profit'] = profit - profit * dataset_new['month']*0.005 

#Adding the processed data from big_data.csv to our training dataset.
#dataset_new['default'] = dataset['default']


# In[5]:


#==============================================================================
#creating np array to calculate total profits after considering revolving balance
#==============================================================================
big_data = dataset_new.iloc[:,[0,-1,4]].values
ids = np.ndarray(shape=(np.unique(big_data[:,0]).size,4),dtype=float)
ids[:,0] = dataset['default']
big_data


# In[6]:


#==============================================================================
#calcualte total profit to the bank
#==============================================================================
#Number of customers
count=0
#Total profit from each customer
total_profit = 0
#Total number of months
total = 0
#number of defaults and non-defaults per user
ndf = 0
df = 0
#Loop through all rows 
for i in range(0,len(big_data)-1):
    
    #Total number of months
    total= total + 1
    if big_data[i,2] == 0:
        ndf = ndf + 1
    else:
        df = df + 1
        
    #check to see if end of present cutomer data
    if(big_data[i,0]==big_data[i+1,0]):
        #Add the absolute profit calculated
        total_profit = total_profit + big_data[i,1]
    else:
        #If not a defaulter
        if ids[count,0] == False:
            ids[count,1] = total_profit + big_data[i,1]
        # If defaulter do not add last payable amount
        else:
            ids[count,1]= total_profit
      
        pro_per = round(0.6*total,0)
        rem_per = total - pro_per
        df_per = df/total
        per_prob = mt.pow(df_per,total)
        pp =  mt.factorial(total)/ mt.factorial(pro_per)*mt.factorial(rem_per)
        probability =pp * per_prob
        ids[count,2] = probability
        #print(probability)
        df = 0
        ndf = 0
        total = 0
        #move to next customer
        count = count + 1
        #reset to profits for next  customer
        total_profit = big_data[i+1,1]
        
#Calculate loss from defaulters        
#If not a defaulter
if ids[count,0] == False:
    ids[count,1] = total_profit + big_data[i,1]
# If defaulter do not add last payable amount
ids[count,1]= total_profit
dataset.describe()


# In[7]:


#==============================================================================
#Append our training data set with total profit to bank
#==============================================================================
#Copy Profits to dataset
dataset['total_profits'] = ids[:,1]
dataset['probability'] = ids[:,2]
pctile = np.percentile(dataset['probability'], 10)
#Calculate Credit score
for i in range(0,len(dataset['total_profits'])):
    if dataset['total_profits'][i] >0 and dataset['probability'][i] > pctile:
        ids[i,3] = True
    else:
        ids[i,3] = False
        
#Copy credit score to dataset
dataset['profitable_score'] = ids[:,3]
dataset.describe()


# In[8]:


#==============================================================================
# Create array for data preprocessing
#==============================================================================
#Assign the Independent Variable
iv = dataset.iloc[:,1:-1].values

#Assign the Dependent Variable
dv = dataset.iloc[:,-1].values


# In[9]:


#==============================================================================
#Imputation I
#==============================================================================
#Remove Special characters and impute text
for i in range(0,len(iv)):
    for j in range(0,28):
        if type(iv[i][j])==str:
            iv[i,j]= ''.join(e for e in iv[i,j] if e.isalnum())
            if j == 11:
                iv[i,j] = iv[i,j][:86]
            if j == 21:
                iv[i,j] = iv[i,j][:64]
iv[0]


# In[10]:


#==============================================================================
#Encoding I
#==============================================================================
#Label encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encode_iv = LabelEncoder()

for i in [1,2,9,11,13,14,15,16,17,18,19,20,21,22]:
    iv[:,i] = encode_iv.fit_transform(iv[:,i].astype(str))
iv[0]


# In[11]:


#==============================================================================
#Imputation II
#==============================================================================

#Clear an instance of the class Imputer
imputer_median = Imputer(missing_values="NaN",strategy="median",axis=0)
imputer_mode = Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
imputer_mode2 = Imputer(missing_values="NaN",strategy="most_frequent")

for i in [0,1,2,9,11,13,14,15,16,17,18,19,20,21,22,23,24,25]:
    iv[:,[i]]=imputer_mode.fit_transform(iv[:,[i]])
    
for i in [3,4,5,6,7,8,10,12,22,26,27,28]:
    iv[:,[i]]=imputer_median.fit_transform(iv[:,[i]])


# In[12]:


#==============================================================================
#Encoding II
#==============================================================================
#One Hot encoding
onehotencoder = OneHotEncoder(categorical_features=[1,2,9,11,13,16,17,18,19,20,21,22])
iv = onehotencoder.fit_transform(iv).toarray()
iv[0]


# In[13]:


#==============================================================================
#Scaling
#==============================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
iv = scaler.fit_transform(iv)
iv


# In[14]:


#==============================================================================
#Spliting train test
#==============================================================================
from sklearn.model_selection import train_test_split
iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=0)


# In[15]:


#==============================================================================
# Fitting Random Forest Classification to the Training set
#==============================================================================
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(iv_train, dv_train)


# In[16]:


#==============================================================================
# Predict the values 
#==============================================================================
prediction_RF = classifier.predict (iv_test)

print("")
print("test sample data :-")
print(dv_test)
print("")
print("predicted output :-")
print(prediction_RF)
print("")
credit_profit = pd.DataFrame()
credit_profit['prediction_RF'] = credit_profit.append(pd.DataFrame(prediction_RF,columns=['A']),ignore_index=True)
credit_profit['test_values'] = dv_test
credit_profit.to_csv('test.csv', sep=',', index=False)


# In[17]:


#==============================================================================
# Create confusion matrix to evaluate performance of data
#==============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix (dv_test, prediction_RF)
TN, FP, FN, TP = confusion_matrix(dv_test, prediction_RF).ravel()
print("Confusion Matrix ")
print(cm)


# In[18]:


#==============================================================================
# Performance Data
#==============================================================================
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative Precision or negative predictive value
NPV = TN/(TN+FN)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print("Sensitivity : " , round(TPR,2))
print("Specivity : ", round(TNR,2))
print("Positive Precision : ",round(PPV,2))
print("Negative Precision : ",round(NPV,2))
print("Accuracy : " , round(ACC,2))

