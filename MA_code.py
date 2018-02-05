

import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import json
import pandas as pd
from pandas import DataFrame
from pandas.io.json import json_normalize


# In[33]:



##Convert Dataset fron JSON to CSV
my_list = []
with open('MDataset_Type1.json') as json_data:
    for line in json_data:
        #line = line.replace('\\','')
         my_list.append(json.loads(line))
        
data = json_normalize(my_list)       


# In[34]:


df = DataFrame(data)


# In[35]:


df.head()


# In[36]:


df.describe()


# In[37]:


df.columns


# # Dataset first processing stages

# In[77]:


import pandas as pd
import re
#Renaming columns

df2 = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) # Remove spaces from columns
df2 = df.rename(columns={c: c.replace('properties.sig_', '') for c in df.columns}) # Remove properties. from columns
df2 = df.rename(columns={c: c.replace('properties.', '') for c in df.columns}) # Remove properties. from columns
    


# In[78]:


df2.columns


# In[79]:


#cutting unuseful features
df2 = df2.iloc[:,5:]


# In[80]:


df2.shape


# In[81]:


df2.columns


# In[82]:


df2 = df2.set_index('label')


# In[83]:


df2.head()
df2.columns


# In[84]:


#checking the dataset for any duplications
df2.duplicated()


# In[85]:


df2 = df2.drop_duplicates()


# In[86]:


df2.duplicated()


# In[87]:


# create a function that counts the number of cell elements(Evidences)
def count_cell_elements(x):
    # that, if x is a string,
    if type(x) is not str:
        # just returns it untouched
        return x
    # but, if not, return it multiplied by 100
    elif x:
        return len(x.split())
        
    # and leave everything else
    else:
        return 


# In[88]:


df2 = df2.applymap(count_cell_elements)


# In[89]:


df2.duplicated()


# In[90]:


df2 = df2.drop_duplicates()


# In[91]:


df2.duplicated()


# In[92]:


df2.head()


# In[93]:


df2.shape


# In[94]:


#Drop all COLUMNS where all cells have missing values
df2 = df2.dropna(axis=1, how='all')


# In[95]:


df2.shape


# In[96]:


df2.columns


# In[97]:


df2 = df2.reset_index()


# In[98]:


#chosing the most occuring families of malware
df2 = df2[df2['label'].isin(['APT1','Crypto','Locker','Zeus','shadowbrokers'])]


# In[99]:


df2.shape


# In[100]:


df2.head()


# In[101]:


df2_no_label = df2.drop(columns='label')


# In[102]:


df2_label_only = df2['label']


# In[104]:


df2 = pd.concat([df2_no_label,df2_label_only],axis=1)


# In[106]:


df2.head()


# # Exploratory analysis

# In[111]:


df2.shape


# Through description, we can see that the dataset has many missing values and is not normalized.

# In[109]:


# descriptions, change precision to 3 places
set_option('precision', 3)
df2.describe()


# In[112]:


df2['label'].value_counts()


# In[131]:


#Slice the first twenty features and get the label/class
df2_first20_cols = df2.iloc[:,:20]
df2_last_col = df2.iloc[:,-1]
df2_partial = pd.concat([df2_first20_cols,df2_last_col], axis = 1)

#fill NULLs with zeros
df2_partial = df2_partial.fillna(0)
print(df2_partial.shape)
df2_partial.head()


# Let's draw the histogram of the first went features to have quick insight on the status of our features. This histogram shows that features need further treatment.

# In[132]:


# histograms
# 1st 20 features
fig = pyplot.figure(figsize = (10,10))
ax = fig.gca()
df2_partial.hist(ax = ax,normed=False)
pyplot.show()


# Density plot to assess the skewness in the features

# In[133]:


# density
# First 20 features

fig = pyplot.figure(figsize = (20,20))
ax = fig.gca()
df2_partial.plot(ax = ax, kind='density', subplots=True, layout=(4,5), sharex=False, legend=True,fontsize=1)
pyplot.show()


# Box plot shows that there are a lot of outliers in the dataset

# In[134]:


import matplotlib.cm as cm
# box and whisker plots
fig = pyplot.figure(figsize = (15,15))
ax = fig.gca()
df2_partial.plot(ax = ax, kind='box', subplots=True, layout=(4,5), sharex=False, sharey=False, legend=True, fontsize=15)
pyplot.show()


# In[135]:


sns.set_style("whitegrid")

pyplot.figure(figsize=(15,12))

# create our boxplot which is drawn on an Axes object
bplot = sns.boxplot( data=df2_partial, whis=[5,95], palette="Set3", orient='h')

title = ('Distribution of values in features' 
         '\nas per each sample')

# We can call all the methods avaiable to Axes objects
bplot.set_title(title, fontsize=20)
bplot.set_xlabel('Evedences count', fontsize=16)
bplot.set_ylabel('Features', fontsize=16)
bplot.tick_params(axis='both', labelsize=12)

sns.despine(left=True) 

# plt.text(-1, -.5, 
#         'Data source: http://www.basketball-reference.com/draft/'
#        '\nAuthor: Savvas Tjortjoglou (savvastjortjoglou.com)'
#         '\nNote: Whiskers represent the 5th and 95th percentiles',
#          fontsize=12)
pyplot.show()


# Correlation matrix to check the correlations among features.

# In[136]:


# correlation matrix

col_names = df2_partial.columns

fig = pyplot.figure(figsize = (8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(df2_partial.corr(),  vmin=-1, vmax=1,cmap="RdYlGn", interpolation='none',aspect='auto')
fig.colorbar(cax)

ticks = numpy.arange(0,20,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(col_names, rotation='vertical')
ax.set_yticklabels(col_names)
#pyplot.suptitle('AAAAAAAAAA', fontsize=15, fontweight='bold')
pyplot.show()



# Scatter plot

# In[137]:


sns.pairplot(df2_partial, kind="scatter")


# #### Plot of features occurances in the datasetÂ¶

# In[142]:


df2.head()


# In[162]:


df2_desc = df2.describe()


# In[163]:


df2_desc


# In[164]:


df2_desc.index


# We see that ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'] are contained in the index. Let's change the index to a full column

# In[165]:


df2_desc.reset_index(level=0, inplace=True)


# In[166]:


df2_desc.head()


# In[168]:


#select all from index where index = count
df2_desc_feature_occ = df2_desc[df2_desc['index']=='count']
df2_desc_feature_occ


# In[169]:


#A plot that shows the occurances of features in the dataset
pyplot.figure(figsize=(50,10))
sns.barplot(data=df2_desc_feature_occ)
pyplot.title('Features Occurances',fontsize=20)
pyplot.ylabel('Occurrences', fontsize=12)
pyplot.xlabel('Features', fontsize=1)
pyplot.xticks(rotation=45)
pyplot.show()


# #### Counting the number of feature occurances by category/class

# In[170]:


df2.groupby('label').count()


# In[171]:


df2_desc_feature_occ_per_cat = df2.groupby('label').count()


# In[172]:


df2_desc_feature_occ_per_cat.reset_index(level=0, inplace=True)


# In[174]:


pyplot.figure(figsize=(10,5))
sns.barplot(x='label', y='api_resolv', data=df2_desc_feature_occ_per_cat)
pyplot.title('Features Occurances',fontsize=20)
pyplot.ylabel('api_resolv', fontsize=12)
pyplot.xlabel('Classes', fontsize=12)
pyplot.xticks(rotation=45)
pyplot.show()


# In[175]:


import seaborn as sns
sns.set()
df2_desc_feature_occ_per_cat.set_index('label').T.plot(kind='bar', stacked=True)


# In[177]:


# Using T stacked plot to visualize the quantity of each feature per class

#plt.figure(figsize=(50,10))
df2_desc_feature_occ_per_cat.set_index('label').T.plot(kind='bar', stacked=True,figsize=(20,6))
pyplot.title('Features Occurances',fontsize=20)
pyplot.ylabel('Occurrences', fontsize=12)
pyplot.xlabel('Features', fontsize=10)
#plt.xticks(rotation=45)
pyplot.show()


# # Experiments and model building

# #### Coding categorical feature values

# In[198]:


import warnings
warnings.filterwarnings("ignore")


# In[185]:


df3 = df2.copy()
df3 = df3.fillna(0)


# In[192]:


#Define a generic function using Pandas replace function
def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded
 
#Coding empty values as 0:
#df5a_select_class.drop(columns=['Unnamed'],axis=1)

df3["label_coded"] = coding(df3["label"],{'APT1':0, 'Crypto':1, 'Locker':2, 'shadowbrokers':3, 'Zeus':4})
print ('Done')

#Drop the categorical label feature
df3 = df3.drop(columns=['label'],axis=1)


# In[193]:


#df3['label_coded'].unique()


# ### Evaluate on ALL data (Unstandardized)

# In[203]:


# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


#Split-out validation dataset
df3_no_label = df3.drop('label_coded',1)
array_ds_full = df3.values
X2 = df3_no_label.copy()
Y2 = df3["label_coded"]
validation_size = 0.20

X_train, X_validation, Y_train, Y_validation = train_test_split(X2, Y2,test_size=validation_size, random_state=seed)


# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2





# Spot-Check Algorithms 7 algo and count times
import time
start = time.time()

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('MLR',LogisticRegression(multi_class='multinomial', solver='newton-cg')))

results = []
names = []
report_msg = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    print(name,"--> is training now... ")
    end = time.time()
    seconds_elapsed = end - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) , time: %f seconds" % (name, cv_results.mean(), cv_results.std(),seconds_elapsed)
    report_msg.append(msg)
print("\nEnd of training.>>> ALL data (Unstandardized) \nModels' accuracies and times used are:")
print("------------------------------------\n")
print ('\n'.join(report_msg))


# ### Evaluate on SELECTED data (Unstandardized)

# In[201]:


#evaluation on selected features
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


#Split-out validation dataset
df3_no_label = df3.drop('label_coded',1)
array_ds_full = df3.values
X2 = df3_no_label.copy()
Y2 = df3["label_coded"]
validation_size = 0.20

X_train, X_validation, Y_train, Y_validation = train_test_split(X2, Y2,test_size=validation_size, random_state=seed)


# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



# feature extraction
select = SelectKBest(score_func=chi2, k=50)
selected_features = select.fit(X2, Y2)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [df3_no_label.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
#X_test_selected = X_test[colnames_selected]



# Spot-Check Algorithms 7 algo and count times
import time
start = time.time()

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('MLR',LogisticRegression(multi_class='multinomial', solver='newton-cg')))

results = []
names = []
report_msg = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train_selected, Y_train, cv=kfold, scoring=scoring)
    print(name,"--> is training now... ")
    end = time.time()
    seconds_elapsed = end - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) , time: %f seconds" % (name, cv_results.mean(), cv_results.std(),seconds_elapsed)
    report_msg.append(msg)
print("\nEnd of training.>>> ALL data (Unstandardized) \nModels' accuracies and times used are:")
print("------------------------------------\n")
print ('\n'.join(report_msg))


# #### Evaluate on ALL data (standardized) --> StandardScaler

# In[204]:


# Standardize the dataset with StandardScaler
import time
start = time.time()

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
pipelines.append(('ScaledMLR', Pipeline([('Scaler', StandardScaler()),('MLR', 
                                                LogisticRegression(multi_class='multinomial', solver='newton-cg'))])))



results = []
names = []
report_msg = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    print(name,"--> is training now... ")
    end = time.time()
    seconds_elapsed = end - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) , time: %f seconds" % (name, cv_results.mean(), cv_results.std(),seconds_elapsed)
    report_msg.append(msg)
print("\nEnd of training.>>> ALL data (standardized) --> StandardScaler \nScaled Models' accuracies and times used are:")
print("------------------------------------\n")
print ('\n'.join(report_msg))


# #### Evaluate on SELECTED data (standardized) --> StandardScaler

# In[206]:


# Standardize the dataset with StandardScaler
import time
start = time.time()

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
pipelines.append(('ScaledMLR', Pipeline([('Scaler', StandardScaler()),('MLR', 
                                                LogisticRegression(multi_class='multinomial', solver='newton-cg'))])))



results = []
names = []
report_msg = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train_selected, Y_train, cv=kfold, scoring=scoring)
    print(name,"--> is training now... ")
    end = time.time()
    seconds_elapsed = end - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) , time: %f seconds" % (name, cv_results.mean(), cv_results.std(),seconds_elapsed)
    report_msg.append(msg)
print("\nEnd of training.>>> ALL data (standardized) --> StandardScaler \nScaled Models' accuracies and times used are:")
print("------------------------------------\n")
print ('\n'.join(report_msg))


# #### Evaluate on ALL data (standardized) --> RobustScaler

# In[207]:


# Standardize the dataset with RobustScaler
import time
start = time.time()

pipelines = []
pipelines.append(('RbScaledLR', Pipeline([('RbScaler', RobustScaler()),('LR', LogisticRegression())])))
pipelines.append(('RbScaledLDA', Pipeline([('RbScaler', RobustScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('RbScaledKNN', Pipeline([('RbScaler', RobustScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('RbScaledCART', Pipeline([('RbScaler', RobustScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('RbScaledNB', Pipeline([('RbScaler', RobustScaler()),('NB', GaussianNB())])))
pipelines.append(('RbScaledSVM', Pipeline([('RbScaler', RobustScaler()),('SVM', SVC())])))
pipelines.append(('RbScaledMLR', Pipeline([('Scaler', StandardScaler()),('MLR', 
                                                LogisticRegression(multi_class='multinomial', solver='newton-cg'))])))

results = []
names = []
report_msg = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    print(name,"--> is training now... ")
    end = time.time()
    seconds_elapsed = end - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) , time: %f seconds" % (name, cv_results.mean(), cv_results.std(),seconds_elapsed)
    report_msg.append(msg)
print("\nEnd of training.>>>ALL data (standardized) --> RobustScaler \nRobust Scaled Models' accuracies and times used are:")
print("------------------------------------\n")
print ('\n'.join(report_msg))


# #### Evaluate on SELECTED data (standardized) --> RobustScaler

# In[208]:


# Standardize the dataset with RobustScaler
import time
start = time.time()

pipelines = []
pipelines.append(('RbScaledLR', Pipeline([('RbScaler', RobustScaler()),('LR', LogisticRegression())])))
pipelines.append(('RbScaledLDA', Pipeline([('RbScaler', RobustScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('RbScaledKNN', Pipeline([('RbScaler', RobustScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('RbScaledCART', Pipeline([('RbScaler', RobustScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('RbScaledNB', Pipeline([('RbScaler', RobustScaler()),('NB', GaussianNB())])))
pipelines.append(('RbScaledSVM', Pipeline([('RbScaler', RobustScaler()),('SVM', SVC())])))
pipelines.append(('RbScaledMLR', Pipeline([('Scaler', StandardScaler()),('MLR', 
                                                LogisticRegression(multi_class='multinomial', solver='newton-cg'))])))

results = []
names = []
report_msg = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train_selected, Y_train, cv=kfold, scoring=scoring)
    print(name,"--> is training now... ")
    end = time.time()
    seconds_elapsed = end - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) , time: %f seconds" % (name, cv_results.mean(), cv_results.std(),seconds_elapsed)
    report_msg.append(msg)
print("\nEnd of training.>>>ALL data (standardized) --> RobustScaler \nRobust Scaled Models' accuracies and times used are:")
print("------------------------------------\n")
print ('\n'.join(report_msg))


# #### Evaluate on selected data (standardized) --> StandardScaler with ANOVA

# In[222]:



# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


X_train, X_validation, Y_train, Y_validation = train_test_split(X2, Y2,test_size=validation_size, random_state=seed)


# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



# feature extraction
select = SelectKBest(f_classif, k=50)
selected_features = select.fit(X2, Y2)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [df3_no_label.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]



# Spot-Check Algorithms 7 algo and count times
import time
start = time.time()



# Standardize the dataset with StandardScaler


pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
pipelines.append(('ScaledMLR', Pipeline([('Scaler', StandardScaler()),('MLR', 
                                                LogisticRegression(multi_class='multinomial', solver='newton-cg'))])))

results = []
names = []
report_msg = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train_selected, Y_train, cv=kfold, scoring=scoring)
    print(name,"--> is training now... ")
    end = time.time()
    seconds_elapsed = end - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) , time: %f seconds" % (name, cv_results.mean(), cv_results.std(),seconds_elapsed)
    report_msg.append(msg)
print("\nEnd of training.>>> selected data (standardized) --> StandardScaler with ANOVA \nScaled Models' accuracies and times used are:")
print("------------------------------------\n")
print ('\n'.join(report_msg))


# #### Evaluate on selected data (standardized) --> RobustScaler with ANOVAÂ¶

# In[221]:


# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


X_train, X_validation, Y_train, Y_validation = train_test_split(X2, Y2,test_size=validation_size, random_state=seed)


# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



# feature extraction
select = SelectKBest(f_classif, k=50)
selected_features = select.fit(X2, Y2)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [df3_no_label.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]

# Standardize the dataset with RobustScaler
import time
start = time.time()

pipelines = []
pipelines.append(('RbScaledLR', Pipeline([('RbScaler', RobustScaler()),('LR', LogisticRegression())])))
pipelines.append(('RbScaledLDA', Pipeline([('RbScaler', RobustScaler()),('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('RbScaledKNN', Pipeline([('RbScaler', RobustScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('RbScaledCART', Pipeline([('RbScaler', RobustScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('RbScaledNB', Pipeline([('RbScaler', RobustScaler()),('NB', GaussianNB())])))
pipelines.append(('RbScaledSVM', Pipeline([('RbScaler', RobustScaler()),('SVM', SVC())])))
pipelines.append(('RbScaledMLR', Pipeline([('Scaler', StandardScaler()),('MLR', 
                                                LogisticRegression(multi_class='multinomial', solver='newton-cg'))])))

results = []
names = []
report_msg = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train_selected, Y_train, cv=kfold, scoring=scoring)
    print(name,"--> is training now... ")
    end = time.time()
    seconds_elapsed = end - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) , time: %f seconds" % (name, cv_results.mean(), cv_results.std(),seconds_elapsed)
    report_msg.append(msg)
print("\nEnd of training.>>>ALL data (standardized) --> RobustScaler \nRobust Scaled Models' accuracies and times used are:")
print("------------------------------------\n")
print ('\n'.join(report_msg))


# ## Evaluation by Standardization (RobustScaler)--> Dimensionality Reducer(ANOVA) --> Ensemble Models

# In[220]:


# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


X_train, X_validation, Y_train, Y_validation = train_test_split(X2, Y2,test_size=validation_size, random_state=seed)


# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



# feature extraction
select = SelectKBest(f_classif, k=50)
selected_features = select.fit(X2, Y2)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [df3_no_label.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]

# Standardize the dataset with RobustScaler
import time
start = time.time()

pipelines = []
pipelines.append(('RbScaledAB', Pipeline([('RbScaler', RobustScaler()),('AB', AdaBoostClassifier())])))
pipelines.append(('RbScaledLGBM', Pipeline([('RbScaler', RobustScaler()),('GBM', GradientBoostingClassifier())])))

pipelines.append(('RbScaledRF', Pipeline([('RbScaler', RobustScaler()),('RF', RandomForestClassifier())])))
pipelines.append(('RbScaledET', Pipeline([('RbScaler', RobustScaler()),('ET', ExtraTreesClassifier())])))


results = []
names = []
report_msg = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train_selected, Y_train, cv=kfold, scoring=scoring)
    print(name,"--> is training now... ")
    end = time.time()
    seconds_elapsed = end - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) , time: %f seconds" % (name, cv_results.mean(), cv_results.std(),seconds_elapsed)
    report_msg.append(msg)
print("\nEnd of training.>>>ALL data (standardized) --> RobustScaler \nRobust Scaled Models' accuracies and times used are:")
print("------------------------------------\n")
print ('\n'.join(report_msg))


# In[219]:


# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


X_train, X_validation, Y_train, Y_validation = train_test_split(X2, Y2,test_size=validation_size, random_state=seed)


# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



# feature extraction
select = SelectKBest(f_classif, k=50)
selected_features = select.fit(X2, Y2)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [df3_no_label.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]




# Standardize the dataset with RobustScaler
import time
start = time.time()


pipelines = []
pipelines.append(('RbScaledAB', Pipeline([('RbScaler', RobustScaler()),('AB', AdaBoostClassifier())])))
pipelines.append(('RbScaledLGBM', Pipeline([('RbScaler', RobustScaler()),('GBM', GradientBoostingClassifier())])))

pipelines.append(('RbScaledRF', Pipeline([('RbScaler', RobustScaler()),('RF', RandomForestClassifier())])))
pipelines.append(('RbScaledET', Pipeline([('RbScaler', RobustScaler()),('ET', ExtraTreesClassifier())])))


results = []
names = []
report_msg = []
matrix_rpt = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train_selected, Y_train, cv=kfold, scoring=scoring)
    model.fit(X_train, Y_train)
    predicted = model.predict(X_validation)
    matrix = confusion_matrix(Y_validation, predicted)
    classification_rpt = classification_report(Y_validation, predicted)
    print(name,"--> is training now... ")
    end = time.time()
    seconds_elapsed = end - start
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) , time: %f seconds,\n\n%s,\n\n%s" % (name, cv_results.mean(), cv_results.std(),seconds_elapsed,matrix,classification_rpt)
    matrix_rpt.append(matrix)
    report_msg.append(msg)
print("\nEnd of training.>>>ALL data (standardized) --> RobustScaler \nRobust Scaled Models' accuracies and times used are:")
print("------------------------------------\n")
print ('\n'.join(report_msg))


# In[224]:


pyplot.figure(figsize = (10,9))
sns.set(font_scale=1.4)
sns.heatmap(matrix,annot=True, square=True,annot_kws={"size": 16})


# ### ROC(AUC) performance of the best model GBMÂ¶

# In[240]:


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

#evaluation on selected features
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


#Split-out validation dataset
df3_no_label = df3.drop('label_coded',1)
array_ds_full = df3.values
X = df3_no_label.copy()
y = df3["label_coded"]
validation_size = 0.20


# Binarize the output
y = label_binarize(y, classes=[0, 1, 2,3,4])
n_classes = y.shape[1]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size,random_state=seed)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(GradientBoostingClassifier())
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

########################################################################################

pyplot.figure()
lw = 3
pyplot.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
pyplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.title('Receiver operating characteristic example')
pyplot.legend(loc="lower right")
pyplot.show()
###############################################################################################

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(10,8))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):
    pyplot.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))


pyplot.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

pyplot.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','greenyellow','green'])

   

pyplot.plot([0, 1], [0, 1], 'k--', lw=lw)
pyplot.xlim([0.0, 1.0])
pyplot.ylim([0.0, 1.05])
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
#pyplot.title('Some extension of Receiver operating characteristic to multi-class')
pyplot.title('Gradient Boosting ROC performance')
pyplot.legend(loc="lower right")
pyplot.show()

