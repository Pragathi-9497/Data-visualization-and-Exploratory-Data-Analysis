#!/usr/bin/env python
# coding: utf-8

# # LAB 2 - Introduction to Data Visualisation & EDA

# # Submitted by
#  - Name: Pragathi S P
#  - Register Number: 21122043
#  - Class: 2MscDs

# # Lab Overview

# # Objective
#  - Part A) Create a dataset with 12 features and 3 classes using make_classification.
# 
#    - Name the features F1, F2, ... upto F12.
#    - F1: Can hold two values (True/False)
#    - F2: Can hold two values (Type 1 or Type 2)
#    - F3: Can hold three values (A or B or C)
#    - F4: Can hold three values (HIGH / MEDIUM / LOW)
#    - Features F5 to F12 will have numeric values generated during the creation process.
# 
# - Part B) Use either of the above datasets to show the usage of distplot, jointplot, pairplot, rugplot, catplot, barplot, countplot, violinplot, striplot, swarmplot and facetgrid plots.
# 
#  - Part C) Load the Iris Dataset, and explore - scatterplot, scatter_3d, heatmap, boxplot, kdeplot etc.In both the cases, write your observations on the plot outputs and how it is relevant.

# # Problem Definition
#  - Understang the data clssifications.
#  - Creating dataframe and works on it.
#  - Exploring data graphs and visualizing it.

# # Approach
#  - Importing all libraries which we needed.
#  - Creating a dataset using make_classification.
#  - Using pandas create a dataframe.
#  - And then works on dataframe like changing the column name, changing some column's all values to another using some logic.
#  - Using import libraries plotting all graph and visualizing those.
#  - Import the IRIS dataset and works on it like visualizing some plot.
#  - Complete all the given objectives.

# # CODE

# ## Question Part- A
#  - Create a dataset with 12 features and 3 classes using make_classification.
#    - Name the features F1, F2, ... upto F12.
#    - F1: Can hold two values (True/False)
#    - F2: Can hold two values (Type 1 or Type 2)
#    - F3: Can hold three values (A or B or C)
#    - F4: Can hold three values (HIGH / MEDIUM / LOW)
#    - Features F5 to F12 will have numeric values generated during the creation process.

# ## Importing important libraries

# In[1]:


import sklearn.datasets as sk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# In[2]:


# Creating a dataset
X, y = sk.make_classification(n_samples=1000, 
                              n_features=12, n_informative=3, n_redundant=0, n_classes=3, 
                              n_clusters_per_class=1, weights=None, random_state=1)
print("The samples generated are :\n ",X)
print("-_____________________________")
print("The labels  are:\n ",y)


# In[3]:


# Creating a dataframe
df=pd.DataFrame(X)
df


# In[4]:


# Create a empty dictionary
d={}
# Looping into the column and replace all column name
for i in range(12):
    d[i]='F'+str(i+1)
df.rename(columns = d, inplace = True)
df


# In[5]:


# Lock the F1 column and change the all values of F1
df.loc[df.F1<0,'F1'] = False
df.loc[df.F1>0,'F1'] = True
df


# In[6]:


# Changing the F2 column all values using some logic
df['F2'] = df['F2'].apply(lambda x: 'Type 2' if x<0 else 'Type 1')
df


# In[7]:


# Changing the F3 column all values using some logic
df['F3'] = df['F3'].apply(lambda x: 'A' if x<0 else 'B' if 0<=x<=1 else 'C')
df


# In[8]:


# Changing the F4 column all values using some logic
df['F4'] = df['F4'].apply(lambda x: 'LOW' if x<0 else 'MEDIUM' if 0<=x<=1 else 'HIGH')
df


# ## Question Part- B
#  - Using above datasets to show the usage of distplot, jointplot, pairplot, rugplot, catplot, barplot, countplot, violinplot, striplot, swarmplot and facetgrid plots.

# ## Distplot

# In[9]:


sns.distplot(df['F8'])


#  - We use distplot to see how data is distributed for one column only.

# ## Jointplot
# 
# ####  - It works for numeric data to compare two columns and how the data is distributed through the structure.

# In[10]:


sns.jointplot(data=df,x='F6',y='F11')


# ## Heatmap
# #### - It shows the correlation between the two different columns with intensity as third column.

# In[11]:


fig=plt.figure(figsize=(10,6))
data = df.pivot('F6','F8','F12')
sns.heatmap(data.head(3))


# In[12]:


fig=plt.figure(figsize=(9,6))
sns.heatmap(df.corr(),annot=True)

 # It shows the correlation between each of the two columns.


# ## Boxplot

# In[13]:


sns.boxplot(x="F6", y="F12", data=df)


#  - Here we can use categorical variable and can see other columns based on that varible.

# ## Rugplot

# In[14]:


sns.kdeplot(data=df, x="F7")
sns.rugplot(data=df, x="F11")


#  - We can add another graph on it so that we can compare that graph with two differnt types.

# ## Countplot

# In[15]:


sns.countplot(x="F3", data=df)


#  - We can see count of categorical column here in hist format.

# ## Violinplot

# In[16]:


sns.violinplot(x=df["F9"],y=df["F3"])


#  - We can see how data is spread through out the axis in both the dimension.

# ## Striplot

# In[17]:


sns.stripplot(x="F3", y="F12", data=df)


#  - We can see three scatter plot with respect three categories.

# ## Swarmplot

# In[18]:


sns.swarmplot(x="F3", y="F9", data=df)


#  - We can see how data is spread for different categories.

# ## Facetgrid

# In[19]:


graph = sns.FacetGrid(df, col ="F3", hue="F2")
# map the above form facetgrid with some attributes
graph.map(plt.scatter, "F9", "F8", edgecolor ="w")
# show the object
plt.show()


#  - The different types of data has been grouped to understand the data.

# ## Barplot

# In[20]:


sns.barplot('F4','F1',data=df)


# In[21]:


sns.barplot('F3','F5',data=df)


#  - We can compare different columns including categorical column to plot in bar plot pattern and see how the data is distributed.

# # Question- Part C
#  - Load the Iris Dataset, and explore - scatterplot, scatter_3d, heatmap, boxplot, kdeplot etc.In both the cases, write your observations on the plot outputs and how it is relevant.

# In[22]:


import sklearn as sk


# In[23]:


from sklearn.datasets import load_iris


# In[24]:


IRIS = load_iris(as_frame=True) # Load Iris dataset
IRIS


# In[25]:


IRIS.feature_names


# In[26]:


IRIS.target


# In[27]:


IRIS.data


# ## Scatterplot

# In[28]:


# Selecting the figure size and create a scatterplot of sepal length and sepal width
fig=plt.figure(figsize=(4,4))
plt.scatter('sepal width (cm)','sepal length (cm)',data=IRIS.data)
plt.xlabel('\nsepal width')
plt.ylabel('sepal length')


# In[47]:


fig=plt.figure(figsize=(6,4))
plt.scatter('petal width (cm)','petal length (cm)',color='pink',data=IRIS.data)
plt.xlabel('\npetal wi')
plt.ylabel('petal width')


# ## Barplot

# In[48]:


fig=plt.figure(figsize=(6,4))
sns.barplot('sepal length (cm)','sepal width (cm)',data=IRIS.data)


# In[51]:


fig=plt.figure(figsize=(6,4))
sns.barplot('petal length (cm)','petal width (cm)',color='orange',data=IRIS.data)


# ## Histoplot

# In[53]:


fig=plt.figure(figsize=(6,4))
sns.histplot(x='sepal length (cm)',color='green',data=IRIS.data)


# In[55]:


fig=plt.figure(figsize=(6,4))
sns.histplot(x='sepal length (cm)',color='pink',data=IRIS.data)


# In[56]:


fig=plt.figure(figsize=(6,4))
sns.histplot(x='sepal length (cm)',color='green',data=IRIS.data,kde=True)


# In[57]:


fig=plt.figure(figsize=(6,4))
sns.histplot(x='sepal width (cm)',color='g',data=IRIS.data,kde=True)


# ## KDEplot

# In[60]:


fig=plt.figure(figsize=(6,4))
sns.kdeplot(x='petal length (cm)',color='red',data=IRIS.data)


# In[62]:


fig=plt.figure(figsize=(6,4))
sns.kdeplot(x='petal width (cm)',color='r',data=IRIS.data)


# In[64]:


fig=plt.figure(figsize=(6,4))
sns.kdeplot(x='sepal length (cm)',color='yellow',data=IRIS.data)


# In[65]:


fig=plt.figure(figsize=(6,4))
sns.kdeplot(x='sepal width (cm)',color='yellow',data=IRIS.data)


# ## Boxplot

# In[66]:


fig=plt.figure(figsize=(6,4))
IRIS.data.boxplot()


# ## Heatmap

# In[69]:


fig=plt.figure(figsize=(9,6))
sns.heatmap(IRIS.data.corr(),annot=True)


# ## Scatter_3d

# In[42]:


get_ipython().system('pip install plotly==5.6.0')


# In[43]:


import plotly.express as px


# In[74]:


fig = px.scatter_3d(IRIS.data, x='sepal length (cm)', y='sepal width (cm)', z='petal width (cm)')
fig.show()


# ## Lineplot

# In[45]:


sns.lineplot(data=IRIS['data'], x=IRIS['data']['petal length (cm)'], y=IRIS['data']['sepal length (cm)'])


# ## Pieplot

# In[46]:


x = np.array([IRIS['data']['sepal length (cm)'].mean(),IRIS['data']['sepal width (cm)'].mean(),IRIS['data']['petal length (cm)'].mean(),IRIS['data']['petal width (cm)'].mean()])
mylabels = IRIS['data'].columns

plt.pie(x, labels = mylabels, startangle = 90)
plt.show() 


#  - Here we are plotting a pie chart for the mean of different columns present in the dataset.
#  - We can see that sepal leanth has maximum mean and petal width has minimum width.

# # Conclusion
#  - Using pandas we can create a dataframe.
#  - loc is label-based, which means that we have to specify rows and columns based on their row and column labels.
#  
# #### - Matplotlib library
#    - Using scatter plot we can see a diagram where each value in the data set is represented by a dot.
#    - Scatter_3d plots are used to plot data points on three axes in the attempt to show the relationship between three variables.
#    - Using plt.pie we create a pie chart representing the data in an array.
#    
# #### - Seaborn library
#    - We use distplot to see how data is distributed for one column only.
#    - Jointplot works for numeric data to compare two columns and how the data is distributed through the structure.
#    - Heatmap shows the correlation between the two different columns.
#    - In boxplot we can use categorical variable and can see other columns based on that varible.
#    - In rugplot we can add another graph on it so that we can compare that graph with two differnt types.
#    - In countplot we can see count of categorical column here in hist format.
#    - Using violinplot we can see how data is spread through out the axis in both the dimension.
#    - Using stirplot we can see two scatter plot with respect two categories.
#    - Using swarmplot we can see how data is spread for different categories.
#    - In facetgrid the different types of data has been grouped to understand the data.
#    - We can compare different columns including categorical column to plot in barplot pattern and see how the data is distributed.
#    - Using histogram we visualize that it represents the distribution of one or more variables by counting the number of observations that fall within disrete bins.
#    - Kdeplot is a Kernel Distribution Estimation Plot which depicts the probability density function of the continuous or non-parametric data variables i.e. we can plot for the univariate or multiple variables altogether.
#    - Using lineplot we can see the posibility of several semantic groups.
# 

# # References
#  - https://seaborn.pydata.org/examples/index.html
#  - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
#  - https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/
#  - https://www.w3schools.com/python/matplotlib_pyplot.asp
#  - https://www.geeksforgeeks.org/introduction-to-seaborn-python/
# 
