#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[3]:


iris=pd.read_csv('Iris.csv')


# In[4]:


print(iris.shape,iris.size)


# In[5]:


iris[['petal_length','petal_width']]


# In[63]:


iris[['sepal_length','sepal_width']]


# In[64]:


iris.columns


# In[65]:


iris[['species']].value_counts()


# In[6]:


#This is 2-D scatter plot of the iris dataset having any of the two features ,here taken petal length and petal width in one colour
iris.plot(kind='scatter',x='petal_length',y='petal_width')
plt.show()


# In[7]:


#This is 2-D scatter plot of the iris dataset having any two features here taken sepallength and sepal width using only one colour
iris.plot(kind='scatter',x='sepal_length',y='sepal_width')
plt.show()


# In[51]:


sns.set_style('whitegrid')
sns.FacetGrid(iris, hue='species').map(plt.scatter,'sepal_length','sepal_width').add_legend()
plt.show()


# In[49]:


sns.pairplot(iris,hue='species')


# In[75]:


iris_setosa=iris.loc[iris['species']=='setosa']
iris_virginica=iris.loc[iris['species']=='virginica']
iris_versicolor=iris.loc[iris['species']=='versicolor']
plt.plot(iris_setosa["petal_length"], np.zeros_like(iris_setosa['petal_length']), 'o')
plt.plot(iris_versicolor["petal_length"], np.zeros_like(iris_versicolor['petal_length']), 'o')
plt.plot(iris_virginica["petal_length"], np.zeros_like(iris_virginica['petal_length']), 'o')
#This is when the y axis is kept at 0 always


# In[87]:


#univariate data analysis but in form of distribution and histogram plots which are more useful
sns.FacetGrid(iris,hue='species').map(sns.distplot,'petal_length').add_legend()
plt.show()
#This gives the distribution plot acccording to one feature only that is petal length


# In[88]:


sns.FacetGrid(iris,hue='species').map(sns.histplot,'petal_width').add_legend()
plt.show()
#univariate data analysis using histogram plot based on one feature only that is petal width


# In[89]:


sns.FacetGrid(iris,hue='species').map(sns.distplot,'sepal_length').add_legend()


# In[90]:


sns.FacetGrid(iris,hue='species').map(sns.histplot,'sepal_width').add_legend()


# In[91]:


sns.FacetGrid(iris,hue='species').map(sns.distplot,'sepal_width').add_legend()
#doing univariate analysis with distribution plot based on one feature only that is sepal width


# In[99]:


#drawing cdf or pdf based on above observations done using univariate data analysis
counts,bin_edges=np.histogram([iris_setosa['petal_length']],bins=10,density=False)
print(counts,sum(counts))
print(bin_edges)
pdf=counts/sum(counts)
print(pdf)
print("The sum of pdf is",sum(pdf))
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf')
plt.plot(bin_edges[1:],cdf,label='cdf')
plt.xlabel("Petal_length_setosa")
plt.ylabel('Probabilities')
plt.legend()
plt.show()


# In[106]:


print("Means:")
print(np.mean(iris_setosa['petal_length']))
print(np.mean(iris_virginica["petal_length"]))
print(np.mean(iris_versicolor["petal_length"]))
print("Standard_deviations:")
print(np.std(iris_setosa['petal_length']))
print(np.std(iris_virginica['petal_length']))
print(np.std(iris_versicolor["petal_length"]))


# In[108]:


print(np.median(iris_setosa['petal_length']))
print(np.median(iris_virginica["petal_length"]))
print(np.median(iris_versicolor["petal_length"]))


# In[113]:


#box plot of the dataset based on the feature petal_length
sns.boxplot(x='species',y='petal_length',data=iris)
plt.show()


# In[112]:


#violin plot of the dataset
sns.violinplot(x='species',y='petal_length',data=iris,size=8)
plt.show()


# In[114]:


sns.boxplot(x='species',y='sepal_length',data=iris)
plt.show()


# In[117]:


sns.violinplot(x='species',y='sepal_width',data=iris,size=8)
plt.show()


# In[ ]:




