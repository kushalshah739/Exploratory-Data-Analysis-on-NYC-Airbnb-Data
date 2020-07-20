#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #MATRIX COMPUTATION
import numpy as np #LINEAR ALGEBRA COMPUTATION
import seaborn as sns


# In[2]:


train = pd.read_csv('nyc_airbnb.csv')


# In[3]:


train.head(n = 10)


# In[4]:


test_list = train['neighbourhood_group']

res=[]

for i in test_list: 
    if i not in res: 
        res.append(i) 
        
test_list


# In[5]:


res


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',s=20, data=train)

plt.title('Longitude-Latitude of the five neighbourhood groups')

#the below visual is a clear visualisation of the five NYC's popular neighbourhoods


# In[7]:


unwanted_coloumns = ['latitude', 'longitude', 'id', 'host_id', 'reviews_per_month', 'last_review', 'calculated_host_listings_count', 'name', 'host_name']

train = train.drop(unwanted_coloumns, axis = 1, inplace = False)

train.head()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

#sns.catplot(x = "neighbourhood_group", data = train, hue = "price", kind = "count")
#sns.factorplot(data = train, x = "neighbourhood_group", y = "price")
#sns.catplot( x = "neighbourhood_group", y = "price", hue="price", data=train)


# In[9]:


from numpy import median

splot = sns.barplot(x="neighbourhood_group", y="price", data=train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Price of neighbourhood groups')


# In[10]:


splot = sns.barplot(x="neighbourhood_group", y="minimum_nights", data=train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Minimum nights of neighbourhood groups')


# In[ ]:





# In[11]:


train['Fare_Range'] = pd.qcut(train['price'], 4) 
splot = sns.barplot(x = 'neighbourhood_group', hue ='Fare_Range', y ='price', data = train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.title('Fare range of different neighbourhood groups')


# In[ ]:





# In[12]:


splot = sns.barplot(x = 'neighbourhood_group', hue ='room_type', y ='price', data = train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Room type of neighbourhood groups vs Price')


# In[ ]:





# In[13]:


splot = sns.barplot(x = 'neighbourhood_group', hue ='room_type', y ='minimum_nights', data = train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

plt.title('Room type of neighbourhood groups vs Minimum nights')


# In[ ]:





# In[ ]:





# In[14]:


from numpy import median

splot = sns.barplot(x="room_type", y="price", data=train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.title('Room type vs Price')


# In[ ]:





# In[15]:


splot = sns.barplot(x="room_type", y="availability_365", data=train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.title('Room type vs 365 Day room availability')


# In[ ]:





# In[16]:


splot = sns.barplot(x="room_type", y="number_of_reviews", data=train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.xlabel('room type')
plt.ylabel('number of reviews')
plt.title('Room type vs Number of reviews')


# In[ ]:





# In[ ]:





# In[17]:


train.isnull().sum() #no missing values


# In[18]:


#let's see how scatterplot will come out 
sns.scatterplot(x='room_type', y='price', data=train)
plt.title('Room types vs Price')


# In[ ]:





# In[19]:


splot = sns.barplot(x = 'neighbourhood_group', hue ='room_type', y ='availability_365', data = train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.xlabel('neighbourhood group')
plt.ylabel('room type availability')
plt.title('Neighbourhood groups vs Room-type availability')


# In[ ]:





# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))
sns.distplot(train.availability_365, kde=False)
plt.xlim(0,365)
plt.title('Availability in year', fontsize=15)
plt.xlabel('Availibility')
plt.ylabel("Frequency")


# In[21]:


splot = sns.barplot(x="neighbourhood_group", y="availability_365", data=train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.xlabel('neighbourhood group')
plt.ylabel('room availability in 365 days')
plt.title('Neighbourhood groups vs 365-Day Room availability')


# In[ ]:





# In[22]:


train.head()


# In[ ]:





# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt



plt.clf()
splot = train.groupby('neighbourhood_group').sum().plot(kind='bar')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.xlabel('neighbourhood')
plt.ylabel('Total count')
plt.title('Neighbourhood groups vs Total count')

#we can see here that Manhattan generates the most revenue out of all neighbourhoods of $42.65M


# In[24]:



magic = train.groupby('neighbourhood_group')['neighbourhood'].agg(','.join).reset_index()
magic

fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)

splot = train.groupby('neighbourhood_group')['neighbourhood'].nunique().plot(kind='bar', ax = ax)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.xlabel('neighbourhood group')
plt.ylabel('number of neighbourhoods')
plt.title('Number of neighbourhoods in each neighbourhood group')
plt.show()


#splot = sns.barplot(x = 'neighbourhood', y = 'neighbourhood', data = magic)


# In[25]:


splot = sns.barplot(x = 'neighbourhood_group', hue ='room_type', y ='number_of_reviews', data = train)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.title('Neighbourhood group rooms vs Number of reviews')


# In[ ]:





# In[26]:


df = train['neighbourhood'].groupby(train['neighbourhood_group']).sum()
df

res=[]

for i in df: 
    if i not in res: 
        res.append(i) 
        
res

#list of all neighbourhoods, the output below doesnt have spaces and commas as the data has been directly extracted from excel without any formatting.

