#"!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import missingno
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')









# In[5]:


dataset = pd.read_csv('C:/Users/spoor/Downloads/Dataset/creditcard.csv')


# In[6]:


dataset.head(10)


# In[5]:


dataset = pd.read_csv('C:/Users/spoor/Downloads/Dataset/creditcard.csv')


# In[7]:


dataset.shape


# In[8]:


dataset.info()


# In[10]:


missingno.matrix(dataset)


# In[11]:


dataset.describe()


# In[12]:


def diagnostic_plots(df, variable):
    plt.figure(figsize = (16, 4))
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins = 30)
    plt.title('Histogram')
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist = "norm", plot = plt)
    plt.ylabel('Variable quantiles')
    plt.subplot(1, 3, 3)
    sns.boxplot(y = df[variable])
    plt.title('Boxplot')

    plt.show()


# In[13]:


diagnostic_plots(dataset, 'Avg_Credit_Limit')


# In[14]:


diagnostic_plots(dataset, 'Total_Credit_Cards')


# In[15]:


diagnostic_plots(dataset, 'Total_visits_bank')


# In[16]:


diagnostic_plots(dataset, 'Total_visits_online')


# In[17]:


diagnostic_plots(dataset, 'Total_calls_made')


# In[18]:


dataset.drop(['Sl_No'], axis = 1, inplace = True)


# In[19]:


dataset.head(10)


# In[20]:


def find_skewed_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary


# In[21]:


credit_upper_limit, credit_lower_limit = find_skewed_boundaries(dataset, 'Avg_Credit_Limit', 1.5)
credit_upper_limit, credit_lower_limit


# In[22]:


online_upper_limit, online_lower_limit = find_skewed_boundaries(dataset, 'Total_visits_online', 1.5)
online_upper_limit, online_lower_limit


# In[23]:


dataset['Avg_Credit_Limit']= np.where(dataset['Avg_Credit_Limit'] > credit_upper_limit, credit_upper_limit,
                       np.where(dataset['Avg_Credit_Limit'] < credit_lower_limit, credit_lower_limit, 
                                dataset['Avg_Credit_Limit']))


# In[24]:


dataset['Total_visits_online']= np.where(dataset['Total_visits_online'] > online_upper_limit, online_upper_limit,
                       np.where(dataset['Total_visits_online'] < online_lower_limit, online_lower_limit,
                                dataset['Total_visits_online']))


# In[25]:


diagnostic_plots(dataset, 'Avg_Credit_Limit')


# In[26]:


diagnostic_plots(dataset, 'Total_visits_online')


# In[27]:


X = dataset[['Avg_Credit_Limit', 'Total_visits_bank', 'Total_Credit_Cards']].iloc[:, :].values


# In[28]:


#elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker = 'o')
plt.title('The Elbow Method')
plt.xlabel('clusters')
plt.ylabel('WCSS')
plt.show()


# In[29]:


#Training the K-Means model 

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# In[36]:


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s = 5 , color = 'blue', 
           label = "Average Credit Limit customers")
ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s = 5 , color = 'orange', 
           label = "Low Credit Limit customers")
ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s = 5 , color = 'green', 
           label = "High Credit Limit customers")
plt.title('Segmentation using Average Credit Limit, Total Credit cards, and Total bank visits')
ax.set_xlabel('Average Credit Limit')
ax.set_ylabel('Total visits bank')
ax.set_zlabel('Total credit cards')
ax.legend()
plt.show()


# In[32]:


#K Means Model
silhouette_score_kmeans = round(silhouette_score(X, y_kmeans), 2)
calinski_harabasz_score_kmeans = round(calinski_harabasz_score(X, y_kmeans), 2)

print('Silhouette Score : {}'.format(silhouette_score_kmeans))
print('Calinski Harabasz Score : {}'.format(calinski_harabasz_score_kmeans))


# In[33]:


X = dataset[['Avg_Credit_Limit', 'Total_visits_bank', 'Total_Credit_Cards']].iloc[:, :].values


# In[34]:


dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[38]:


#Training the Hierarchial clustering model 

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# In[39]:


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], X[y_hc == 0, 2], s = 5 , color = 'blue',
           label = "Average - High Credit Limit customers")
ax.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], X[y_hc == 1, 2], s = 5 , color = 'orange',
           label = "Low Credit Limit customers")
ax.set_xlabel('Average Credit Limit')
ax.set_ylabel('Total visits bank')
ax.set_zlabel('Total credit cards')
ax.legend()
plt.show()


# In[69]:


#Hierarchial Model
silhouette_score_hc = round(silhouette_score(X, y_hc), 2)
calinski_harabasz_score_hc = round(calinski_harabasz_score(X, y_hc), 2)
print('Silhouette Score : {}'.format(silhouette_score_hc))
print('Calinski Harabasz Score : {}'.format(calinski_harabasz_score_hc))


# In[38]:


#results from kmeans and hierarchial clustering models

table = []
print('Segmentation using Average Credit Limit, Total Credit cards, and Total bank visits')
table.append(['S.No', 'Clustering Model', 'Silhouette Score', 'Calinski Harabasz Score'])
table.append([1, 'K - Means clustering', silhouette_score_kmeans, calinski_harabasz_score_kmeans])
table.append([2, 'Hierarchial clustering', silhouette_score_hc, calinski_harabasz_score_hc])
print(tabulate(table, headers = 'firstrow', tablefmt = 'fancy_grid'))


# In[39]:


X = dataset[['Avg_Credit_Limit', 'Total_visits_online', 'Total_Credit_Cards']].iloc[:, :].values


# In[40]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker = 'o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[43]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# In[68]:


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s = 5 , color = 'blue', 
           label = "Average Credit Limit customers")
ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s = 5 , color = 'orange', 
           label = "Low Credit Limit customers")
ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s = 5 , color = 'green', 
           label = "High Credit Limit customers")
plt.title('Segmentation using Average Credit Limit, Total Credit cards, and Total bank visits')
ax.set_xlabel('Average Credit Limit')
ax.set_ylabel('Total visits online')
ax.set_zlabel('Total credit cards')
ax.legend()
plt.show()


# In[45]:


silhouette_score_kmeans = round(silhouette_score(X, y_kmeans), 2)
calinski_harabasz_score_kmeans = round(calinski_harabasz_score(X, y_kmeans), 2)

print('Silhouette Score : {}'.format(silhouette_score_kmeans))
print('Calinski Harabasz Score : {}'.format(calinski_harabasz_score_kmeans))


# In[46]:


X = dataset[['Avg_Credit_Limit', 'Total_visits_online', 'Total_Credit_Cards']].iloc[:, :].values


# In[47]:


dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[48]:


hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# In[49]:


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], X[y_hc == 0, 2], s = 5 , color = 'blue',
           label = "Average to High Credit Limit customers")
ax.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], X[y_hc == 1, 2], s = 5 , color = 'orange',
           label = "Low Credit Limit customers")
ax.set_xlabel('Average Credit Limit')
ax.set_ylabel('Total visits online')
ax.set_zlabel('Total credit cards')
ax.legend()
plt.show()


# In[50]:


silhouette_score_hc = round(silhouette_score(X, y_hc), 2)
calinski_harabasz_score_hc = round(calinski_harabasz_score(X, y_hc), 2)
print('Silhouette: {}'.format(silhouette_score_hc))
print('Calinski Harabasz: {}'.format(calinski_harabasz_score_hc))


# In[51]:


table = []
print('Segmentation using Average Credit Limit, Total Credit cards, and Total visits online')
table.append(['S.No', 'Clustering Model', 'Silhouette Score', 'Calinski Harabasz Score'])
table.append([1, 'K - Means clustering', silhouette_score_kmeans, calinski_harabasz_score_kmeans])
table.append([2, 'Hierarchial clustering', silhouette_score_hc, calinski_harabasz_score_hc])
print(tabulate(table, headers = 'firstrow', tablefmt = 'fancy_grid'))


# In[52]:


#calculating recency
recency_data = pd.DataFrame()
recency_data['Customer Key'] = dataset['Customer Key']
recency_data['Recency'] = [10] * len(recency_data)
recency_data = recency_data.drop_duplicates(subset = "Customer Key")


# In[53]:


recency_data


# In[54]:


#calculating frequency

frequency_data = pd.DataFrame()
frequency_data['Customer Key'] = dataset['Customer Key']
frequency_data['Frequency'] = dataset['Total_visits_bank'] + dataset['Total_visits_online'] + dataset['Total_calls_made']
frequency_data = frequency_data.drop_duplicates(subset = "Customer Key")


# In[55]:


frequency_data


# In[56]:


#calculating monetary

monetary_data = pd.DataFrame()
monetary_data['Customer Key'] = dataset['Customer Key']
monetary_data['Monetary'] = dataset['Avg_Credit_Limit']
monetary_data = monetary_data.drop_duplicates(subset = "Customer Key")


# In[57]:


monetary_data


# In[58]:


recency_frequency_data = recency_data.merge(frequency_data, on = 'Customer Key')
merged_data = recency_frequency_data.merge(monetary_data, on = 'Customer Key')


# In[59]:


merged_data


# In[60]:


merged_data['R_rank'] = merged_data['Recency'].rank(ascending = False)
merged_data['F_rank'] = merged_data['Frequency'].rank(ascending = True)
merged_data['M_rank'] = merged_data['Monetary'].rank(ascending = True)


# In[61]:


merged_data


# In[62]:


merged_data['R_rank_norm'] = (merged_data['R_rank'] / merged_data['R_rank'].max())*100
merged_data['F_rank_norm'] = (merged_data['F_rank'] / merged_data['F_rank'].max())*100
merged_data['M_rank_norm'] = (merged_data['F_rank'] / merged_data['M_rank'].max())*100
 
merged_data.drop(columns = ['R_rank', 'F_rank', 'M_rank'], inplace = True)


# In[63]:


merged_data


# In[64]:


merged_data['RFM Score'] = 0.15 * merged_data['R_rank_norm'] + 0.28 * merged_data['F_rank_norm'] + 0.57 * merged_data['M_rank_norm']
merged_data['RFM Score'] *= 0.05
final_data = merged_data[['Customer Key', 'RFM Score']]


# In[65]:


final_data.head(10)


# In[66]:


final_data["Customer Segment"] = np.where(final_data['RFM Score'] > 4.5, "Top Customers", 
                                           (np.where(final_data['RFM Score'] > 4, "High value Customer", 
                                                     (np.where(final_data['RFM Score'] > 3, "Medium Value Customer", 
                                                               np.where(final_data['RFM Score'] > 1.6, 'Low Value Customers', 
                                                                        'Lost Customers'))))))
final_data


# In[67]:


plt.figure(figsize = (6, 6))
plt.pie(final_data['Customer Segment'].value_counts(), labels = final_data['Customer Segment'].value_counts().index,
        autopct = '%.0f%%')
plt.title('Customer Segments based on RFM Score')
plt.show()


# In[ ]:





# In[ ]:




