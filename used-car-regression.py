#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[2]:


# read in data
data = pd.read_csv('car details v4.csv')


# In[3]:


cols = ['Price','Year', 'Kilometer','Fuel Tank Capacity']
data = data[cols].dropna()


# In[4]:


# select features and target variable
X_train, y_train = np.array(data[['Year','Kilometer','Fuel Tank Capacity']]), np.array(data['Price'])


# In[5]:


# begin exploratory data analysis
print('X_train.shape: ',X_train.shape)
print('y_train shape: ', y_train.shape)


# In[6]:


def get_feat_ranges(X):
    for i in range(X.shape[1]):
        print(np.min(X[:,i]), '<= X{} <= '.format(i+1), np.max(X[:,i]))
        print('\n')
    print('max distance: ', np.ptp(X, axis=0))


# In[7]:


# compare ranges of features to determine if scaling is necessary
get_feat_ranges(X_train)


# In[8]:


# observe relationships between X and y
# NOTE: these relationships indicate this model will not be accurate. Data is highly skewed to one direction and correlation
# between X1, X2, and price is weak
X_labels = ['Year','Kilometer','Fuel Tank Capacity']
fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_labels[i])
ax[0].set_ylabel('Price')
plt.show()


# In[9]:


def get_corrcoef(X, y):
    w_init = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        print(np.corrcoef(X[:,i], y)[0,1])


# In[10]:


get_corrcoef(X_train, y_train)


# In[11]:


# normalize features to put everything on similar scale
def zscore_normalization(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X-mu)/sigma
    
    return (X_norm, mu, sigma)


# In[12]:


X_norm, mu, sigma = zscore_normalization(X_train)
get_feat_ranges(X_norm)


# In[13]:


# calculate cost (squared difference between predictions and actual values)
def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost


# In[14]:


# NOTE: cost is extremely high due to the fact this data is potentially unfit for this prediction, as well as the fact
# feature values are very high (i.e. kilometers driven can be up to 200,000)
cost = compute_cost(X_train, y_train, 0, 0)
cost


# In[15]:


# determine gradient of loss function, will be essential for calculating gradient descent
def compute_gradient(X, y, w, b): 
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


# In[16]:


# repeatedly update parameters w1, w2, w3, and b with values that eventually converge on the absolute minimum of the cost 
# function
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):

        dj_db,dj_dw = gradient_function(X, y, w, b)

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db              
      
        if i<100000:    
            J_history.append( cost_function(X, y, w, b))

        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history


# In[17]:


# determine learning rate that will allow model to w to converge without overshooting cost function minimum
alpha = .00067
b_init = 0.
w_final, b_final, J_hist = gradient_descent(X_norm, y_train, [0,0,0], 0, compute_cost, compute_gradient, alpha, 10000)


# In[18]:


# values of w that provide most optimal prediction of y
w_final


# In[19]:


# visualize cost function (total and initial iterations)
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[8000:10000])), J_hist[8000:10000])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()


# In[20]:


# view top 3 predictions
for i in range(3):
    print('prediction: ' , np.dot(X_norm[i], w_final) + b_final)
    print('actual: ', y_train[i])


# In[21]:


# conclusions:
# the data used for this prediction may not be the best choice to predict price as is.
# as seen in the scatter plots showing the relationship between features and y, the distributions are not promising for
# a clean output, especially the kilometer feature.
# improvements could be made to the model by restricting the ranges of the features so that there are not as many outliers.
# new features should be introduced instead of or in addition to existing features, ideally with more normal distributions
# and stronger relationships to the target variable

