#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def safelog(x):
    return(np.log(x + 1e-100))


# In[2]:


np.random.seed(421)
# mean parameters
class_means = np.array([[+0.0, +2.5], 
                        [-2.5, -2.0], 
                        [+2.5, -2.0]])
# covariance parameters
class_covariances = np.array([[[+3.2, +0.0], 
                               [+0.0, +1.2]],
                              [[+1.2, -0.8], 
                               [-0.8, +1.2]],
                              [[+1.2, +0.8], 
                               [+0.8, +1.2]]])
# sample sizes
class_sizes = np.array([120, 90, 90])


# In[3]:


# generate random samples
points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])
X = np.vstack((points1, points2, points3))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))


# In[4]:


# write data to a file
np.savetxt("hw1_data_set.csv", np.hstack((X, y[:, None])), fmt = "%f,%f,%d")


# In[5]:


# plot data points generated
plt.figure(figsize = (10, 10))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[6]:


# read data into memory
data_set = np.genfromtxt("hw1_data_set.csv", delimiter = ",")

# get X and y values
X = data_set[:,[0, 1]]
y_truth = data_set[:,2].astype(int)

# get number of classes and number of samples
K = np.max(y_truth)
N = data_set.shape[0]

# one-of-K encoding
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth - 1] = 1


# In[7]:


# calculate sample means
sample_means = [np.mean(X[y == (c + 1)],axis=0) for c in range(K)]
print(sample_means)


# In[8]:


#calculate covariances
sample_covariances = [np.cov(np.transpose(X[y==c+1])) for c in range(K)]
print(sample_covariances)


# In[9]:


# calculate prior probabilities
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print(class_priors)


# In[10]:


# define the softmax function
def softmax(X, W, w0):
    scores = np.matmul(np.hstack((X, np.ones((N, 1)))), np.vstack((W, w0)))
    scores = np.exp(scores - np.repeat(np.amax(scores, axis = 1, keepdims = True), K, axis = 1))
    scores = scores / np.repeat(np.sum(scores, axis = 1, keepdims = True), K, axis = 1)
    return(scores)


# In[11]:


def gc(X,i):
    Wc=[-0.5*np.linalg.inv(sample_covariances[c]) for c in range(K)]
    wc=[np.matmul(np.linalg.inv(sample_covariances[c]),sample_means[c]) for c in range(K)]
    wc0=[
        -0.5*np.matmul(np.matmul(np.transpose(sample_means[c]), np.linalg.inv(sample_covariances[c])), sample_means[c])
         + -0.5*np.log(np.linalg.det(sample_covariances[c]))
        + np.log(class_priors[c]) 
        for c in range(K)]
    result =np.matmul(np.matmul(np.transpose(X), Wc[i]), X) + np.matmul(np.transpose(wc[i]), X) + wc0[i]
    return result
    #print(result)


# In[12]:


def gradient_W(X, y_truth, y_predicted):
    return(np.asarray([-np.sum(np.repeat((Y_truth[:,c] - Y_predicted[:,c])[:, None], X.shape[1], axis = 1) * X, axis = 0) for c in range(K)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum(Y_truth - Y_predicted, axis = 0))


# In[13]:


# set learning parameters
eta = 0.01
epsilon = 1e-3


# In[14]:


np.random.seed(421)
W = np.random.uniform(low = -0.01, high = 0.01, size = (X.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))


# In[15]:


# learn W and w0 using gradient descent
iteration = 1
objective_values = []
while 1:
    
    Y_predicted = softmax(X, W, w0)
#    for i in range(K):
#        for j in range(3):
#            Y_predicted[i]=  gc(X[i,:],j)

    objective_values = np.append(objective_values, -np.sum(Y_truth * safelog(Y_predicted)))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(X, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break

    iteration = iteration + 1
#print(W)
#print(w0)
print(Y_predicted)


# In[16]:


plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[17]:


y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[18]:


# evaluate discriminant function on a grid
x1_interval = np.linspace(-8, +8, 1201)
x2_interval = np.linspace(-8, +8, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
for c in range(K):
    discriminant_values[:,:,c] = W[0, c] * x1_grid + W[1, c] * x2_grid + w0[0, c]

A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C

plt.figure(figsize = (10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize = 10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize = 10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize = 10)
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[ ]:





# In[ ]:




