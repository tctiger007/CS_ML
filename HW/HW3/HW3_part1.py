#!/usr/bin/env python
# coding: utf-8

# In[19]:


#ALL OF THE FOLLOWING VARIABLES ARE SUGGESTIONS, YOU MAY NOT NEED TO USE ALL OF THESE
#ALSO, YOU MAY ADD MORE VARIABLES AS MAKE SENSE

import numpy as np
np.random.seed(100)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork(object):
#sizes = list showing the number of nodes/layer and number of layers. For example, 
#[2,5] means 2 hidden layers and first hidden layer has 2 nodes and the second hidden layer has 5 nodes
    def __init__(self, sizes):
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y,1) for y in sizes[1:]]
        self.weights = [np.random.rand(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    def backprop(self,x,y):
        #gradient descent layer by layer 
        gdLbL_b = [np.zeros(b.shape) for b in self.biases]
        gdLbL_w = [np.zeros(w.shape) for w in self.weights]
        #activation function 
        a = x 
        a_s = [x]
        zs = []
        for b, w in zip(self.biases,self.weights):
            z = np.dot(w,a)+b   ############
            zs.append(z)
            a = sigmoid(z)
            a_s.append(a)
        #delta in backpropagation
        d = (a_s[-1]-y) * sigmoid_derivative(zs[-1])
        gdLbL_b[-1] = d
        gdLbL_w[-1] = np.dot(d, a_s[-2].transpose())
        for i in range(2,self.numLayers):
            z=zs[-i]
            d = np.dot(self.weights[-i+1].transpose(),d) * sigmoid_derivative(z)
            gdLbL_b[-i] = d
            gdLbL_w[-i] = np.dot(d,a[-i-1].transpose())
        return (gdLbL_b,gdLbL_w)

    def train(self,training_data, learningRate=0.001, maxIterations=10000, tol=0.0001):
        #The training_data is a list of tuples (x, y)
        gdLbL_b = [np.zeros(b.shape) for b in self.biases]
        gdLbL_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in training_data:
            d_gdLbL_b, d_gdLbL_w = self.backprop(x, y)
            gdLbL_b = [nb+dnb for nb, dnb in zip(gdLbL_b, d_gdLbL_b)]
            gdLbL_w = [nw+dnw for nw, dnw in zip(gdLbL_w, d_gdLbL_w)]
        for w, nw in zip(self.weights, gdLbL_w):
            self.weights = w-(learningRate/len(training_data))*nw 
        if self.weights[maxIterations] - self.weights[maxIterations-1] > tol:
            print('Hit the max iterations')
        self.biases = [b-(learningRate/len(training_data))*nb for b, nb in zip(self.biases, gdLbL_b)]

