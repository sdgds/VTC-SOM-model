#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 4/16/2020
BrainSOM mapping AI to HumanBrain 
@author: Zhangyiyuan
"""
import copy
import PIL.Image as Image
import sys
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom
from scipy.integrate import odeint
from scipy.stats import zscore
from warnings import warn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import minisom
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform



def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None):
    iterations = np.arange(num_iterations) % data_len
    if random_generator:
        random_generator.shuffle(iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations

def _wrap_index__in_verbose(iterations):
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    sys.stdout.write(progress)
    beginning = time()
    sys.stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        sys.stdout.write(progress)

def fast_norm(x):
    return np.sqrt(np.dot(x, x.T))




class VTCSOM(minisom.MiniSom):
    
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=1,
                 neighborhood_function='gaussian'):
        """
        x : int
            x dimension of the feature map.
        y : int
            y dimension of the feature map.
        input_len : int
            Number of the elements of the vectors in input.
        sigma : float
            Spread of the neighborhood function (sigma(t) = sigma / (1 + t/T) where T is num_iteration/2)
        learning_rate : 
            initial learning rate (learning_rate(t) = learning_rate / (1 + t/T)
        neighborhood_function : function, optional (default='gaussian')
            possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'
        """
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = np.random.RandomState(0)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        
        # random initialization W
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)

        self._x = x
        self._y = y
        self._activation_map = np.zeros((x, y))
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)  # used to evaluate the neighborhood function
        self._xx, self._yy = np.meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)

        neig_functions = {'gaussian': self._gaussian,
                          'exp_decay': self._exp_decay,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle,
                          'circle': self._circle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and divmod(sigma, 1)[1] != 0:
            warn('sigma should be an integer when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]
        
        
    def Normalize_X(self, x):
        temp = np.sum(np.multiply(x, x))
        x /= np.sqrt(temp)
        return x
    
    def Normalize_W(self):
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)

    
    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*sigma*sigma
        ax = np.exp(-np.power(self._xx-self._xx.T[c], 2)/d)
        ay = np.exp(-np.power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T
    
    def _exp_decay(self, c, lamda):
        ax = np.exp(-lamda*np.abs(self._xx-self._xx.T[c]))
        ay = np.exp(-lamda*np.abs(self._yy-self._yy.T[c]))
        return (ax * ay).T
    
    def _mexican_hat(self, c, exc_sigma=2, exc_mag=1, inh_sigma=4, inh_mag=0.8):
        d1 = 2*np.pi*exc_sigma*exc_sigma
        d2 = 2*np.pi*inh_sigma*inh_sigma
        ax_exc = exc_mag * np.exp(-np.power(self._neigx-c[0], 2)/d1)
        ay_exc = exc_mag * np.exp(-np.power(self._neigy-c[1], 2)/d1)
        exc = np.outer(ax_exc, ay_exc)
        ax_inh = inh_mag * np.exp(-np.power(self._neigx-c[0], 2)/d2)
        ay_inh = inh_mag * np.exp(-np.power(self._neigy-c[1], 2)/d2)
        inh = np.outer(ax_inh, ay_inh)       
        return exc - inh
    
    def _circle(self, c, radius):
        circle = np.zeros((self._x, self._y))
        for i in range(self._x):
            for j in range(self._y):
                d = np.sqrt((i-c[0])**2+(j-c[1])**2)
                if d<=radius:
                    circle[i,j] = 1
        return circle
    
    def plot_3D_constrain(self, Z):
        fig = plt.figure(figsize=(10,8))
        ax = plt.axes(projection='3d')
        xx = np.arange(0,self._x,1)
        yy = -np.arange(-self._y,0,1)
        X, Y = np.meshgrid(xx, yy)
        surf = ax.plot_surface(X,Y,Z, cmap='jet')
        ax.set_zlim3d(0)
        fig.colorbar(surf)
        plt.show()
    
    
    
    """ Train Model """  
    ###########################################################################
    ###########################################################################
    ### Training functions
    def Train(self, data, num_iteration, step_len, verbose):
        """Trains the SOM.
        data : np.array Data matrix (sample numbers, feature numbers).
        num_iteration : Maximum number of iterations.
        """            
        start_num = num_iteration[0]
        end_num = num_iteration[1]
        random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), end_num-start_num,
                                              verbose, random_generator)
        q_error = np.array([])
        for t, iteration in enumerate(tqdm(iterations)):
            t = t + start_num
            self.update(data[iteration], 
                        self.winner(data[iteration]), 
                        t, end_num) 
            if (t+1) % step_len == 0:
                q_error = np.append(q_error, np.abs(self.change).sum())
        if verbose:
            print('\n quantization error:', self.quantization_error(data))
        return q_error

    
    
    ### Avtivation and select winner
    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        x = self.Normalize_X(x)
        s = np.subtract(x, self._weights)  # x - w
        self._activation_map = np.linalg.norm(s, axis=-1)

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map
    
    def winner(self, x, k=0):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return np.unravel_index(self._activation_map.reshape(-1).argsort()[k],
                                self._activation_map.shape)
        
    def activation_response(self, data, k=0):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = np.zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x, k)] += 1
        return a
    
    
    
    ### Updating function
    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        # structual constrain
        def asymptotic_decay(scalar, t, max_iter):
            return scalar / (1+t/(max_iter/2))
        eta = asymptotic_decay(self._learning_rate, t, max_iteration)
        g = self.neighborhood(win, self._sigma) * eta
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
        self.Normalize_W()
        
        
        
        
        