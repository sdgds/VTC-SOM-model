#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.stats import zscore
import scipy.stats as stats
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import BrainSOM
import copy



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])}

alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()

def cohen_d(x1, x2):
    s1 = x1.std()
    return (x1.mean()-x2)/s1

def Functional_map_pca(som, pca, pca_index): 
    class_name = ['face', 'place', 'body', 'object']
    f1 = os.listdir("/data/HCP_WM/" + 'face')
    if '.DS_Store' in f1:
        f1.remove('.DS_Store')
    f2 = os.listdir("/data/HCP_WM/" + 'place')
    if '.DS_Store' in f2:
        f2.remove('.DS_Store')
    f3 = os.listdir("/data/HCP_WM/" + 'body')
    if '.DS_Store' in f3:
        f3.remove('.DS_Store')
    f4 = os.listdir("/data/HCP_WM/" + 'object')
    if '.DS_Store' in f4:
        f4.remove('.DS_Store')
    Response = []
    for index,f in enumerate([f1,f2,f3,f4]):
        for pic in f:
            img = Image.open("/data/HCP_WM/"+class_name[index]+"/"+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
    Response = np.array(Response) 
    Response = zscore(Response, axis=0)
    Response_som = []
    for response in Response:
        Response_som.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_som = np.array(Response_som)
    return Response_som[:111,:,:],Response_som[111:172,:,:],Response_som[172:250,:,:],Response_som[250:,:,:]

def som_mask(som, Response, Contrast_respense, contrast_index, threshold_cohend):
    t_map, p_map = stats.ttest_1samp(Response, Contrast_respense[contrast_index])
    mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
    Cohend_mask = np.zeros((som._weights.shape[0],som._weights.shape[1]))
    for i in range(som._weights.shape[0]):
        for j in range(som._weights.shape[1]):
            cohend = cohen_d(Response[:,i,j], Contrast_respense[contrast_index][i,j])
            if (p_map[i,j] < 0.05/40000) and (cohend>threshold_cohend):
                Cohend_mask[i,j] = cohend
                mask[i,j] = 1
    return Cohend_mask, mask





"""Train the VTC-SOM"""
###############################################################################
###############################################################################
Data = np.load('/data/Data.npy')
Data = zscore(Data, axis=0)
pca = PCA(svd_solver='auto')
pca.fit(Data)
Data_pca = pca.transform(Data)[:,[0,1,2,3]]

som = BrainSOM.VTCSOM(200, 200, 4, sigma=6.2, learning_rate=1, neighborhood_function='gaussian')
som.Train(copy.deepcopy(Data_pca), [0,200000], step_len=1000000, verbose=False)
    
    
    
    
"""Simulated region in model------Figure 2a"""
###############################################################################
###############################################################################
Response_face,Response_place,Response_body,Response_object = Functional_map_pca(som, pca, pca_index=[0,1,2,3])
Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
thre = 0.3
face_mask = som_mask(som, Response_face, Contrast_respense, 0, thre)
place_mask = som_mask(som, Response_place, Contrast_respense, 1, thre)
limb_mask = som_mask(som, Response_body, Contrast_respense, 2, thre)
object_mask = som_mask(som, Response_object, Contrast_respense, 3, thre)

plt.figure(figsize=(8,8))
plt.imshow(face_mask[1], cmap='Reds', alpha=1, label='face')
plt.imshow(place_mask[1], cmap='Greens',  alpha=0.4, label='place')
plt.imshow(limb_mask[1], cmap='Oranges',  alpha=0.3, label='limb')
plt.imshow(object_mask[1], cmap='Blues',  alpha=0.3, label='object')
plt.axis('off')
plt.savefig('/results/artificial_regions.png')


