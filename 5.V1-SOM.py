#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from scipy.stats import zscore
import BrainSOM


"""Figure 9"""

### Data
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
                             std = [0.229, 0.224, 0.225])])
    }
        
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()

class NET(torch.nn.Module):
    def __init__(self, model, selected_layer):
        super(NET, self).__init__()
        self.model = model
        self.selected_layer = selected_layer
        self.conv_output = 0
    def hook_layer(self):
        def hook_function(module, layer_in, layer_out):
            self.conv_output = layer_out
        self.model.features[self.selected_layer].register_forward_hook(hook_function)
    def layeract(self, x):
        self.hook_layer()
        self.model(x)

def generate_letter(letter, theta, size):
    def draw_rotated_text(image, angle, xy, text, fill, *args, **kwargs):
        # get the size of our image
        width, height = image.size
        max_dim = max(width, height)   
        # build a transparency mask large enough to hold the text
        mask_size = (max_dim * 2, max_dim * 2)
        mask = Image.new('L', mask_size, 0)   
        # add text to mask
        draw = ImageDraw.Draw(mask)
        draw.text((max_dim, max_dim), text, 255, *args, **kwargs)  
        if angle % 90 == 0:
            # rotate by multiple of 90 deg is easier
            rotated_mask = mask.rotate(angle)
        else:
            # rotate an an enlarged mask to minimize jaggies
            bigger_mask = mask.resize((max_dim*8, max_dim*8),
                                      resample=Image.BICUBIC)
            rotated_mask = bigger_mask.rotate(angle).resize(
                mask_size, resample=Image.LANCZOS) 
        # crop the mask to match image
        mask_xy = (max_dim - xy[0], max_dim - xy[1])
        b_box = mask_xy + (mask_xy[0] + width, mask_xy[1] + height)
        mask = rotated_mask.crop(b_box)  
        # paste the appropriate color, with the text transparency mask
        color_image = Image.new('RGBA', image.size, fill)
        image.paste(color_image, mask)
    letter_pic = np.random.normal(0,2,(224,224,3))+200
    letter_pic = np.uint(letter_pic)
    letter_img = Image.fromarray(np.uint8(letter_pic))
    #font = ImageFont.truetype('arial.ttf', size=size)
    draw_rotated_text(letter_img, theta, (112, 112), 
                      letter, (0,0,0), anchor='mm')#, font=font)
    return letter_img
    
theta_range = np.linspace(0,2*np.pi,256)
colorbar = mpl.cm.gist_rainbow(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])
plt.figure(figsize=(8,8))
for x in np.arange(-1,1.01,0.05):
    for y in np.arange(-1,1.01,0.05):
        if y>0 and x>0:
            theta = np.arctan(y/x)
        if y>0 and x<0:
            theta = np.pi + np.arctan(y/x)
        if y<0 and x<0:
            theta = np.pi + np.arctan(y/x)
        if y<0 and x>0:
            theta = 2*np.pi + np.arctan(y/x)     
        d = np.abs(theta_range-theta)
        r = np.sqrt(x**2+y**2)
        if r < 1:
            plt.scatter(x, y, color=color_map(np.where(d==d.min())[0]), alpha=r)
plt.axis('off')
    
    


### Train SOM by bar
###############################################################################
# Based on last layer
def layer_activation_for_bar(model_truncate, rotation_theta):
    img = generate_letter('I', theta, 200).convert('RGB')
    picimg = data_transforms['val'](img).unsqueeze(0)
    model_truncate.layeract(picimg)
    return model_truncate.conv_output.data.numpy()[0]

model_truncate = NET(alexnet, selected_layer=1)
Layer_act = []
for theta in tqdm(np.arange(90,270,0.1)):
    act = layer_activation_for_bar(model_truncate,theta).reshape(-1)
    Layer_act.append(act)
Layer_act = np.array(Layer_act)

deleted_col_1 = np.where(np.std(Layer_act, axis=0)==0)[0]
Layer_act = np.delete(Layer_act, deleted_col_1, axis=1)
deleted_col_2 = np.where(np.std(Layer_act, axis=0)==0)[0]
Layer_act = np.delete(Layer_act, deleted_col_2, axis=1)
mean_data = np.mean(Layer_act, axis=0)
std_data = np.std(Layer_act, axis=0)
Layer_act = zscore(Layer_act, axis=0)

pca = PCA(svd_solver='full')
pca.fit(Layer_act)
Data_pca = pca.transform(Layer_act)[:,[0,1]]
   
sig = 2.4
som = BrainSOM.VTCSOM(200, 200, 2, sigma=sig, learning_rate=1, 
                      neighborhood_function='gaussian')
q_error = som.Train(Data_pca, num_iteration=[0,1800*10], step_len=1000000, verbose=False)



#### Test Pinwheel 
###############################################################################
import matplotlib as mpl

theta_range = np.linspace(0,np.pi,256)
colorbar = mpl.cm.gist_rainbow(np.arange(256))
color_map = mpl.colors.ListedColormap(colorbar, name='myColorMap', N=colorbar.shape[0])

model_truncate = NET(alexnet, selected_layer=1)

def layer_activation_for_bar(model_truncate, rotation_theta):
    img = generate_letter('I', theta, 200).convert('RGB')
    picimg = data_transforms['val'](img).unsqueeze(0)
    model_truncate.layeract(picimg)
    return model_truncate.conv_output.data.numpy()[0]

SOM_act = []
for theta in tqdm(np.arange(90,270,1)):
    act = layer_activation_for_bar(model_truncate,theta).reshape(1,-1)
    act = np.delete(act, deleted_col_1, axis=1)
    act = np.delete(act, deleted_col_2, axis=1)
    act = (act-mean_data)/std_data
    som_act = 1/som.activate(pca.transform(act)[0,[0,1]])
    SOM_act.append(som_act)
SOM_act = np.array(SOM_act) 
    
OR_map = np.zeros(som._weights.shape[0:2])
for i in range(som._weights.shape[0]):
    for j in range(som._weights.shape[1]):
        OR_map[i,j] = np.pi * SOM_act[:,i,j].argmax()*1/180
plt.figure(dpi=300)
plt.matshow(OR_map, cmap=color_map)
plt.axis('off')
plt.savefig('/results/V1_OR.png')



### Qualitative test
# Fourier analysis
fft2 = np.fft.fft2(OR_map)
shift2center = np.fft.fftshift(fft2)

plt.figure(dpi=300)
plt.title('Energy')
temp = np.abs(shift2center)
temp[100,100] = 0
temp[100,99] = 0
temp[100,101] = 0
temp[99,100] = 0
temp[101,100] = 0
plt.imshow(temp, cmap='Greys')
plt.colorbar()
plt.axis('off')
plt.savefig('/results/V1_fft.png')


# Gradient map
def Gradient_map(OR_map):
    return np.sqrt(np.square(np.gradient(OR_map)).sum(axis=0))

GM = Gradient_map(OR_map)
plt.figure(dpi=300)
plt.imshow(np.power(GM,0.5), cmap='Greys')
plt.axis('off')
plt.savefig('/results/V1_gradient.png')


    
