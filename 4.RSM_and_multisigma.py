#!\usr\bin\env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.stats import zscore
import scipy.stats as stats
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy as sp
import sys
import BrainSOM


"""Figure 5 6 7 8"""

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



### RSM between VTC and model
def normolize(X):
    return 2*((X-X.min())/(X.max()-X.min()))-1

RSM_macaque = np.load('/data/RSM_macaque.npy')
RSM_som = np.load('/data/RSM_som.npy')
RSM_macaque_triu = np.triu(RSM_macaque, 1)
RSM_som_triu = np.triu(RSM_som, 1)

plt.figure(dpi=500)
plt.imshow(normolize(RSM_som), cmap='rainbow')
plt.axis('off')
plt.colorbar()
plt.savefig('/results/RSM_som.png')
plt.figure(dpi=500)
plt.imshow(normolize(RSM_macaque), cmap='rainbow')
plt.axis('off')
plt.colorbar()
plt.savefig('/results/RSM_macaque.png')
print(sp.stats.pearsonr(RSM_som_triu[np.where(RSM_som_triu!=0)],
                        RSM_macaque_triu[np.where(RSM_macaque_triu!=0)]))



### RSM in regions between VTC and model
def r_RSM(A,B):
    temp = np.ones((51,51))
    temp_triu = np.triu(temp, 1)
    r,p = sp.stats.pearsonr(A[np.where(temp_triu==1)], B[np.where(temp_triu==1)])
    return r,p

def Region_RSA(RSM_macaque_region, RSM_som_region, region_name):
    plt.figure(dpi=500)
    plt.subplot(121)
    plt.imshow(RSM_som_region, cmap='rainbow');plt.axis('off')
    plt.subplot(122)
    plt.imshow(RSM_macaque_region, cmap='rainbow');plt.axis('off')
    plt.savefig('/results/RSM_'+region_name+'.png')
    print(r_RSM(RSM_macaque_region, RSM_som_region))
    
RSM_macaque_face = np.load('/data/RSM_macaque_face.npy')
RSM_macaque_body = np.load('/data/RSM_macaque_body.npy')
RSM_macaque_spiky = np.load('/data/RSM_macaque_spiky.npy')
RSM_macaque_stubby = np.load('/data/RSM_macaque_stubby.npy')
RSM_som_face = np.load('/data/RSM_som_face.npy')
RSM_som_body = np.load('/data/RSM_som_body.npy')
RSM_som_spiky = np.load('/data/RSM_som_spiky.npy')
RSM_som_stubby = np.load('/data/RSM_som_stubby.npy')

Region_RSA(RSM_som_face, RSM_macaque_face, 'face')
Region_RSA(RSM_som_body, RSM_macaque_body, 'body')
Region_RSA(RSM_som_spiky, RSM_macaque_spiky, 'spiky')
Region_RSA(RSM_som_stubby, RSM_macaque_stubby, 'stubby')



### View-invariant
from matplotlib import colors

Datasets = np.load('/data/Datasets.npy')
for i,index in enumerate([[0,1,2],[3,4,5],[6,7,8],[9,10,11]]):
    dataset1 = Datasets[index[0]]
    dataset2 = Datasets[index[1]]
    dataset3 = Datasets[index[2]]
    vmin = min(np.min(dataset1), np.min(dataset2), np.min(dataset3))
    vmax = max(np.max(dataset1), np.max(dataset2), np.max(dataset3))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # RSM
    fig = plt.figure(figsize=(7,4), dpi=300)
    bx1 = fig.add_subplot(234)
    x = bx1.imshow(dataset1, norm=norm, cmap='rainbow');bx1.axis('off')
    bx2 = fig.add_subplot(235)
    y = bx2.imshow(dataset2, norm=norm, cmap='rainbow');bx2.axis('off')
    bx3 = fig.add_subplot(236)
    z = bx3.imshow(dataset3, norm=norm, cmap='rainbow');bx3.axis('off')
    fig.subplots_adjust(hspace=0, wspace=0.3)
    plt.savefig('/results/'+str(i+1)+'.png')



### Transitions in the functional organization
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


Data = np.load('/data/Data.npy')
Data = zscore(Data, axis=0)
pca = PCA(svd_solver='auto')
pca.fit(Data)

for sig in [0.1,0.7,1.7,3.1,6.2,9.3,10.0]:
    som = BrainSOM.VTCSOM(200, 200, 4, sigma=sig, learning_rate=1, neighborhood_function='gaussian')
    som._weights = np.load('/data/som_sigma_'+str(sig)+'.npy')
    Response_face,Response_place,Response_body,Response_object = Functional_map_pca(som, pca, pca_index=[0,1,2,3])
    Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                         np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
    face_mask = som_mask(som, Response_face, Contrast_respense, 0, 0.5)
    place_mask = som_mask(som, Response_place, Contrast_respense, 1, 0.5)
    limb_mask = som_mask(som, Response_body, Contrast_respense, 2, 0.5)
    object_mask = som_mask(som, Response_object, Contrast_respense, 3, 0.5)
    
    plt.figure(figsize=(8,8))
    plt.imshow(face_mask[1], cmap='Reds', alpha=1, label='face')
    plt.imshow(place_mask[1], cmap='Greens',  alpha=0.4, label='place')
    plt.imshow(limb_mask[1], cmap='Oranges',  alpha=0.3, label='limb')
    plt.imshow(object_mask[1], cmap='Blues',  alpha=0.3, label='object')
    plt.axis('off')
    plt.savefig('/results/artificial_regions_'+str(sig)+'.png')



