# -*- coding: utf-8 -*-

"""Simulated maps------Figure 3"""
###############################################################################
###############################################################################
import torch
import os
import copy
import csv
from tqdm import tqdm
import numpy as np
from skimage import transform
from scipy.stats import zscore
import scipy.signal as signal
import scipy.stats as stats
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nibabel as nib
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric
from dipy.viz import regtools
import BrainSOM


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


threshold_cohend = 0.5
HCP_data = nib.load('/data/HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s4_MSMAll.dscalar.nii')
mask = nib.load('/data/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
vtc_mask = np.where((mask==7)|(mask==18)|(mask==22)|(mask==127)|(mask==136)|(mask==138)|(mask==154)|(mask==163)|(mask==7+180)|(mask==18+180)|(mask==22+180)|(mask==127+180)|(mask==136+180)|(mask==138+180)|(mask==154+180)|(mask==163+180))[0]
hcp_vtc = np.zeros(91282)
hcp_vtc[vtc_mask] = 1
R_vtc_mask = np.where((mask==7)|(mask==18)|(mask==22)|(mask==127)|(mask==136)|(mask==138)|(mask==154)|(mask==163))[0]
R_hcp_vtc = np.zeros(91282)
R_hcp_vtc[R_vtc_mask] = 1
L_vtc_mask = np.where((mask==7+180)|(mask==18+180)|(mask==22+180)|(mask==127+180)|(mask==136+180)|(mask==138+180)|(mask==154+180)|(mask==163+180))[0]
L_hcp_vtc = np.zeros(91282)
L_hcp_vtc[L_vtc_mask] = 1

hcp_face = HCP_data.dataobj[19,:]
hcp_face = hcp_face * hcp_vtc
hcp_face = np.where(hcp_face>=threshold_cohend, 1, 0)

hcp_place = HCP_data.dataobj[20,:]
hcp_place = hcp_place * hcp_vtc
hcp_place = np.where(hcp_place>=threshold_cohend, 1, 0)

hcp_limb = HCP_data.dataobj[18,:]
hcp_limb = hcp_limb * hcp_vtc
hcp_limb = np.where(hcp_limb>=threshold_cohend, 1, 0)

hcp_object = HCP_data.dataobj[21,:]
hcp_object = hcp_object * hcp_vtc
hcp_object = np.where(hcp_object>=threshold_cohend, 1, 0)



# make dict
list_of_block = list(HCP_data.header.get_index_map(1).brain_models)
CORTEX_Left = list_of_block[0]
Dict_hcp_to_32kL = dict()
for vertex in range(CORTEX_Left.index_count):
    Dict_hcp_to_32kL[vertex] = CORTEX_Left.vertex_indices[vertex]
Dict_32kL_to_hcp = {v:k for k,v in Dict_hcp_to_32kL.items()}

list_of_block = list(HCP_data.header.get_index_map(1).brain_models)
CORTEX_Right = list_of_block[1]
Dict_hcp_to_32kR = dict()
for vertex in range(CORTEX_Right.index_count):
    Dict_hcp_to_32kR[vertex] = CORTEX_Right.vertex_indices[vertex]
Dict_32kR_to_hcp = {v:k for k,v in Dict_hcp_to_32kR.items()}
        
def get_Lvtc_position(plot=False):
    # geometry information
    geometry = nib.load('/data/S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    mask = nib.load('/data/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
    L_vtc_mask = np.where((mask==7+180)|(mask==18+180)|(mask==22+180)|(mask==127+180)|(mask==136+180)|(mask==138+180)|(mask==154+180)|(mask==163+180))[0]
    L_hcp_vtc = np.zeros(91282)
    L_hcp_vtc[L_vtc_mask] = 1
    # hcp mapping to 32K
    vtc_32K = []
    for i in np.where(L_hcp_vtc[:CORTEX_Left.index_count]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kL[i])
    position = geometry[vtc_32K][:,[0,1]]
    if plot==False:
        pass
    if plot==True:
        plt.figure()
        plt.scatter(geometry[:,0], geometry[:,1], marker='.')
        plt.scatter(position[:,0], position[:,1], marker='s')
    return position

def get_Rvtc_position(plot=False):
    # geometry information
    geometry = nib.load('/data/S1200.R.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    mask = nib.load('/data/MMP_mpmLR32k.dlabel.nii').dataobj[0][:]
    R_vtc_mask = np.where((mask==7)|(mask==18)|(mask==22)|(mask==127)|(mask==136)|(mask==138)|(mask==154)|(mask==163))[0]
    R_hcp_vtc = np.zeros(91282)
    R_hcp_vtc[R_vtc_mask] = 1
    # hcp mapping to 32K
    vtc_32K = []
    for i in np.where(R_hcp_vtc[29696:CORTEX_Right.index_count+29696]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kR[i])
    position = geometry[vtc_32K][:,[0,1]]
    if plot==False:
        pass
    if plot==True:
        plt.figure()
        plt.scatter(geometry[:,0], geometry[:,1], marker='.')
        plt.scatter(position[:,0], position[:,1], marker='s')
    return position

def get_L_hcp_space_mask(hcp_index, threshold, plot=False):
    # geometry information
    geometry = nib.load('/data/S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    hcp_data = HCP_data.dataobj[hcp_index,:]
    hcp_data = hcp_data * L_hcp_vtc
    hcp_data = np.where(hcp_data>=threshold, 1, 0)
    # hcp mapping to 32K
    vtc_32K = []
    hcp_32K = []
    for i in np.where(L_hcp_vtc[:CORTEX_Left.index_count]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kL[i])
        hcp_32K.append(hcp_data[i])
    if plot==False:
        pass
    if plot==True:
        # plot
        plt.figure()
        plt.scatter(geometry[:,0], geometry[:,1], marker='.')
        plt.scatter(geometry[vtc_32K][:,0], geometry[vtc_32K][:,1], marker='.',
                    c=hcp_32K, cmap=plt.cm.jet)
    class_index = np.where(np.array(hcp_32K)==1)[0].tolist()
    position = []
    for i in class_index:
        position.append(geometry[vtc_32K[i]][[0,1]].tolist())
    position = np.array(position)
    value = hcp_32K
    return position, value

def get_R_hcp_space_mask(hcp_index, threshold, plot=False):
    # geometry information
    geometry = nib.load('/data/S1200.R.flat.32k_fs_LR.surf.gii').darrays[0].data
    # data
    hcp_data = HCP_data.dataobj[hcp_index,:]
    hcp_data = hcp_data * R_hcp_vtc
    hcp_data = np.where(hcp_data>=threshold, 1, 0)
    # hcp mapping to 32K
    vtc_32K = []
    hcp_32K = []
    for i in np.where(R_hcp_vtc[29696:CORTEX_Right.index_count+29696]==1)[0]:
        vtc_32K.append(Dict_hcp_to_32kR[i])
        hcp_32K.append(hcp_data[i+29696])
    if plot==False:
        pass
    if plot==True:
        # plot
        plt.figure()
        plt.scatter(geometry[:,0], geometry[:,1], marker='.')
        plt.scatter(geometry[vtc_32K][:,0], geometry[vtc_32K][:,1], marker='.',
                    c=hcp_32K, cmap=plt.cm.jet)
    class_index = np.where(np.array(hcp_32K)==1)[0].tolist()
    position = []
    for i in class_index:
        position.append(geometry[vtc_32K[i]][[0,1]].tolist())
    position = np.array(position)
    value = hcp_32K
    return position, value

def fill_hcp_map(position):
    round_position = []
    for v in range(position.shape[0]):
        round_position.append([np.floor(position[v,:][0]), np.floor(position[v,:][1])])
        round_position.append([np.floor(position[v,:][0]), np.ceil(position[v,:][1])])
        round_position.append([np.ceil(position[v,:][0]), np.floor(position[v,:][1])])
        round_position.append([np.ceil(position[v,:][0]), np.ceil(position[v,:][1])])
    round_position = np.array(round_position)
    return round_position
 
def make_moving_map(position, shift, rotation_theta, hemisphere):
    '''
    shift is a list, like [70,90]
    hemisphere is 'left' or 'right'
    '''
    round_position = fill_hcp_map(position)
    if hemisphere=='left':
        vtc_round_position = fill_hcp_map(get_Lvtc_position(plot=False))
    if hemisphere=='right':
        vtc_round_position = fill_hcp_map(get_Rvtc_position(plot=False))    
    round_position[:,0] -= vtc_round_position[:,0].min()-170
    round_position[:,1] -= vtc_round_position[:,1].min()-160
    round_position = np.int0(round_position)
    moving = np.zeros((340,340))
    for pos in round_position:
        moving[pos[0],pos[1]] = 1
    moving_copy = copy.deepcopy(moving)
    it = np.nditer(moving, flags=['multi_index'])
    while not it.finished:
        for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
            for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                if moving_copy[it.multi_index]==1:
                    moving[ii,jj] = 1
        it.iternext()
    moving = transform.rotate(moving, rotation_theta)
    pos = np.where(moving==1)
    move = np.zeros((340,340))
    move[pos[0]+shift[0], pos[1]+shift[1]] = 1
    return move
    
def Make_mapping_vtc2sheet(position, shift, rotation_theta, hemisphere):
    moving = make_moving_map(position, shift, rotation_theta, hemisphere)
    static = np.zeros((340,340))  
    static[70:270,70:270] = 1
    regtools.overlay_images(static, moving, 'Static', 'Overlay', 'Moving')    
    dim = static.ndim
    metric = SSDMetric(dim)    
    level_iters = [500, 200, 100, 50, 10]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(static, moving)
    regtools.plot_2d_diffeomorphic_map(mapping)   
    warped_moving = mapping.transform(moving, 'linear')
    regtools.overlay_images(static, warped_moving, 'Static', 'Overlay', 'Warped moving')
    return mapping

def Mapping_area2sheet(mapping, area_position, shift, rotation_theta, hemisphere):
    moving = make_moving_map(area_position, shift, rotation_theta, hemisphere)
    static = np.zeros((340,340))  
    static[70:270,70:270] = 1    
    warped_moving = mapping.transform(moving, 'linear')
    warped_moving = warped_moving[70:270,70:270]
    return warped_moving

def Mapping_som_area_2_vtc(mapping, som_area):
    '''som_area: array (200x200)'''      
    moving = np.zeros((340,340))  
    moving[70:270,70:270] = som_area
    vtc_area = mapping.transform_inverse(moving)    
    return vtc_area

def drive_area_2_vtc_axis(warped_area, shift, inverse_rotation_theta, hemisphere):
    if hemisphere=='left':
        position = get_Lvtc_position(plot=False)
        geometry = nib.load('/data/S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
        Dict_32k_to_hcp = Dict_32kL_to_hcp
    if hemisphere=='right':
        position = get_Rvtc_position(plot=False)
        geometry = nib.load('/data/S1200.R.flat.32k_fs_LR.surf.gii').darrays[0].data
        Dict_32k_to_hcp = Dict_32kR_to_hcp
    # shift
    warped_area_temp = np.zeros((340,340))
    warped_area_temp[np.where(warped_area!=0)[0]-shift[0], np.where(warped_area!=0)[1]-shift[1]] = warped_area[np.where(warped_area!=0)]
    warped_area = warped_area_temp
    # rotation
    warped_area = transform.rotate(warped_area, inverse_rotation_theta)
    vtc_round_position = fill_hcp_map(position)
    Xs = np.float32(np.where(warped_area!=0)[0])
    Ys = np.float32(np.where(warped_area!=0)[1])
    Values = warped_area[np.where(warped_area!=0)]
    # shift to fmri axis
    Xs += vtc_round_position[:,0].min()-170
    Ys += vtc_round_position[:,1].min()-160
    # units position and index in 32K space
    units_position = []
    units_index_in_32K = []
    for unit in zip(Xs,Ys):
        temp = np.abs(position-unit)
        t = temp[:,0] + temp[:,1]
        units_position.append(position[t.argmin(),:])
        units_index_in_32K.append(np.where(geometry[:,[0,1]]==position[t.argmin(),:])[0][0])
    units_position = np.array(units_position)  
    units_index_in_32K = np.array(units_index_in_32K)
    units_index_in_hcp = []
    for i in units_index_in_32K:
        units_index_in_hcp.append(Dict_32k_to_hcp[i])
    units_in_hcp = np.zeros(91282)
    if hemisphere=='left':
        units_in_hcp[units_index_in_hcp] = Values
    if hemisphere=='right':
        units_in_hcp[[x+29696 for x in units_index_in_hcp]] = Values
    return units_in_hcp

def save_warped_area_as_gii(units_in_hcp, out_dir):
    """"out_dir: .dtseries.nii"""
    data_nii = nib.load('/data/seg1_1_Atlas.dtseries.nii')
    img = np.tile(units_in_hcp, (245,1))
    IMG = nib.cifti2.cifti2.Cifti2Image(img, header=data_nii.header)
    nib.save(IMG, out_dir)

def Mapping_som_units_2_vtc_units(mapping, som, sigma, shift, rotation_theta, hemisphere, units):
    ''''units: [100:102,100:102]'''
    if hemisphere=='left':
        vtc = make_moving_map(get_Lvtc_position(plot=False), shift, rotation_theta, hemisphere)
    if hemisphere=='right':
        vtc = make_moving_map(get_Rvtc_position(plot=False), shift, rotation_theta, hemisphere)        
    som_units = np.zeros((200,200))
    som_units[units] = 1
    moving = np.zeros((250,250))  
    moving[20:220,20:220] = som_units
    vtc_units = mapping.transform_inverse(moving)    
    plt.figure(figsize=(12,12))
    plt.imshow(som_units);plt.axis('off')
    plt.figure(figsize=(12,12))
    plt.imshow(vtc, alpha=1);plt.axis('off')
    plt.imshow(vtc_units, alpha=0.5, cmap='jet');plt.axis('off')
    return som_units, vtc_units

def Mapping_som_structure_constrain_2_vtc_map(mapping, som, sigma, shift, rotation_theta, hemisphere, plot_number):
    if hemisphere=='left':
        vtc = make_moving_map(get_Lvtc_position(plot=False), shift, rotation_theta, hemisphere)
    if hemisphere=='right':
        vtc = make_moving_map(get_Rvtc_position(plot=False), shift, rotation_theta, hemisphere)        
    x = np.random.choice(np.arange(0,200,1), plot_number)
    y = np.random.choice(np.arange(0,200,1), plot_number)
    point_pair = zip(x,y)
    som_structure = np.zeros((200,200))
    vtc_structure_constrain = np.zeros((340,340))
    for point in point_pair:
        som_structure_constrain = som._gaussian(point,sigma)
        som_structure += som_structure_constrain
        moving = np.zeros((340,340))  
        moving[70:270,70:270] = som_structure_constrain
        vtc_structure_constrain += mapping.transform_inverse(moving)    
    plt.figure(figsize=(12,12))
    plt.imshow(som_structure);plt.axis('off')
    plt.figure(figsize=(12,12))
    plt.imshow(vtc, alpha=1);plt.axis('off')
    plt.imshow(vtc_structure_constrain, alpha=0.5, cmap='jet');plt.axis('off')
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    xx = np.arange(0,340,1)
    yy = -np.arange(-340,0,1)
    X, Y = np.meshgrid(xx, yy)
    surf = ax.plot_surface(X,Y,vtc_structure_constrain, cmap='jet')
    ax.set_zlim3d(0)
    fig.colorbar(surf)
    return som_structure, vtc_structure_constrain
   
def plot_vertex_distance(vertex_dir, hemisphere):
    if hemisphere=='left':
        geometry = nib.load('/data/S1200.L.flat.32k_fs_LR.surf.gii').darrays[0].data
    if hemisphere=='right':
        geometry = nib.load('/data/S1200.R.flat.32k_fs_LR.surf.gii').darrays[0].data
    hcp_data = nib.load(vertex_dir).darrays[0].data
    hcp_data = np.where(hcp_data>0)[0]
    plt.figure()
    plt.scatter(geometry[:,0], geometry[:,1], marker='.')
    plt.scatter(geometry[hcp_data][:,0], geometry[hcp_data][:,1], marker='.', cmap=plt.cm.jet)




som = BrainSOM.VTCSOM(200, 200, 4, sigma=6.2, learning_rate=1, neighborhood_function='gaussian')
som._weights = np.load('/data/som_sigma_6.2.npy')



### Eccentricity
def cohen_d(x1, x2):
    s1 = x1.std()
    s1_ = (x1.shape[0]-1)*(s1**2)
    s2 = x2.std()
    s2_ = (x2.shape[0]-1)*(s2**2)
    s_within = np.sqrt((s1_+s2_)/(x1.shape[0]+x2.shape[0]-2))
    return (x1.mean()-x2.mean())/s_within

def generate_eccentricity_stimuli(pic_size, inner_width, exter_width):
    center_position = (pic_size/2, pic_size/2)
    stim = np.zeros((pic_size, pic_size))
    for i in range(pic_size):
        for j in range(pic_size):
            d = np.sqrt((i-center_position[0])**2+(j-center_position[1])**2)
            if inner_width < d < exter_width:
                stim[i,j] = 1
            else:
                pass
    stimuli = np.zeros((pic_size,pic_size,3))
    stimuli[:,:,0] = stim
    stimuli[:,:,1] = stim
    stimuli[:,:,2] = stim
    return stimuli

def Response_eccentricity(som, pca_index, fovea_inner=0, fovea_exter=range(10,30), preph_inner=50, preph_exter=range(60,80)): 
    Data = np.load('/data/Data.npy')
    Data = zscore(Data, axis=0)
    pca = PCA()
    pca.fit(Data)
    Response_ecc = []
    for i in fovea_exter:
        stimuli = generate_eccentricity_stimuli(224,fovea_inner,i)
        picimg = torch.Tensor(stimuli).permute(2,0,1).unsqueeze(0)
        output = alexnet(picimg).data.numpy()
        Response_ecc.append(output[0])
    for i in preph_exter:
        stimuli = generate_eccentricity_stimuli(224,preph_inner,i)
        picimg = torch.Tensor(stimuli).permute(2,0,1).unsqueeze(0)
        output = alexnet(picimg).data.numpy()
        Response_ecc.append(output[0])    
    Response_ecc = np.array(Response_ecc)
    Response_ecc = zscore(Response_ecc, axis=0)
    Response_ec = []
    Response_ec_map = []
    for response in Response_ecc:
        Response_ec.append(pca.transform(response.reshape(1,-1))[0,pca_index])
        Response_ec_map.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_ec = np.array(Response_ec)
    Response_ec_map = np.array(Response_ec_map)
    return Response_ec_map[:len(fovea_exter)], Response_ec_map[len(fovea_exter):]

ECC_response_1,ECC_response_2 = Response_eccentricity(som, [0,1,2,3])
ECC_response_1_avg = ECC_response_1.mean(axis=0)
ECC_response_2_avg = ECC_response_2.mean(axis=0)

threshold_cohend = 0.5
Eccentricity_map = np.zeros((200,200))
for i in range(200):
    for j in range(200):
        max_value = np.max([ECC_response_1_avg[i,j], ECC_response_2_avg[i,j]])
        index = np.argmax([ECC_response_1_avg[i,j], ECC_response_2_avg[i,j]])
        t, p = stats.ttest_ind(ECC_response_1[:,i,j], ECC_response_2[:,i,j])
        cohend = cohen_d(ECC_response_1[:,i,j], ECC_response_2[:,i,j])
        if (p < 0.05/40000) and (np.abs(cohend)>threshold_cohend):
            Eccentricity_map[i,j] = index
        else:
            Eccentricity_map[i,j] = -1
plt.figure(figsize=(8,8))      
plt.scatter(np.where(Eccentricity_map==0)[0], np.where(Eccentricity_map==0)[1], color='red')
plt.scatter(np.where(Eccentricity_map==1)[0], np.where(Eccentricity_map==1)[1], color='green') 
plt.xlim([0,200])  
plt.ylim([0,200]) 
Fovea_map = np.where(Eccentricity_map+1!=1, 0, 1)
Preph_map = np.where(Eccentricity_map!=1, 0, 1)


# left
shift = [70,90]
rotation_theta = 215
mapping_Left = Make_mapping_vtc2sheet(get_Lvtc_position(plot=False), shift, rotation_theta, hemisphere='left')    
warped_Fovea_map = Mapping_som_area_2_vtc(mapping_Left, Fovea_map)
save_warped_area_as_gii(drive_area_2_vtc_axis(warped_Fovea_map, shift, -rotation_theta, 'left'), 
                        '/results/warped_Fovea_map_left_sigma6.2.dtseries.nii')
warped_Preph_map = Mapping_som_area_2_vtc(mapping_Left, Preph_map)
save_warped_area_as_gii(drive_area_2_vtc_axis(warped_Preph_map, shift, -rotation_theta, 'left'), 
                        '/results/warped_Preph_map_left_sigma6.2.dtseries.nii')

plt.figure()
plt.imshow(warped_Fovea_map);plt.axis('off')
plt.savefig('/results/warped_Fovea_map.png')
plt.figure()
plt.imshow(warped_Preph_map);plt.axis('off')
plt.savefig('/results/warped_Preph_map.png')




### Animate vs Inanimate
def Response_animate_inanimate(som, pca_index): 
    Data = np.load('/data/Data.npy')
    Data = zscore(Data, axis=0)
    pca = PCA()
    pca.fit(Data)
    def Response_som(som, picdir):
        f = os.listdir(picdir)
        for i in f:
            if i[-3:] != 'jpg':
                f.remove(i)
        Response = []
        for pic in f:
            img = Image.open(picdir+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
        Response = np.array(Response) 
        return Response
    Response_big_animate = Response_som(som, '/data/Animate_Inanimate/Big-Animate/')
    Response_big_inanimate = Response_som(som, '/data/Animate_Inanimate/Big-Inanimate/')
    Response_small_animate = Response_som(som, '/data/Animate_Inanimate/Small-Animate/')
    Response_small_inanimate = Response_som(som, '/data/Animate_Inanimate/Small-Inanimate/')
    Response = np.vstack((Response_big_animate,Response_small_animate,Response_big_inanimate,Response_small_inanimate))
    Response = zscore(Response, axis=0)
    Response_ai = []
    for response in Response:
        Response_ai.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_ai = np.array(Response_ai)
    return Response_ai[:120], Response_ai[120:]

### Animate Inanimate (Konkle + MIT)
Animate_response,Inanimate_response = Response_animate_inanimate(som, [0,1,2,3])
Animate_response_avg = np.array(Animate_response).mean(axis=0)
Inanimate_response_avg = np.array(Inanimate_response).mean(axis=0)

threshold_cohend = 0.5
Animate_Inanimate_map = np.zeros((200,200))
for i in range(200):
    for j in range(200):
        max_value = np.max([Animate_response_avg[i,j], Inanimate_response_avg[i,j]])
        index = np.argmax([Animate_response_avg[i,j], Inanimate_response_avg[i,j]])
        t, p = stats.ttest_ind(Animate_response[:,i,j], Inanimate_response[:,i,j])
        cohend = cohen_d(Animate_response[:,i,j], Inanimate_response[:,i,j])
        if (p < 0.05/40000) and (np.abs(cohend)>threshold_cohend):
            Animate_Inanimate_map[i,j] = index
        else:
            Animate_Inanimate_map[i,j] = -1

plt.figure(figsize=(8,8))      
plt.scatter(np.where(Animate_Inanimate_map==0)[0], np.where(Animate_Inanimate_map==0)[1], color='red')
plt.scatter(np.where(Animate_Inanimate_map==1)[0], np.where(Animate_Inanimate_map==1)[1], color='green') 
plt.xlim([0,200])  
plt.ylim([0,200]) 


# left
shift = [70,90]
rotation_theta = 215
Animate_map = np.where(Animate_Inanimate_map+1!=1, 0, 1)
Inanimate_map = np.where(Animate_Inanimate_map!=1, 0, 1)
mapping_Left = Make_mapping_vtc2sheet(get_Lvtc_position(plot=False), shift, rotation_theta, hemisphere='left')    
warped_Animate_map = Mapping_som_area_2_vtc(mapping_Left, Animate_map)
save_warped_area_as_gii(drive_area_2_vtc_axis(warped_Animate_map, shift, -rotation_theta, 'left'), 
                        '/results/warped_Animate_map_left_sigma6.2.dtseries.nii')
warped_Inanimate_map = Mapping_som_area_2_vtc(mapping_Left, Inanimate_map)
save_warped_area_as_gii(drive_area_2_vtc_axis(warped_Inanimate_map, shift, -rotation_theta, 'left'), 
                        '/results/warped_Inanimate_map_left_sigma6.2.dtseries.nii')

plt.figure()
plt.imshow(warped_Animate_map);plt.axis('off')
plt.savefig('/results/warped_Animate_map.png')
plt.figure()
plt.imshow(warped_Inanimate_map);plt.axis('off')
plt.savefig('/results/warped_Inanimate_map.png')



### Big vs Small
def Response_big_small(som, pca_index): 
    Data = np.load('/data/Data.npy')
    Data = zscore(Data, axis=0)
    pca = PCA()
    pca.fit(Data)
    def Response_som(som, picdir):
        f = os.listdir(picdir)
        for i in f:
            if i[-3:] != 'jpg':
                f.remove(i)
        Response = []
        for pic in f:
            img = Image.open(picdir+pic).convert('RGB')
            picimg = data_transforms['val'](img).unsqueeze(0) 
            output = alexnet(picimg).data.numpy()
            Response.append(output[0])
        Response = np.array(Response) 
        return Response
    Response_big = Response_som(som, '/data/BigSmallObjects/Big/')
    Response_small = Response_som(som, '/data/BigSmallObjects/Small/')
    Response = np.vstack((Response_big,Response_small))
    Response = zscore(Response, axis=0)
    Response_bs = []
    for response in Response:
        Response_bs.append(1/som.activate(pca.transform(response.reshape(1,-1))[0,pca_index]))
    Response_bs = np.array(Response_bs)
    return Response_bs[:200], Response_bs[200:]

Big_response,Small_response = Response_big_small(som, [0,1,2,3])
Big_response_avg = np.array(Big_response).mean(axis=0)
Small_response_avg = np.array(Small_response).mean(axis=0)

threshold_cohend = 0.5
Big_Small_map = np.zeros((200,200))
for i in range(200):
    for j in range(200):
        max_value = np.max([Big_response_avg[i,j], Small_response_avg[i,j]])
        index = np.argmax([Big_response_avg[i,j], Small_response_avg[i,j]])
        t, p = stats.ttest_ind(Big_response[:,i,j], Small_response[:,i,j])
        cohend = cohen_d(Big_response[:,i,j], Small_response[:,i,j])
        if (p < 0.05/40000) and (np.abs(cohend)>threshold_cohend):
            Big_Small_map[i,j] = index
        else:
            Big_Small_map[i,j] = -1

plt.figure(figsize=(8,8))      
plt.scatter(np.where(Big_Small_map==0)[0], np.where(Big_Small_map==0)[1], color='green')
plt.scatter(np.where(Big_Small_map==1)[0], np.where(Big_Small_map==1)[1], color='red') 
plt.xlim([0,200])  
plt.ylim([0,200])

# left
shift = [70,90]
rotation_theta = 215
Big_map = np.where(Big_Small_map+1!=1, 0, 1)
Small_map = np.where(Big_Small_map!=1, 0, 1)
mapping_Left = Make_mapping_vtc2sheet(get_Lvtc_position(plot=False), shift, rotation_theta, hemisphere='left')    
warped_Big_map = Mapping_som_area_2_vtc(mapping_Left, Big_map)
save_warped_area_as_gii(drive_area_2_vtc_axis(warped_Big_map, shift, -rotation_theta, 'left'), 
                        '/results/warped_Big_map_left_sigma6.2.dtseries.nii')
warped_Small_map = Mapping_som_area_2_vtc(mapping_Left, Small_map)
save_warped_area_as_gii(drive_area_2_vtc_axis(warped_Small_map, shift, -rotation_theta, 'left'), 
                        '/results/warped_Small_map_left_sigma6.2.dtseries.nii')


plt.figure()
plt.imshow(warped_Big_map);plt.axis('off')
plt.savefig('/results/warped_Big_map.png')
plt.figure()
plt.imshow(warped_Small_map);plt.axis('off')
plt.savefig('/results/warped_Small_map.png')



