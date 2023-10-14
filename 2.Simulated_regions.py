# -*- coding: utf-8 -*-

"""Simulated region to VTC------Figure 2b"""
###############################################################################
###############################################################################
import os
import copy
from tqdm import tqdm
import numpy as np
from skimage import transform
from scipy.stats import zscore
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




"""Symmetric Diffeomorphic Registration"""
###############################################################################
###############################################################################
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
    dim = static.ndim
    metric = SSDMetric(dim)    
    level_iters = [500, 200, 100, 50, 10]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    mapping = sdr.optimize(static, moving)
    warped_moving = mapping.transform(moving, 'linear')
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
pca = PCA()
pca.fit(Data)

som = BrainSOM.VTCSOM(200, 200, 4, sigma=6.2, learning_rate=1, neighborhood_function='gaussian')
som._weights = np.load('/data/som_sigma_6.2.npy')
Response_face,Response_place,Response_body,Response_object = Functional_map_pca(som, pca, pca_index=[0,1,2,3])
Contrast_respense = [np.vstack((Response_place,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_body,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_object)).mean(axis=0),
                     np.vstack((Response_face,Response_place,Response_body)).mean(axis=0)]
face_mask = som_mask(som, Response_face, Contrast_respense, 0, 0.5)
place_mask = som_mask(som, Response_place, Contrast_respense, 1, 0.5)
limb_mask = som_mask(som, Response_body, Contrast_respense, 2, 0.5)
object_mask = som_mask(som, Response_object, Contrast_respense, 3, 0.5)



shift = [70,90]
rotation_theta = 215
mapping_Left = Make_mapping_vtc2sheet(get_Lvtc_position(plot=False), shift, rotation_theta, hemisphere='left')    
warped_face = Mapping_som_area_2_vtc(mapping_Left, face_mask[1])
save_warped_area_as_gii(drive_area_2_vtc_axis(warped_face, shift, -rotation_theta, 'left'), 
                        '/results/warped_face_left_sigma6.2.dtseries.nii')
warped_place = Mapping_som_area_2_vtc(mapping_Left, place_mask[1])
save_warped_area_as_gii(drive_area_2_vtc_axis(warped_place, shift, -rotation_theta, 'left'), 
                        '/results/warped_place_left_sigma6.2.dtseries.nii')
warped_limb = Mapping_som_area_2_vtc(mapping_Left, limb_mask[1])
save_warped_area_as_gii(drive_area_2_vtc_axis(warped_limb, shift, -rotation_theta, 'left'), 
                        '/results/warped_limb_left_sigma6.2.dtseries.nii')
warped_object = Mapping_som_area_2_vtc(mapping_Left, object_mask[1])
save_warped_area_as_gii(drive_area_2_vtc_axis(warped_object, shift, -rotation_theta, 'left'), 
                        '/results/warped_object_left_sigma6.2.dtseries.nii')



plt.figure()
plt.imshow(warped_face);plt.axis('off')
plt.savefig('/results/warped_face.png')
plt.figure()
plt.imshow(warped_place);plt.axis('off')
plt.savefig('/results/warped_place.png')
plt.figure()
plt.imshow(warped_limb);plt.axis('off')
plt.savefig('/results/warped_limb.png')
plt.figure()
plt.imshow(warped_object);plt.axis('off')
plt.savefig('/results/warped_object.png')



