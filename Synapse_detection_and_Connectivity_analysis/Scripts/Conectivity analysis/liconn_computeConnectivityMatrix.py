import json
import numpy as np
import pandas as pd
import pickle
from cloudvolume import CloudVolume
import json

def nearest_nonzero_idx_v2(a,x,y):
    tmp = a[x,y,0,0]
    a[x,y,0,0] = 0
    r,c = np.nonzero(a[:,:,0,0])
    a[x,y,0,0] = tmp
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    min_dist = ((r - x)**2 + (c - y)**2).min()
    return r[min_idx], c[min_idx], min_dist

##load synapse detections
presyn_coords = []
with open("../../Data/Fig2a_presyn_coords_all.txt") as f:
    lines = f.readlines() # list containing lines of file
    i = 1
    for line in lines:
        line = line.strip()
        if line:
            data = [item.strip() for item in line.split(',')]
            one_line = []
            for index, elem in enumerate(data):
                if data[index][-1]==']':
                    data[index] = data[index][:-1]
                if data[index][0]=='[':
                    data[index] = data[index][1:]
                one_line.append(data[index])
            float_str = [float(k) for k in one_line]
            presyn_coords.append(float_str) # append dictionary to list
print(len(presyn_coords))

postsyn_coords = []
with open("../../Data/Fig2a_postsyn_coords_all.txt") as f:
    lines = f.readlines() # list containing lines of file
    i = 1
    for line in lines:
        line = line.strip()
        if line:
            data = [item.strip() for item in line.split(',')]
            one_line = []
            for index, elem in enumerate(data):
                if data[index][-1]==']':
                    data[index] = data[index][:-1]
                if data[index][0]=='[':
                    data[index] = data[index][1:]
                one_line.append(data[index])
            float_str = [float(k) for k in one_line]
            postsyn_coords.append(float_str) # append dictionary to list
print(len(postsyn_coords))

#load segmentation
vol = CloudVolume('file://../../Data/Fig2a_segmentation', parallel=True, progress=True)

#load helper dictionaries that enumerate segment IDs
with open('../../Data/Fig2a_segmentation_dict_full.pkl', 'rb') as f:
    seg_dict = pickle.load(f)
with open('../../Data/Fig2a_segmentation_inverted_dict_full.pkl', 'rb') as f:
    inv_seg_dict = pickle.load(f)

seg_presyn_map = np.zeros((len(seg_dict), len(presyn_coords)))
seg_postsyn_map = np.zeros((len(seg_dict), len(postsyn_coords)))
connectivity_matrix = np.zeros((len(seg_dict), len(seg_dict)))

#find corresponding segment ID for each postsynapse
dendrites = pd.read_csv('../../Data/Fig2a_dendrites.csv', header = None)
dendrites_ids = list(dendrites.iloc[:,0].values)
postsyn_rounded = np.stack(postsyn_coords).astype('uint32')

for i in range(len(postsyn_rounded)):
    z,y,x = postsyn_rounded[i]
    underlying_seg_id = vol[int(x/2), int(y/2), int(z/2)] #segmentation is x2 downscaled in all 3 dimentions
    print(underlying_seg_id[0][0][0][0])
    if underlying_seg_id!=0:
        seg_postsyn_map[inv_seg_dict[underlying_seg_id[0][0][0][0]],i]=1
    else:
        underlying_seg_area = vol[(int(x/2)-10):(int(x/2)+10), (int(y/2)-10):(int(y/2)+10), int(z/2)]
        if np.amax(underlying_seg_area>0):
            nearest_nonzero1,nearestnonzero2,dist = nearest_nonzero_idx_v2(underlying_seg_area,10,10)
            if underlying_seg_area[nearest_nonzero1,nearestnonzero2,0,0] in dendrites_ids:
                if dist<=30:
                    print(underlying_seg_area[nearest_nonzero1,nearestnonzero2,0,0])
                    seg_postsyn_map[inv_seg_dict[underlying_seg_area[nearest_nonzero1,nearestnonzero2,0,0]],i]=1
np.savez('liconn_seg_postsyn_map_Fig2a.npz', seg_postsyn_map=seg_postsyn_map)

#find corresponding segment ID for each presynapse
axons = pd.read_csv('../../Data/Fig2a_axons.csv', header = None)
axons_ids = list(axons.iloc[:,0].values)
presyn_rounded = np.stack(presyn_coords).astype('uint32')
for i in range(len(presyn_rounded)):
    z,y,x = presyn_rounded[i]
    underlying_seg_id = vol[int(x/2), int(y/2), int(z/2)]
    print(underlying_seg_id[0][0][0][0])
    if underlying_seg_id!=0:
        seg_presyn_map[inv_seg_dict[underlying_seg_id[0][0][0][0]],i]=1
    else:
        underlying_seg_area = vol[(int(x/2)-10):(int(x/2)+10), (int(y/2)-10):(int(y/2)+10), int(z/2)]
        if np.amax(underlying_seg_area>0):
            nearest_nonzero1,nearestnonzero2,dist = nearest_nonzero_idx_v2(underlying_seg_area,10,10)
            if underlying_seg_area[nearest_nonzero1,nearestnonzero2,0,0] in axons_ids:
                if dist<=30:
                    print(underlying_seg_area[nearest_nonzero1,nearestnonzero2,0,0])
                    seg_presyn_map[inv_seg_dict[underlying_seg_area[nearest_nonzero1,nearestnonzero2,0,0]],i]=1
np.savez('liconn_seg_presyn_map_Fig2a.npz', seg_presyn_map=seg_presyn_map)

#load full synapse detections (pre-post matches)
data = np.load('../../Data/syn_matches_map_Fig2a.npz')#\
matches_map = data['matches_map']
#compute connectivity matrix
for i in range(matches_map.shape[0]):
    print(i)
    for j in range(matches_map.shape[1]):
        if matches_map[i,j]!=0:
            post_id = i
            pre_id = j
            presyn_seg_id = np.nonzero(seg_presyn_map[:,pre_id])[0]
            postsyn_seg_id = np.nonzero(seg_postsyn_map[:,post_id])[0]
            connectivity_matrix[presyn_seg_id,postsyn_seg_id]+=1
np.savez('liconn_connectivity_matrix_Fig2a.npz', matches_map=connectivity_matrix)

