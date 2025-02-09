import numpy as np
from cloudvolume import CloudVolume
import pickle
import pandas as pd

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
print("Loaded segmentation")

#load helper dictionaries that enumerate segment IDs
with open('../../Data/Fig2a_segmentation_dict_full.pkl', 'rb') as f:
    seg_dict = pickle.load(f)
print("Loaded segmentation dict")
with open('../../Data/Fig2a_segmentation_inverted_dict_full.pkl', 'rb') as f:
    inv_seg_dict = pickle.load(f)
print("Loaded segmentation dict inverted")
    
##load connectivity matrix
data = np.load('../../Data/liconn_connectivity_matrix_Fig2a.npz')#\
print("Loaded connectivity matrix")
connectivity_map = data['matches_map']

#load presyn-segmentation corespondance 
seg_presyn_map = np.load('../../Data/liconn_seg_presyn_map_Fig2a.npz')['seg_presyn_map']
print("Loaded presyn_map")
presyn_per_seg = np.sum(seg_presyn_map,axis=1)
seg_per_presyn = np.sum(seg_presyn_map,axis=0)

#load postsyn-segmentation corespondance 
seg_postsyn_map = np.load('../../Data/liconn_seg_postsyn_map_Fig2a.npz')['seg_postsyn_map']
print("Loaded postsyn_map")
postsyn_per_seg = np.sum(seg_postsyn_map,axis=1)
seg_per_postsyn = np.sum(seg_postsyn_map,axis=0)

## analyze axons
axons = pd.read_csv('../../Data/Fig2a_axons.csv', header = None)
axons_ids = list(axons.iloc[:,0].values)

num_of_connected_presyn = []
num_of_connections_with_postsyn=[]
num_of_connected_dendrites= []
connections = []
for index in axons_ids:
    num = presyn_per_seg[inv_seg_dict[index]]
    num_of_connected_presyn.append(num)
    num = np.sum(connectivity_map[inv_seg_dict[index],:])
    connected_struct = []
    for indx in np.nonzero(connectivity_map[inv_seg_dict[index],:])[0]:
        connected_struct.append(seg_dict[indx])
    connections.append(connected_struct)
    num_of_connections_with_postsyn.append(num)
    num_of_connected_dendrites.append(len(connected_struct))

axons['num_of_presyn'] = num_of_connected_presyn
axons['num_of_connections_with_postsyn'] = num_of_connections_with_postsyn
axons['num_of_connected_dendrites'] = num_of_connected_dendrites
axons['connected_dendrites_ids'] = connections
axons.to_csv('Fig2a_axons_connectivity.csv',index=False)

## analyze dendrites

dendrites = pd.read_csv('../../Data/Fig2a_dendrites.csv', header = None)
dendrites_ids = list(dendrites.iloc[:,0].values)

num_of_connected_postsyn = []
num_of_connections_with_presyn=[]
num_of_connected_axons= []
connections = []

for index in dendrites_ids:
    num = postsyn_per_seg[inv_seg_dict[index]]
    num_of_connected_postsyn.append(num)
    num = np.sum(connectivity_map[:,inv_seg_dict[index]])
    connected_struct = []
    for indx in np.nonzero(connectivity_map[:,inv_seg_dict[index]])[0]:
        connected_struct.append(seg_dict[indx])
    connections.append(connected_struct)
    num_of_connections_with_presyn.append(num)
    num_of_connected_axons.append(len(connected_struct))

dendrites['num_of_postsyn'] = num_of_connected_postsyn
dendrites['num_of_connections_with_presyn'] = num_of_connections_with_presyn
dendrites['num_of_connected_axons'] = num_of_connected_axons
dendrites['connected_axons_ids'] = connections
dendrites.to_csv('Fig2a_dendrites_connectivity.csv',index=False)
