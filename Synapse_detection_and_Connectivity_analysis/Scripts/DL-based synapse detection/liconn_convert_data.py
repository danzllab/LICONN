import os
from skimage.io import imread
import tifffile
import numpy as np
import zarr
from skimage.util import invert

import tifffile
from scipy import ndimage as ndi
import pandas as pd
from skimage import filters
from skimage.filters import threshold_otsu,threshold_local
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.morphology import ball, binary_dilation
from skimage.measure import label, regionprops

def erase_small_in_z(labels, min_size):
    rps = regionprops(labels)
    for r in rps:
        len_in_z = r.bbox[3]-r.bbox[0]
        if len_in_z<min_size:
            labels[labels==r.label]=0
    return labels

def erase_small(labels, min_size):
    unique, counts = np.unique(labels, return_counts=True)
    for i in range(len(unique)):
        if counts[i]<min_size:
            labels[labels==unique[i]]=0
    return labels

#functions
#remove_background
def remove_backg(img, sig1=1,sig2=15):
    ('removing background from immuno channel')
    img_sig =ndi.gaussian_filter(img, (sig1, sig1,sig1))
    img_bcg = ndi.gaussian_filter(img, (sig2,sig2,sig2))
    cleaned = img_sig - img_bcg
    cleaned[cleaned<0]=0
    return cleaned

def create_immuno_mask(img, bcg):
    mi = 101
    ma = np.percentile(img, 99.995)
    img_rescale = rescale_intensity(img, in_range=(mi, ma))
    img_clean =remove_backg(img_rescale,0.5,10)
    
    threshold =  threshold_otsu(img_clean)
    print(threshold)
    mask = img_clean > threshold
    labels = label(mask)
    rp = regionprops(labels, intensity_image = bcg)
    max_intensity = []
    labels_list = []
    for r in rp:
        labels_list.append(r.label)
        max_intensity.append(np.amax(r.intensity_image))
    data = list(zip(labels_list, max_intensity))
    df = pd.DataFrame(data = data ,columns = ["label", "max_intensity"])
    intensity_threshold = threshold_otsu(df["max_intensity"].values)
    df['is_synapse'] = np.where(df['max_intensity'] > intensity_threshold, 'Yes', 'No')
    labels_real = df[df["is_synapse"]=='Yes']
    real_labels = list(labels_real["label"])
    img_real_labels = np.isin(labels,real_labels)
    img_real_labels  = labels*img_real_labels
    mask_filtered = img_real_labels>0
    return mask_filtered

#list of images to be converted
paths = [ 
'../../Data/LICONN_test_dataset.tif']

for p in paths:
    # 1. read image 
    raw_data = imread(p)
    raw_data = raw_data.astype('float32')

    print(raw_data.shape)
    name_zarr = 'LICONN_test_dataset.zarr'

    liconn = raw_data[:,:,:,2]
    bassoon = raw_data[:,:,:,1]
    shank2 = raw_data[:,:,:,0]
    mask_bassoon = create_immuno_mask(bassoon,liconn)
    fake_bassoon_signal = ndi.gaussian_filter(mask_bassoon.astype('float32'),(1,1,1))
    mask_shank2 = create_immuno_mask(shank2,liconn)
    fake_shank2_signal = ndi.gaussian_filter(mask_shank2.astype('float32'),(1,1,1))

    file = zarr.open(os.path.join(os.path.dirname(__file__), name_zarr), "a") 
    root_group = zarr.group()
    
    print(p)
    for ds_name, data in [
        ("liconn_data_raw", liconn),
        ("bassoon_data", fake_bassoon_signal),
        ("basson_mask", mask_bassoon),
        ("shank2_data", fake_shank2_signal),
        ("shank2_mask", mask_shank2),


    ]:
        
        if "liconn" in ds_name:
            pmin, pmax = np.percentile(
                data, (0, 100)
            )  
            data = (data - pmin) / (pmax - pmin)  # scale range
            data *= 255  # scale to uint8 range
            data = np.clip(data, 0, 255)  # clip at [0, 255]
        
            data = data.astype('uint8') 
        if "bassoon_data" in ds_name:
            pmin, pmax = np.percentile(
                data, (0, 100)
            )  
            data = (data - pmin) / (pmax - pmin)  # scale range
            data *= 255  # scale to uint8 range
            data = np.clip(data, 0, 255)  # clip at [0, 255]
        
            data = data.astype('uint8')
        if "shank2_data" in ds_name:
            pmin, pmax = np.percentile(
                data, (0, 100)
            )  
            data = (data - pmin) / (pmax - pmin)  #  scale range
            data *= 255  # scale to uint8 range
            data = np.clip(data, 0, 255)  # clip at [0, 255]
        
            data = data.astype('uint8')
        if "mask" in ds_name:
            data = data.astype('float32')

        resolution = [300,150,150]  # (z,y,x)
        print('processing data')
        file[f"volumes/{ds_name}"] = data
        file[f"volumes/{ds_name}"].attrs["offset"] = [0] * 3
        file[f"volumes/{ds_name}"].attrs[
            "resolution"
        ] = resolution  # [z,y,x] voxel size
