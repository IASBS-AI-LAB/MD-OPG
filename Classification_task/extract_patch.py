import numpy as np
import matplotlib.pyplot as plt
import os
from patchify import patchify
import random
from tqdm.notebook import tqdm
from scipy import ndimage
import pandas as pd
import cv2
import json
from glob import glob as g


def load_image(dir):

    im_orig=cv2.imread(dir)
    shapes = im_orig.shape
    return  im_orig, shapes

def make_mask(dir, im_shape):
    # Initialize the masks as zeros
    mask_p = np.zeros((im_shape[0], im_shape[1]))  # Polygon mask for "p"
    mask_o = np.zeros((im_shape[0], im_shape[1]))  # Polygon mask for "o"
    
    zone = None  # To hold the square zone coordinates
    
    f = open(dir)
    data = json.load(f)
    
    for i in range(len(data['shapes'])):
        pts = np.array(data["shapes"][i]["points"], dtype=np.int32)
        
        # Handle polygons with label "p" and "o" as before
        if data['shapes'][i]['label'] == "p":
            mask_p = cv2.fillPoly(mask_p, [pts], 1)
        elif data['shapes'][i]['label'] == "o":
            mask_o = cv2.fillPoly(mask_o, [pts], 1)
        elif data['shapes'][i]['label'] == "z":
            # Handle the square/rectangular zone
            x_min, y_min = pts[0]  # Top-left corner of the zone
            x_max, y_max = pts[1]  # Bottom-right corner of the zone
            zone = (x_min, y_min, x_max, y_max)
        else:
            print("Unknown label in the JSON file.")

    # Crop the mask to the bounding box of the zone if it exists
    #if zone:
    #mask_o = mask_o[zone[1]:zone[3], zone[0]:zone[2]]
    #mask_p = mask_p[zone[1]:zone[3], zone[0]:zone[2]]



def make_patches(img_dirs, label_dirs, mode='train', seed=1234):
    # Lists for patches with non-zero pixels
    x0 = []  
    y0 = []  

    # Lists for patches with all zero pixels
    x_n = []  # Image patches with all zero pixels
    y_n = []  # Corresponding mask patches with all zero pixels

    for img_path in tqdm(img_dirs):
        print("im_path is:",img_path)
        # Load image and its metadata
        image, shapes = load_image(img_path)
        print("image_shape is:",image.shape)
        
        # Prepare the corresponding mask and zone
        temp = img_path.split("_")[1]
        #print("temp is:",temp)
        mask_dir = [s for s in label_dirs if temp in s]
        print("mask_dir_is",mask_dir)
        mask_o, mask_p, zone = make_mask(mask_dir[0], shapes)
        print("shape of zone:",zone)
        print("the_shape_of_original_masK",mask_p.shape)
        # If a zone is defined, crop the image and mask to the zone
        
        image = image[zone[1]:zone[3], zone[0]:zone[2], :]
        mask_p = mask_p[zone[1]:zone[3], zone[0]:zone[2]]
        print(img_path,image.shape , mask_p.shape)
        # Patchify the cropped image and mask into smaller 64x64 patches
        img_patches = patchify(image, (64, 64, 3), step=32)
        mskp_patches = patchify(mask_p, (64, 64), step=32)

        # Iterate over patches
        for i in range(mskp_patches.shape[0]):
            for j in range(mskp_patches.shape[1]):
                # Flatten the mask patch to check for non-zero pixels
                mask_patch = mskp_patches[i][j]

                # If the mask patch contains non-zero pixels, add to non-zero list
                if mask_patch.sum() != 0:
                    x0.append(img_patches[i][j])
                    y0.append(mask_patch)
                # If the mask patch is all zeros, add to zero list
                else:
                    x_n.append(img_patches[i][j])
                    y_n.append(mask_patch)

        # Shuffle the non-zero and zero patches (if in train mode)
        if mode == 'train':
            np.random.seed(seed)
            
            # Shuffle non-zero patches
            z0 = list(zip(x0, y0))
            random.shuffle(z0)
            x0, y0 = zip(*z0)
            x0 = list(x0)
            y0 = list(y0)
            
            # Shuffle zero patches
            z1 = list(zip(x_n, y_n))
            random.shuffle(z1)
            x_n, y_n = zip(*z1)
            x_n = list(x_n)
            y_n = list(y_n)

    
    return x0, y0, x_n, y_n

images = g('/.../*.jpg')
images_test = images[140:]
images = images[0:140]
labels = g('/.../*.json')

print(len(images))
print(len(labels))
print(len(images_test))

x_abnormal , y_abnormal , x_normal, y_normal = make_patches(images[:], labels, mode='train', seed=1234)
