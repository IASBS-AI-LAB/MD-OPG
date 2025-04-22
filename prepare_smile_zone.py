import cv2
import json
import numpy as np
from glob import glob as g

def load_image(dir):

    im_orig=cv2.imread(dir)
    #im = np.where(im_orig>50,1,0)
    shapes = im_orig.shape[:2]
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
            mask_o = cv2.fillPoly(mask_p, [pts], 1)
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

    return mask_o, mask_p, zone

output_dir_IMG = '/.../S_Z_IMG'
output_dir_MSK = '/.../S_Z_MSK'

label_dirs = g('/.../*.json')
img_path = g('/.../*.jpg')

max_width = 1000
max_height = 440



for i in img_path:
    image, shapes = load_image(i)
    temp = i.split("/")[-1]
    temp = temp.split(".")[0]
    mask_dir = [s for s in label_dirs if temp == s.split("/")[-1].split(".")[0]]
    mask_o, mask_p, zone = make_mask(mask_dir[0], shapes)
    image = image[zone[1]:zone[3], zone[0]:zone[2], :]
    mask_p = mask_p[zone[1]:zone[3], zone[0]:zone[2]]
    mask_p = np.expand_dims(mask_p, axis=2)
      # Get cropped dimensions
    height ,width,_ = mask_p.shape  # Correct order: (height, width)
    #print(width, height)
    print(mask_p.shape)
    
    # Calculate padding
    if height > max_height or width > max_width:
        print(f"Skipping {temp} due to size exceeding target dimensions")
        continue

    top_pad = (max_height - height) // 2
    bottom_pad = max_height - height - top_pad
    left_pad = (max_width - width) // 2
    right_pad = max_width - width - left_pad

    # Apply padding (black pixels for image, 0 for mask)
    padded_img = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_msk = cv2.copyMakeBorder(mask_p, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0,0,0])
    print(padded_img.shape)
    # Save the padded image and mask
    
    # Save the padded image
    img_save_path = os.path.join(output_dir_IMG, f"{temp}.jpg")
    cv2.imwrite(img_save_path, padded_img)

    # Convert mask to uint8 and save
    padded_msk_uint8 = (padded_msk * 255).astype(np.uint8)  # Scale and convert
    mask_save_path = os.path.join(output_dir_MSK, f"{temp}.png")
    cv2.imwrite(mask_save_path, padded_msk_uint8)
    print(padded_msk_uint8.shape)
    print(f"Saved: {img_save_path}, {mask_save_path}")