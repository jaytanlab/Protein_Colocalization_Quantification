# python process.py --mode single_total

import os
import cv2
import csv
import copy
import numpy as np
from argparse import ArgumentParser

# Set color random seed
np.random.seed(2023)

# Red 2, Green 1, Blue 0
# Set input arguments
parser = ArgumentParser()
parser.add_argument('--first_channel', type=int, default=0)
args = parser.parse_args()

rootDir = "./"

# Set directory
original_image_dir = os.path.join(rootDir, 'original')
output_dir = os.path.join(rootDir, "result")

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get files
fileprefixname_list = []
all_files = os.listdir(original_image_dir)
all_files.sort()
for filename in all_files:
    if filename.endswith(('.png', '.tif', '.jpg')): # lossless format
        file_suffix = filename.split("-")[-1]
        file_suffix = file_suffix.split("_")[-1]
        true_name = filename[:-len(file_suffix)]
        if true_name not in fileprefixname_list:
            fileprefixname_list.append(true_name)
fileprefixname_list.sort()

fout = open(os.path.join(output_dir, "result.csv"), 'w')
fout.write("fileprefix,intensity\n")

for fileprefixname in fileprefixname_list:

    print("fileprefixname", fileprefixname)

    cell_numbers = []
    for filename in all_files:
        if not filename.startswith(fileprefixname): continue

        # Load the painted image and the original image
        original = os.path.join(original_image_dir, filename)

        original = cv2.imread(original)
        
        percentile = np.percentile(original[:, :, args.first_channel], 77)
        
        # Threshold the red channel of the painted image to get the cells
        _, thresholded = cv2.threshold(original[:, :, args.first_channel], percentile, 255, cv2.THRESH_BINARY)
        
        # Find the contours (i.e., the cells) in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]

        print(f"Frame {filename}: Num of cells detected {len(contours)}")

        valid_mask_first_channel = np.zeros_like(original)

        used_contours = []
        # save cross channel used
        valid_cell_num = 0
        for contour in contours:
            # Create a mask of the cell
            mask = np.zeros_like(original)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

            # Use the mask to get the cell's pixels in the original image
            cell_pixels = cv2.bitwise_and(original, mask)
            first_channel = cell_pixels[:, :, args.first_channel]
            if np.float32(first_channel > 250).sum() > 50:
                continue
            valid_cell_num += 1
            used_contours.append(contour)
        cell_numbers.append(valid_cell_num)
        print(f"Frame {filename}: valid_cell_num {valid_cell_num}")
        
        # Create an image to visualize the detected contours
        original_copy = copy.deepcopy(original)
        original_coutours = cv2.drawContours(original_copy, used_contours, -1, (0, 255, 255), 3)
        cv2.imwrite(os.path.join(output_dir, filename[:-4] + "_coutours.png"), original_coutours)

    # Print the basic numbers
    print("Mean", np.mean(cell_numbers))
    
    # Save the data of the current prefix
    for cell_number in cell_numbers:
        fout.write(f"{fileprefixname},{cell_number}\n")

fout.close()    