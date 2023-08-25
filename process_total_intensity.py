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
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--first_channel', type=int, default=1)
parser.add_argument('--second_channel', type=int, default=None)
parser.add_argument('--first_channel_percentile', type=float, default=85.0)
parser.add_argument('--first_channel_threshold', type=float, default=None)
args = parser.parse_args()

rootDir = "./"

# Set directory
original_image_dir = os.path.join(rootDir, 'original')
painted_image_dir = os.path.join(rootDir, 'painted')
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

segmentation_file_suffix = "png"

fout = open(os.path.join(output_dir, "result.csv"), 'w')
fout.write("fileprefix,intensity\n")

for fileprefixname in fileprefixname_list:

    print("fileprefixname", fileprefixname)

    all_intensity = []
    for filename in all_files:
        if not filename.startswith(fileprefixname): continue

        # Load the painted image and the original image
        original = os.path.join(original_image_dir, filename)
        painted = os.path.join(painted_image_dir, filename[:-len(filename.split(".")[-1])] + segmentation_file_suffix)

        original = cv2.imread(original)
        painted = cv2.imread(painted)

        # Threshold the red channel of the painted image to get the cells
        _, thresholded = cv2.threshold(painted[:, :, 2], 200, 255, cv2.THRESH_BINARY)
        
        # Find the contours (i.e., the cells) in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5000]

        print(f"Frame {filename}: Num of cells detected {len(contours)}")

        valid_mask_first_channel = np.zeros_like(original)
        if args.second_channel != None:
            valid_mask_second_channel = np.zeros_like(original)

        used_contours = []
        # save cross channel used
        for contour in contours:
            # Create a mask of the cell
            mask = np.zeros_like(original)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

            # Use the mask to get the cell's pixels in the original image
            cell_pixels = cv2.bitwise_and(original, mask)

            # Select the cell area with color (considering that the segmentation can be inaccurate)
            first_channel = cell_pixels[:, :, args.first_channel]
            cell_intensity = np.percentile(first_channel[cell_pixels[:, :, args.first_channel] > 0], 20)
            inside_cell = cell_pixels[:, :, args.first_channel] > cell_intensity
            
            # print("cell_intensity", cell_intensity)
            # Filter out the cells with too bright outside range (a trick to check overexpression)
            if cell_intensity < 15: continue

            # Calculate the valid mask of the first channel within the cell
            if len(first_channel[inside_cell]) == 0:
                percentile_intensity = 255
            else:
                percentile_intensity = np.percentile(first_channel[inside_cell], args.first_channel_percentile)
            first_channel_mask = (first_channel > percentile_intensity + 30)
            
            first_channel_mask_3d = np.repeat(first_channel_mask[:, :, np.newaxis], 3, axis=2)
            valid_mask_first_channel = np.logical_or(valid_mask_first_channel, first_channel_mask_3d)
            
            if args.mode == "single_total":
                summation = np.where(first_channel_mask, first_channel, np.zeros_like(first_channel))
                intensity = np.sum(np.float32(summation)) # / np.sum(np.float32(inside_cell == True))
                
            else:
                raise NotImplementedError("mode must be in [single total, coloc total, coloc over first, coloc over second]")
            all_intensity.append(intensity)
            used_contours.append(contour)
        
        # Create an image to visualize the detected contours
        original_copy = copy.deepcopy(original)
        original_coutours = cv2.drawContours(original_copy, used_contours, -1, (0, 255, 255), 3)
        cv2.imwrite(os.path.join(output_dir, filename[:-4] + "_coutours.png"), original_coutours)

        # Draw the first channel used
        masked_image = np.zeros_like(original)
        masked_image = np.where(valid_mask_first_channel > 0, original, masked_image)
        cv2.imwrite(os.path.join(output_dir, filename[:-4] + "_first_channel.png"), masked_image)
        
    # Print the basic numbers
    print("Mean", np.mean(all_intensity))
    
    # Save the data of the current prefix
    for intensity in all_intensity:
        fout.write(f"{fileprefixname},{intensity}\n")

fout.close()    