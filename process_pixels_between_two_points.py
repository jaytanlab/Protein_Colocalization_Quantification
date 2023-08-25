import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import savgol_filter
from scipy.signal import convolve2d

def smooth_savgol(y, window_length=5, polyorder=3):
    # The window_length needs to be an odd number. Adjust if needed.
    if window_length % 2 == 0:
        window_length += 1

    return savgol_filter(y, window_length=window_length, polyorder=polyorder)

def get_white_points(img):
    data = np.array(img)

    # Define a 3x3 filter of ones
    kernel = np.ones((3, 3))

    # Compute the sum of the 3x3 region around each pixel for each channel
    r_conv = convolve2d(data[:, :, 0], kernel, mode='same')
    g_conv = convolve2d(data[:, :, 1], kernel, mode='same')
    b_conv = convolve2d(data[:, :, 2], kernel, mode='same')

    # Check where the convolved output is 2295 (9 * 255) for all channels
    white_pixels = np.where((r_conv == 2295) & (g_conv == 2295) & (b_conv == 2295))
    
    sorted_pixels = sorted(zip(white_pixels[0], white_pixels[1]))
    topmost = sorted_pixels[0]
    bottommost = sorted_pixels[-1]
    
    return topmost, bottommost

def get_pixel_values_on_segment(img, start, end):
    data = np.array(img)
    length = int(np.hypot(end[1]-start[1], end[0]-start[0]))
    x = np.linspace(start[1], end[1], length)
    y = np.linspace(start[0], end[0], length)
    
    r_values = data[y.astype(int), x.astype(int), 0].astype(np.float64)
    g_values = data[y.astype(int), x.astype(int), 1].astype(np.float64)
    b_values = data[y.astype(int), x.astype(int), 2].astype(np.float64)
    
    return r_values, g_values, b_values

def visualize_rgb(r, g, b, image_i):
    r /= np.max(r)
    g /= np.max(g)
    b /= np.max(b)

    r = smooth_savgol(r)
    g = smooth_savgol(g)
    b = smooth_savgol(b)

    x = np.arange(len(r))
    
    # First Image: Green and Blue
    plt.figure()
    g_smooth = g
    b_smooth = b
        
    plt.plot(x, g_smooth, color='green', label='Green')
    plt.plot(x, b_smooth, color='blue', label='Blue')
    plt.legend()
    plt.title("Green and Blue Values Along the Segment")
    plt.xlabel("Pixel Along the Segment")
    plt.ylabel("Normalized Value")
    plt.savefig(f"./output_gb_{image_i}.png")
    plt.close()
    
    # Second Image: Red and Blue
    plt.figure()
    r_smooth = r
    b_smooth = b
    
    plt.plot(x, r_smooth, color='red', label='Red')
    plt.plot(x, b_smooth, color='blue', label='Blue')
    plt.legend()
    plt.title("Red and Blue Values Along the Segment")
    plt.xlabel("Pixel Along the Segment")
    plt.ylabel("Normalized Value")
    plt.savefig(f"./output_rb_{image_i}.png")
    plt.close()

def export_to_csv(r, g, b, image_i):
    df = pd.DataFrame({'Red': r, 'Green': g, 'Blue': b})
    df.to_csv(f"rgb_values_{image_i}.csv", index=False)

def main():
    for image_i in range(0, 11):
        input_image = Image.open("input.png")
        input_selected_image = Image.open(f"input_selected_{image_i}.png")

        start_point, end_point = get_white_points(input_selected_image)
        print("start_point", start_point)
        print("end_point", end_point)
        r_values, g_values, b_values = get_pixel_values_on_segment(input_image, start_point, end_point)
        export_to_csv(r_values, g_values, b_values, image_i)
        visualize_rgb(r_values, g_values, b_values, image_i)

if __name__ == "__main__":
    main()