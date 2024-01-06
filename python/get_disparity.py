import numpy as np
from tqdm import tqdm

def get_disparity(im1, im2, max_disp, window_size):
    # Ensure the images are grayscale
    if len(im1.shape) == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Initialize variables
    rows, cols = im1.shape
    disparity_map = np.zeros((rows, cols), np.float32)

    # Padding images
    padded_im1 = np.pad(im1, ((0, 0), (max_disp, 0)), mode='constant', constant_values=0)
    padded_im2 = np.pad(im2, ((0, 0), (max_disp, 0)), mode='constant', constant_values=0)

    # Sum of Squared Differences (SSD)
    for row in tqdm(range(rows), desc='Calculating disparity map', unit='rows'):
        for col in range(cols):
            min_ssd = float('inf')
            best_disp = 0

            for d in range(max_disp + 1):
                ssd = 0
                for u in range(-window_size // 2, window_size // 2):
                    for v in range(-window_size // 2, window_size // 2):
                        ssd += ((int(padded_im1[row + u, col + v + d]) - int(padded_im2[row + u, col + v])) ** 2)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disp = d

            disparity_map[row, col] = best_disp

    return disparity_map



