import numpy as np
import cv2

def epipolarCorrespondence(im1, im2, F, pts1):
    """
    Args:
        im1:    Image 1
        im2:    Image 2
        F:      Fundamental Matrix from im1 to im2
        pts1:   coordinates of points in image 1
    Returns:
        pts2:   coordinates of points in image 2
    """

    pts2 = []
    # adjust window size here
    window_size = 15
    half_window_size = window_size // 2

    # Make sure the input images are in grayscale
    if len(im1.shape) == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # loop all points in pts1, find corresoonding points in pts2
    for pt1 in pts1:
        # get xy coordiantes of pt1
        x1, y1 = int(pt1[0]), int(pt1[1])

        # Compute the epipolar line in the second image
        v = np.array([x1, y1, 1])
        epline = np.dot(F, v.T)

        # get all possible corresoonding x coordinates in pts2 
        x2s = np.arange(im2.shape[1])
        # y = (-C - Ax) / B
        y2s = (-epline[2] - epline[0] * x2s) / epline[1]

        # get window of pt1
        pt1_window = im1[y1-half_window_size : y1+half_window_size+1, x1-half_window_size : x1+half_window_size+1]
        
        # init corresponding pt2
        opt_pt2 = None
        max_ncc_score = -np.inf

        # loop through x2, y2 in the epline in img2
        for x2, y2 in zip(x2s, y2s):
            x2, y2 = int(x2), int(y2)
            if y2-half_window_size >= 0 and y2+half_window_size < im2.shape[0] and x2-half_window_size >= 0 and x2+half_window_size < im2.shape[1]:
                pt2_window = im2[y2-half_window_size:y2+half_window_size+1, x2-half_window_size:x2+half_window_size+1]

                # If the window goes outside the bounds of the images, skip this candidate
                if pt1_window.shape != pt2_window.shape or pt1_window.size == 0 or pt2_window.size == 0:
                    continue

                # Normalize the patches (windows) before computing NCC
                pt1_window_normalized = (pt1_window - np.mean(pt1_window)) / (np.std(pt1_window) + 1e-10)
                pt2_window_normalized = (pt2_window - np.mean(pt2_window)) / (np.std(pt2_window) + 1e-10)

                # Compute normalized cross-correlation
                ncc_score = np.mean(pt1_window_normalized * pt2_window_normalized)

                # update best pt2
                if ncc_score > max_ncc_score:
                    max_ncc_score = ncc_score
                    opt_pt2 = (x2, y2)
                
        pts2.append(opt_pt2)
        
    pts2 = np.array(pts2)

    return pts2
