import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_camera_parameters(camera_params_str):
    # Split the camera parameter string into parts and convert it to a floating point number
    params = [float(x) for x in camera_params_str.split()]
    
    # K
    K = np.array([[params[0], params[1], params[2]],
                  [params[3], params[4], params[5]],
                  [params[6], params[7], params[8]]])
    
    # R & t 
    R = np.array([[params[9], params[10], params[11]],
                  [params[12], params[13], params[14]],
                  [params[15], params[16], params[17]]])
    t = np.array([[params[18]],
                  [params[19]],
                  [params[20]]])
    
    Rt = np.hstack((R, t))
    return K, Rt

def load_images_and_params():
    images = []
    camera_matrices = []

    # open file
    with open('../data/templeR_par.txt', 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            parts = line.split()
            image_name = '../data/' + parts[0]
            params_str = ' '.join(parts[1:])
            
            # read image
            image = cv2.imread(image_name)

            # set backgroud to pure black
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_threshold = 50
            _, thresholded = cv2.threshold(hsv[:, :, 2], v_threshold, 255, cv2.THRESH_BINARY)
            black_background = np.where(thresholded[..., None] == 255, image, 0)
            black_background_rgb = cv2.cvtColor(black_background, cv2.COLOR_BGR2RGB)

            images.append(black_background_rgb)
            
            # get K and Rt from files
            K, Rt = parse_camera_parameters(params_str)
            camera_matrices.append((K, Rt))

    return images, camera_matrices

def get_3d_coord(q, P, d):
    # q is coordiante (x, y)
    # P is projection matrix
    # d is depth

    P3_3 = P[:, :3]
    P4_3 = P[:, 3]

    q_homogeneous = np.array([q[0], q[1], 1])

    rhs = d * q_homogeneous - P4_3

    X_3D = np.linalg.inv(P3_3).dot(rhs)
    
    return X_3D

def project_points(X, P):
    # Project the 3D point onto the 2D image plane
    # X is a 3D coordinate point
    # K is the internal parameter matrix
    # P is the external parameter matrix (R|t)
    X_homogeneous = np.append(X, 1)  # (x, y, z, 1)
    pixel_coord_homogeneous = P.dot(X_homogeneous)
    x, y, d = pixel_coord_homogeneous / pixel_coord_homogeneous[2]

    return (x, y), d 

def project_3d_to_2d(X, P):
    # Project the 3D point onto the 2D image plane
    # X is a 3D coordinate point
    # K is the internal parameter matrix
    # P is the external parameter matrix (R|t)
    homogenous_X = np.hstack((X, np.ones((X.shape[0], 1))))
    projected = P.dot(homogenous_X.T).T
    projected /= projected[:, 2:3]
    return np.round(projected[:, :2]).astype(int)

#  Collect colors for the given coordinates from image I.
def collect_colors(I, coordinates):
    colors = []
    for x, y in coordinates:
        if 0 <= x < I.shape[1] and 0 <= y < I.shape[0]:
            colors.append(I[y, x])
    return np.array(colors)

def compute_consistency(I0, I1, X, P0, P1):
    # Project S^2 3d coordinates in X into image I0.
    coords_I0 = project_3d_to_2d(X, P0)
    # Project S^2 3d coordinates in X into image I1.
    coords_I1 = project_3d_to_2d(X, P1)

    # Collect S^2 pixel colors C0.
    C0 = collect_colors(I0, coords_I0)
    # Collect S^2 pixel colors C1.
    C1 = collect_colors(I1, coords_I1)

    # Return NormalizedCrossCorrelation(C0, C1).
    return normalized_cross_correlation(C0, C1)

def normalized_cross_correlation(C0, C1):
    # Returns 0 if one or both arrays are empty
    if C0.size == 0 or C1.size == 0:
        return 0  
    # Converts a one-dimensional array to a two-dimensional array
    if C0.ndim == 1:
        C0 = C0[np.newaxis, :]  
    if C1.ndim == 1:
        C1 = C1[np.newaxis, :]

    # Compute average red, average green, and average blue of pixels in C0 an C1.
    mean_C0 = np.mean(C0, axis=(0, 1))
    mean_C1 = np.mean(C1, axis=(0, 1))

    # Subtract average red, average green, and average blue from each pixel color in C0 and C1
    C0_normalized = C0 - mean_C0
    C1_normalized = C1 - mean_C1

    # Compute the L2 norm of all intensities (red, green, and blue together)
    norm_C0 = np.linalg.norm(C0_normalized, ord=2, axis=(0, 1), keepdims=True)
    norm_C1 = np.linalg.norm(C1_normalized, ord=2, axis=(0, 1), keepdims=True)

    # If both norms are zero, the regions are perfectly correlated
    if np.all(norm_C0 == 0) and np.all(norm_C1 == 0):
        return 1.0  # Return the maximum NCC score

    # Avoid division by zero
    if np.any(norm_C0 == 0) or np.any(norm_C1 == 0):
        return 0

    # Divide each pixel color by the L2 norm
    C0_normalized /= norm_C0
    C1_normalized /= norm_C1

    # Consider C0 (and C1) to be a 1D vector
    C0_vector = C0_normalized.flatten()
    C1_vector = C1_normalized.flatten()

    # Return a dot product 
    ncc_score = np.dot(C0_vector, C1_vector)

    return ncc_score


def create_depth_map(images, camera_matrices, min_depth, max_depth, depth_step, S=5, consistency_threshold=0.95):
    # reference image
    I0 = images[0]
    # get projection matrix
    K0 = camera_matrices[0][0]
    Rt0 = camera_matrices[0][1]
    P0 = np.dot(K0, Rt0)
    
    depth_map = np.zeros((I0.shape[0], I0.shape[1]), dtype=np.float32)
    half_S = S // 2
    
    # loop through all pixel in I0
    for y in  tqdm(range(I0.shape[0]), desc='Calculating depth map', unit='rows'):
        for x in range(I0.shape[1]):
            # if pixel value is black, skip
            if np.all(I0[y, x] == [0, 0, 0]): 
                continue

            S2_coordinates = [(ix, iy) for iy in range(int(y - half_S), int(y + half_S) + 1) 
                          for ix in range(int(x - half_S), int(x + half_S) + 1)]
            # flattened_S2_pixels_coordinates = S2_coordinates.flatten()

            best_score = -np.inf
            best_depth = 0
            
            # loop through depth
            for d in tqdm(np.arange(min_depth, max_depth, depth_step), desc=f'Calculating best depth for point ({x}, {y})', unit='rows'):
            # for d in np.arange(min_depth, max_depth, depth_step):
                scores = []
                X = []
                
                # get all 3d coordinate in S^2 pixels around (y. x) 
                for p in S2_coordinates:
                    coord_3d = get_3d_coord(p, P0, d)
                    X.append(coord_3d)

                X = np.array(X)
                
                # calculate score base on I1, I2, I3
                for i in range(1, 5):  
                    Ki = camera_matrices[i][0]
                    Rti = camera_matrices[i][1]
                    Pi = np.dot(Ki, Rti)
                    score = compute_consistency(I0, images[i], X, P0, Pi)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                # print(avg_score)

                if avg_score < consistency_threshold:
                    continue  # If below the threshold, skip this depth
                
                # update the best score
                if avg_score > best_score:
                    best_score = avg_score
                    best_depth = d
            
            # If the best score is above the threshold, the best depth is set to depth map
            if best_score >= consistency_threshold:
                depth_map[y, x] = best_depth
                # print(depth_map[y, x])
    
    return depth_map

def visualize_corners(images, camera_matrices, bbox_corners):
    for idx, (image, (K, Rt)) in enumerate(zip(images, camera_matrices)):
        P = np.dot(K, Rt) 
        projected_corners = [project_points(corner, P)[0] for corner in bbox_corners]

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
        for corner in projected_corners:
            x, y = corner
            plt.scatter(x, y, color='red', s=40) 
        plt.title(f'Image {idx} Projected Corners')
        plt.axis('off')
        plt.show()

def save_point_cloud(depth_map, camera_params, image, filename='../results/point_cloud.obj'):
    with open(filename, 'w') as file:
        K, Rt = camera_params
        P = np.dot(K, Rt)
        for y in tqdm(range(depth_map.shape[0]), desc='Saving point cloud'):
            for x in range(depth_map.shape[1]):
                if 0.45 < depth_map[y, x] < 0.6:
                    X = get_3d_coord((x, y), P, depth_map[y, x])
                    file.write(f'v {X[0]} {X[1]} {X[2]}\n')
                    # # optional: with color
                    # color = image[y, x]  
                    # file.write(f'vn {color[2]/255.0} {color[1]/255.0} {color[0]/255.0}\n')

# def test_compute_consistency():
#     # Define the size of the synthetic image
#     img_size = (100, 100, 3)

#     # Create synthetic images with consistent colors
#     I0 = np.ones(img_size) * np.array([255, 0, 0])  # Entirely red image
#     I1 = np.ones(img_size) * np.array([255, 0, 0])  # Entirely red image, for consistency

#     # Generate a synthetic 3D point that lies in front of both cameras
#     X = np.array([50, 50, 1])  # This should project roughly to the center of the image

#     # Create synthetic camera projection matrices
#     # Assuming an orthographic projection for simplicity
#     P0 = np.array([[1, 0, 0, 0],
#                    [0, 1, 0, 0],
#                    [0, 0, 1, 0]])
#     P1 = np.array([[1, 0, 0, 0],
#                    [0, 1, 0, 0],
#                    [0, 0, 1, 0]])

#     # Call compute_consistency
#     consistency_score = compute_consistency(I0, I1, X, P0, P1)

#     # Print intermediate results
#     print(f"Projected point in I0: {project_points(X, P0)[0]}")
#     print(f"Projected point in I1: {project_points(X, P1)[0]}")
#     print(f"Consistency score: {consistency_score}")
    
#     # Check the result
#     if not np.isnan(consistency_score) and consistency_score != 0:
#         print("Test Passed: Consistency score is non-zero and not NaN")
#     else:
#         print("Test Failed: Consistency score is zero or NaN")



def main():
    images, camera_matrices = load_images_and_params()

    # # 3.4.1
    # bbox_min = np.array([-0.023121, -0.038009, -0.091940])
    # bbox_max = np.array([0.078626, 0.121636, -0.017395])
    # bbox_corners = np.array([[bbox_min[0], bbox_min[1], bbox_min[2]],  # Bottom-Back-Left
    #                          [bbox_max[0], bbox_min[1], bbox_min[2]],  # Bottom-Back-Right
    #                          [bbox_min[0], bbox_max[1], bbox_min[2]],  # Bottom-Front-Left
    #                          [bbox_max[0], bbox_max[1], bbox_min[2]],  # Bottom-Front-Right
    #                          [bbox_min[0], bbox_min[1], bbox_max[2]],  # Top-Back-Left
    #                          [bbox_max[0], bbox_min[1], bbox_max[2]],  # Top-Back-Right
    #                          [bbox_min[0], bbox_max[1], bbox_max[2]],  # Top-Front-Left
    #                          [bbox_max[0], bbox_max[1], bbox_max[2]]]) # Top-Front-Right

    # visualize_corners(images, camera_matrices, bbox_corners)

    # 3.4.2
    depth_map = create_depth_map(images, camera_matrices, min_depth=0.48, max_depth=0.6, depth_step=0.001, S=5, consistency_threshold=0.95)

    # visualize depth map
    plt.figure(figsize=(10, 10))
    plt.imshow(depth_map, cmap='gray', vmin=0, vmax=1) 
    plt.colorbar() 
    plt.title('Depth Map')
    plt.axis('off')
    plt.savefig('../results/d1.png', dpi=300)
    plt.close()

    # visualize depth map
    im0 = cv2.imread('../data/templeR0013.png', cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 10))
    plt.imshow(depth_map*(im0 > 40), cmap='gray')  
    plt.colorbar() 
    plt.title('Depth Map')
    plt.axis('off')
    plt.savefig('../results/d2.png', dpi=300)
    plt.close()


    # save for visualization in mashlab
    save_point_cloud(depth_map, camera_matrices[0], images[0], filename='../results/point_cloud.obj')

    # # Run the test function
    # test_passed = test_compute_consistency()

if __name__ == '__main__':  
    main()


