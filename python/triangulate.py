import numpy as np

def triangulate(P1, pts1, P2, pts2):
    num_points = pts1.shape[0]
    pts3d = np.zeros((num_points, 3))
    pts_err1 = 0
    pts_err2 = 0

    for i in range(num_points):
        # Construct matrix A for homogeneous equation system Ax = 0
        # Such that A = [P1; P2] and x = [X, Y, Z, 1]^T
        # Extract the point coordinates
        A = np.vstack([
            pts1[i, 1] * P1[2, :] - P1[1, :],
            P1[0, :] - pts1[i, 0] * P1[2, :],
            pts2[i, 1] * P2[2, :] - P2[1, :],
            P2[0, :] - pts2[i, 0] * P2[2, :]
        ])
        
        # Solve for the 3D point using SVD
        U, S, Vt = np.linalg.svd( A)
        X = Vt[-1]
        pts3d[i] = (X / X[3])[:3]  # Normalize the homogeneous coordinates


    # Convert pts3d to homogeneous coordinates for re-projection
    pts3d_homogeneous = np.hstack((pts3d, np.ones((num_points, 1))))

    # Re-project to image planes
    re_pro1 = P1 @ pts3d_homogeneous.T
    re_pro2 = P2 @ pts3d_homogeneous.T

    # Normalize the re-projected points
    re_pro1 /= re_pro1[2, :]
    re_pro2 /= re_pro2[2, :]

    # Calculate reprojection error
    for i in range(num_points):
        err1 = np.sqrt((pts1[i, 0] - re_pro1[0, i])**2 + (pts1[i, 1] - re_pro1[1, i])**2)
        err2 = np.sqrt((pts2[i, 0] - re_pro2[0, i])**2 + (pts2[i, 1] - re_pro2[1, i])**2)
        pts_err1 += err1
        pts_err2 += err2

    # Calculate average error
    pts_err1 /= num_points
    pts_err2 /= num_points

    print("Re-projection Error of pts1 and pts2:", pts_err1, pts_err2)

    return pts3d