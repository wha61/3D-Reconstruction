import numpy as np
from numpy.linalg import svd
from refineF import refineF

def eightpoint(pts1, pts2, M):
    """
    eightpoint:
        pts1 - Nx2 matrix of (x,y) coordinates
        pts2 - Nx2 matrix of (x,y) coordinates
        M    - max(imwidth, imheight)
    """
    

    # Implement the eightpoint algorithm
    # Generate a matrix F from correspondence '../data/some_corresp.npy'
    F = None

    # 0. (Normalize points)
    pts1 = pts1 / M
    pts2 = pts2 / M

    # 1. Construct the M x 9 matrix A
    # get row of pts1
    N = pts1.shape[0]
    # initialize A
    A = np.zeros((N, 9))
    # substitute values
    for i in range(N):
        A[i] = [pts2[i, 0] * pts1[i, 0], pts2[i, 0] * pts1[i, 1], pts2[i, 0],
                pts2[i, 1] * pts1[i, 0], pts2[i, 1] * pts1[i, 1], pts2[i, 1],
                pts1[i, 0], pts1[i, 1], 1]
        
    # 2. Find the SVD of A
    U, S, Vt = svd(A)
    # find the col of V with least singular value
    F = Vt[-1].reshape(3,3)

    # 4. (Enforce rank 2 constraint on F)
    U, S, Vt = svd(F)
    # Set the smallest singular value to zero
    S[2] = 0  
    F = U @ np.diag(S) @ Vt

    # Refine the fundamental matrix using local minimization
    F = refineF(F, pts1, pts2)

    # 5. (Un-normalize F)
    scale = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F = scale.T @ F @ scale


    return F

# optional, use ubtract the mean coordinate then divide by the standard deviation for rescaling for better results
# did not work

# def normalize(pts):
#     # Normalizing points
#     mean = np.mean(pts, axis=0)
#     std = np.std(pts, axis=0)
#     T = np.array([[1/std[0], 0, -mean[0]/std[0]],
#                   [0, 1/std[1], -mean[1]/std[1]],
#                   [0, 0, 1]])
#     pts_normalized = np.dot(T, np.vstack((pts.T, np.ones(pts.shape[0]))))
#     return pts_normalized.T, T

# def eightpoint(pts1, pts2, M):
#     """
#     Implement the eightpoint algorithm in Python
#     Args:
#     pts1 - Nx2 matrix of (x,y) coordinates
#     pts2 - Nx2 matrix of (x,y) coordinates
#     M    - max(imwidth, imheight)
#     Returns:
#     F - The computed fundamental matrix
#     """

#     # Normalize the points
#     p1_normalized, T1 = normalize(pts1)
#     p2_normalized, T2 = normalize(pts2)

#     # Construct matrix for linear system using all points
#     x1, y1 = p1_normalized[:, 0], p1_normalized[:, 1]
#     x2, y2 = p2_normalized[:, 0], p2_normalized[:, 1]
#     A = np.vstack([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(len(x1))]).T

#     # Compute the fundamental matrix
#     _, _, Vt = svd(A)
#     F = Vt[-1].reshape(3, 3)

#     # Enforce rank-2 constraint
#     U, S, Vt = svd(F)
#     S[2] = 0
#     F = np.dot(U, np.dot(np.diag(S), Vt))

#     # Denormalize
#     F = np.dot(T2.T, np.dot(F, T1))

#     # Refinement step (optional, depends on the implementation of refineF)
#     F = refineF(F, pts1, pts2)

#     return F

