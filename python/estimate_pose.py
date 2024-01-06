import numpy as np
from scipy.linalg import svd

def estimate_pose(x, X):
    """
    computes the pose matrix (camera matrix) P given 2D and 3D
    points.
    
    Args:
        x: 2D points with shape [2, N]
        X: 3D points with shape [3, N]
    """
    P = None
    # Number of points
    num_points = x.shape[1]

    # Initialize the matrix A which we'll use to solve the homogeneous equation system A * p = 0
    A = np.zeros((2 * num_points, 12))

    # Build the matrix A row by row
    for i in range(num_points):
        # Current point in homogeneous coordinates
        X_hom = np.append(X[:, i], 1)
        x_hom = np.append(x[:, i], 1)

        # Fill in the rows of A corresponding to the current point
        A[2*i] = np.concatenate(( -X_hom, np.zeros(4), x_hom[0] * X_hom ))
        A[2*i + 1] = np.concatenate(( np.zeros(4), -X_hom, x_hom[1] * X_hom ))

    # Apply Singular Value Decomposition to A
    # The solution to A * p = 0 is the last column of V (or the last row of V transposed)
    U, S, V_transposed = svd(A)

    # Reshape the solution into a 3x4 matrix
    P = V_transposed[-1].reshape(3, 4)

    return P