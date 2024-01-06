import numpy as np
from scipy.linalg import svd, qr

def rq(A):
    """
    Performs RQ decomposition of a matrix A.
    """
    # Perform QR decomposition on the inverse of A
    Q, R = qr(np.linalg.inv(A))
    
    # Invert the outputs to get RQ
    R = np.linalg.inv(R)
    Q = np.linalg.inv(Q)
    
    # Make sure the diagonal elements of R are positive
    for i in range(3):
        if R[i, i] < 0:
            R[:, i] *= -1
            Q[i, :] *= -1
    
    # R should be upper triangular and Q should be orthonormal
    return R, Q

def estimate_params(P):
    """
    computes the intrinsic K, rotation R, and translation t from
    given camera matrix P.
    
    Args:
        P: Camera matrix
    """
    K, R, t = None, None, None

    # Compute the camera center c by using SVD. 
    # Hint: c is the eigenvector corresponding to the smallest eigenvalue
    # P = K[R|t] and Pc = 0, since the camera center projects to the origin
    U, S, Vt = svd(P)
    c = Vt[-1]
    c = c / c[-1]  # Ensure homogeneity (scale such that the last element is 1)
    c = c[:3] 

    # Compute intrinsic and rotation using RQ decomposition
    K, R = rq(P[:, :3])
    if np.linalg.det(R) < 0:
        R = -R

    # Compute the translation t by t = -Rc
    t = -np.dot(R, c)

    
    return K, R, t

