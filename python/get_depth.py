import numpy as np

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
    creates a depth map from a disparity map (DISPM).
    """
    depthM = np.zeros_like(dispM, dtype=float)

    # Calculate the baseline (distance between the optical centers)
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    b = np.sqrt(np.sum((c1 - c2) ** 2))

    # Focal length from the K matrix
    f = K1[0, 0]

    # Initialize the depth map with zeros
    depthM = np.zeros_like(dispM, dtype=float)

    # Replace all zeros in dispM with np.inf to avoid division by zero
    dispM_replaced = np.where(dispM == 0, np.inf, dispM)

    # Calculate the depth map
    depthM = b * f / dispM_replaced

    # Set depth to 0 where dispM was 0 to preserve the intended meaning
    depthM[dispM == 0] = 0

    return depthM

